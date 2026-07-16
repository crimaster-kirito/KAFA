"""Auditable heterogeneous canonical-probability distillation for TREC.

The legacy AG News runners remain frozen.  This runner consumes a hashed task
specification and protocol, supports arbitrary K, and fails closed when a
declared configuration or source artifact differs from the protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import (LoraConfig, get_peft_model, get_peft_model_state_dict,
                  set_peft_model_state_dict)
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.classification_task import (  # noqa: E402
    audit_probability_payload, canonical_sha256, load_task_spec,
    mixture_assignment_sha256)
from kafa_script.diagnose_anchor_gradient_separation import (  # noqa: E402
    gradient_cosine, parameter_gradients, rank_percentiles, summary_quantiles)
from kafa_script.run_real_data_plumbing_smoke import collate  # noqa: E402


COARSE_LABELS = ("ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trec_train(path: Path) -> list[dict]:
    """Parse and exact-deduplicate the development source."""
    rows, seen = [], set()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            source_id = hashlib.sha256(raw_line).hexdigest()
            if source_id in seen:
                continue
            seen.add(source_id)
            rows.append(_parse_trec_line(raw_line, line_number, source_id))
    return rows


def load_trec_official(path: Path) -> list[dict]:
    """Parse the official set.  Callers must enforce official-only access."""
    rows = []
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            source_id = hashlib.sha256(raw_line).hexdigest()
            rows.append(_parse_trec_line(raw_line, line_number, source_id))
    return rows


def _parse_trec_line(raw_line: bytes, line_number: int, source_id: str) -> dict:
    content = raw_line.rstrip(b"\r\n")
    try:
        label_token, question = content.split(b" ", 1)
        coarse = label_token.split(b":", 1)[0].decode("ascii")
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError(f"malformed TREC line {line_number}") from error
    if coarse not in COARSE_LABELS:
        raise ValueError(f"unknown TREC coarse label {coarse!r}")
    return {
        "source_id": source_id,
        "line": line_number,
        "text": question.decode("latin-1"),
        "label": COARSE_LABELS.index(coarse),
    }


def load_jsonl(path: Path) -> list[dict]:
    rows, seen = [], set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            source_id = canonical_sha256(raw)
            if source_id in seen:
                continue
            seen.add(source_id)
            rows.append({**raw, "source_id": source_id})
    return rows


def select_manifest_rows(rows: list[dict], manifest: dict, split: str) -> list[dict]:
    by_id = {row["source_id"]: row for row in rows}
    split_record = manifest["splits"][split]
    records = split_record["records"]
    requested = [record["source_id"] for record in records]
    missing = [source_id for source_id in requested if source_id not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} manifest records missing from {split}")
    if split_record["count"] != len(records):
        raise ValueError(f"manifest count differs from records in {split}")
    declared_hash = hashlib.sha256(
        "\n".join(requested).encode("ascii")).hexdigest()
    if split_record["source_ids_sha256"] != declared_hash:
        raise ValueError(f"manifest source-ID hash differs in {split}")
    selected = [by_id[source_id] for source_id in requested]
    if len({row["source_id"] for row in selected}) != len(selected):
        raise ValueError(f"duplicate records in manifest split {split}")
    for record, row in zip(records, selected):
        if int(record["label"]) != int(row["label"]):
            raise ValueError(f"manifest label differs from source in {split}")
        if int(record["line"]) != int(row["line"]):
            raise ValueError(f"manifest line differs from source in {split}")
    return selected


def resolve_manifest_splits(rows: list[dict], manifest: dict,
                            split_names: list[str]) -> dict[str, list[dict]]:
    """Resolve and jointly verify content-disjoint manifest partitions."""
    resolved = {name: select_manifest_rows(rows, manifest, name)
                for name in split_names}
    owners = {}
    for name, split_rows in resolved.items():
        for row in split_rows:
            source_id = row["source_id"]
            if source_id in owners:
                raise ValueError(
                    f"manifest split overlap: {source_id} occurs in "
                    f"{owners[source_id]} and {name}")
            owners[source_id] = name
    return resolved


def stable_rank(namespace: str, seed: int, source_id: str) -> str:
    return hashlib.sha256(
        f"trec-server-v1|{namespace}|{seed}|{source_id}".encode("utf-8")
    ).hexdigest()


def build_client_mixtures(client_bases: list[list[dict]], off_task_rows: list[dict],
                          target_spec, off_task_spec, seed: int,
                          conflict_ratio: float, off_task_ratio: float) \
        -> list[list[dict]]:
    """Construct deterministic mixtures with disjoint off-task records."""
    counts = [round(len(base) * off_task_ratio) for base in client_bases]
    ranked_off_task = sorted(
        off_task_rows,
        key=lambda row: (stable_rank("off-task", seed, row["source_id"]),
                         row["source_id"]),
    )
    if len(ranked_off_task) < sum(counts):
        raise ValueError("insufficient disjoint off-task rows")
    offset, mixtures = 0, []
    for client_index, (base, n_off_task) in enumerate(zip(client_bases, counts)):
        n_conflict = round(len(base) * conflict_ratio)
        if n_off_task + n_conflict >= len(base):
            raise ValueError("conflict and off-task counts must sum below client size")
        grouped_positions = defaultdict(list)
        for index, row in enumerate(base):
            grouped_positions[int(row[target_spec.label_field])].append(index)
        if set(grouped_positions) != set(range(target_spec.num_classes)):
            raise ValueError("client base does not cover every target class")
        def apportioned(total: int, kind: str) -> dict[int, int]:
            quotient, remainder = divmod(total, target_spec.num_classes)
            order = sorted(
                range(target_spec.num_classes),
                key=lambda label: stable_rank(
                    f"apportion-{kind}-{client_index}", seed, str(label)))
            return {label: quotient + int(label in order[:remainder])
                    for label in range(target_spec.num_classes)}

        off_quota = apportioned(n_off_task, "off-task")
        conflict_quota = apportioned(n_conflict, "conflict")
        off_positions, conflict_positions = set(), set()
        for label in range(target_spec.num_classes):
            positions = sorted(
                grouped_positions[label],
                key=lambda index: (
                    stable_rank(f"position-{client_index}-class-{label}", seed,
                                base[index]["source_id"]),
                    base[index]["source_id"],
                ),
            )
            off_per_class = off_quota[label]
            conflict_per_class = conflict_quota[label]
            if len(positions) < off_per_class + conflict_per_class:
                raise ValueError(f"class {label} has insufficient mixture rows")
            off_positions.update(positions[:off_per_class])
            conflict_positions.update(
                positions[off_per_class:off_per_class + conflict_per_class])
        chosen_off_task = iter(ranked_off_task[offset:offset + n_off_task])
        offset += n_off_task
        mixture = []
        for index, row in enumerate(base):
            if index in off_positions:
                source = next(chosen_off_task)
                label = int(source[off_task_spec.label_field])
                mixture.append({
                    "kind": "off_task",
                    "source_id": source["source_id"],
                    "text": source[off_task_spec.text_field],
                    "answer": off_task_spec.verbalizer(label),
                })
            else:
                original = int(row[target_spec.label_field])
                training = (target_spec.conflicting_label(original)
                            if index in conflict_positions else original)
                mixture.append({
                    "kind": "conflict" if index in conflict_positions else "clean",
                    "source_id": row["source_id"],
                    "text": row[target_spec.text_field],
                    "original_label": original,
                    "training_label": training,
                    "answer": target_spec.verbalizer(training),
                })
        mixtures.append(mixture)
    off_ids = [item["source_id"] for mixture in mixtures for item in mixture
               if item["kind"] == "off_task"]
    if len(off_ids) != len(set(off_ids)):
        raise AssertionError("off-task records overlap across clients")
    return mixtures


def encode_prompt_answer(tokenizer, prompt: str, answer: str,
                         max_length: int) -> dict[str, list[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
    if not answer_ids:
        raise ValueError("verbalizer tokenized to an empty sequence")
    if len(answer_ids) >= max_length:
        raise ValueError("max_length must exceed every verbalizer token length")
    prompt_ids = prompt_ids[-(max_length - len(answer_ids)):]
    return {
        "input_ids": prompt_ids + answer_ids,
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


def encode_target(tokenizer, row: dict, spec, max_length: int,
                  label: int | None = None) -> dict:
    label = int(row[spec.label_field]) if label is None else label
    return encode_prompt_answer(
        tokenizer, spec.prompt(row[spec.text_field]), spec.verbalizer(label),
        max_length)


def encode_mixture(tokenizer, item: dict, target_spec, off_task_spec,
                   max_length: int) -> dict:
    spec = off_task_spec if item["kind"] == "off_task" else target_spec
    return encode_prompt_answer(tokenizer, spec.prompt(item["text"]),
                                item["answer"], max_length)


def candidate_loss_matrix(model, tokenizer, rows: list[dict], spec, device,
                          max_length: int, batch_size: int,
                          requires_grad: bool = False) -> torch.Tensor:
    candidates = [encode_target(tokenizer, row, spec, max_length, label)
                  for row in rows for label in range(spec.num_classes)]
    loader = DataLoader(
        candidates, batch_size=batch_size * spec.num_classes, shuffle=False,
        collate_fn=lambda batch: collate(tokenizer, batch))
    chunks = []
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits[:, :-1]
            targets = batch["input_ids"][:, 1:]
            mask = batch["labels"][:, 1:] != -100
            token_losses = F.cross_entropy(logits.transpose(1, 2), targets,
                                           reduction="none")
            chunks.append((token_losses * mask).sum(1) /
                          mask.sum(1).clamp_min(1))
    return torch.cat(chunks).reshape(len(rows), spec.num_classes)


@torch.no_grad()
def evaluate(model, tokenizer, rows: list[dict], spec, device,
             max_length: int, batch_size: int) -> dict:
    model.eval()
    matrix = candidate_loss_matrix(model, tokenizer, rows, spec, device,
                                   max_length, batch_size).float().cpu()
    probabilities = torch.softmax(-matrix, dim=1)
    predictions = probabilities.argmax(1)
    targets = torch.tensor([int(row[spec.label_field]) for row in rows])
    accuracy = float((predictions == targets).float().mean())
    f1s, recalls = [], []
    for label in range(spec.num_classes):
        tp = ((predictions == label) & (targets == label)).sum().float()
        fp = ((predictions == label) & (targets != label)).sum().float()
        fn = ((predictions != label) & (targets == label)).sum().float()
        precision = tp / (tp + fp).clamp_min(1)
        recall = tp / (tp + fn).clamp_min(1)
        f1s.append(float(2 * precision * recall /
                         (precision + recall).clamp_min(1e-12)))
        recalls.append(float(recall))
    confidence = probabilities.max(1).values
    correct = (predictions == targets).float()
    ece = 0.0
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        mask = ((confidence >= lower) &
                (confidence <= upper if index == 9 else confidence < upper))
        if mask.any():
            ece += float(mask.float().mean() *
                         (correct[mask].mean() - confidence[mask].mean()).abs())
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1s) / len(f1s),
        "canonical_label_nll": float(
            -probabilities[torch.arange(len(rows)), targets]
            .clamp_min(1e-12).log().mean()),
        "target_answer_loss": float(
            matrix[torch.arange(len(rows)), targets].mean()),
        "ece": ece,
        "per_class_recall": recalls,
        "confusion_matrix": [[int(((targets == true) & (predictions == pred)).sum())
                              for pred in range(spec.num_classes)]
                             for true in range(spec.num_classes)],
    }


def make_lora(model_path: Path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")).to(device)
    return model, tokenizer


def clone_adapter_state(model) -> dict[str, torch.Tensor]:
    return {key: value.detach().float().cpu().clone()
            for key, value in get_peft_model_state_dict(model).items()}


def state_bytes(state: dict[str, torch.Tensor]) -> int:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.tell()


def serialized_probability_payload(probabilities: torch.Tensor) -> dict:
    """Describe the exact tensor serialization used for communication cost."""
    buffer = io.BytesIO()
    torch.save(probabilities, buffer)
    payload = buffer.getvalue()
    return {
        "encoding": "torch-save-tensor",
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }


def train_client(model, tokenizer, mixture: list[dict], target_spec,
                 off_task_spec, device, steps: int, batch_size: int,
                 max_length: int, learning_rate: float, seed: int) -> float:
    encoded = [encode_mixture(tokenizer, item, target_spec, off_task_spec,
                              max_length) for item in mixture]
    loader = DataLoader(
        encoded, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=lambda batch: collate(tokenizer, batch))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate)
    iterator, losses = iter(loader), []
    model.train()
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return sum(losses) / len(losses)


def route_client_mixture(model, tokenizer, mixture: list[dict],
                         anchors: list[dict], target_spec, off_task_spec,
                         device, max_length: int, keep_fraction: float,
                         gate_threshold: float) -> tuple[list[dict], dict]:
    """Route without consulting hidden kind/original labels for scoring."""
    parameters = [parameter for parameter in model.parameters()
                  if parameter.requires_grad]
    grouped = defaultdict(list)
    for row in anchors:
        grouped[int(row[target_spec.label_field])].append(row)
    if set(grouped) != set(range(target_spec.num_classes)):
        raise ValueError("anchors do not cover every class")
    anchor_gradients = {}
    for label in range(target_spec.num_classes):
        batch = collate(tokenizer, [encode_target(
            tokenizer, row, target_spec, max_length) for row in grouped[label]])
        batch = {key: value.to(device) for key, value in batch.items()}
        model.zero_grad(set_to_none=True)
        model(**batch).loss.backward()
        anchor_gradients[label] = parameter_gradients(parameters)
    reference = []
    for label in range(target_spec.num_classes):
        for row in grouped[label]:
            batch = collate(tokenizer, [encode_target(
                tokenizer, row, target_spec, max_length)])
            batch = {key: value.to(device) for key, value in batch.items()}
            model.zero_grad(set_to_none=True)
            model(**batch).loss.backward()
            reference.append(gradient_cosine(parameter_gradients(parameters),
                                              anchor_gradients[label]))
    target_items = [item for item in mixture if item["kind"] != "off_task"]
    target_losses = candidate_loss_matrix(
        model, tokenizer,
        [{"text": item["text"], "label": item["training_label"]}
         for item in target_items],
        target_spec, device, max_length, 8).float().cpu()
    target_loss_by_id = {item["source_id"]: losses for item, losses in
                         zip(target_items, target_losses)}
    records = []
    for item in mixture:
        encoded = encode_mixture(tokenizer, item, target_spec, off_task_spec,
                                 max_length)
        batch = collate(tokenizer, [encoded])
        batch = {key: value.to(device) for key, value in batch.items()}
        model.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        gradient = parameter_gradients(parameters)
        if item["kind"] == "off_task":
            cosine = max(gradient_cosine(gradient, anchor_gradients[label])
                         for label in range(target_spec.num_classes))
            claimed_label, margin = None, None
        else:
            claimed_label = int(item["training_label"])
            cosine = gradient_cosine(gradient, anchor_gradients[claimed_label])
            losses = target_loss_by_id[item["source_id"]]
            margin = float(losses[claimed_label] - losses.min())
        records.append({
            "source_id": item["source_id"],
            "claimed_label": claimed_label,
            "loss": float(loss.detach()),
            "anchor_cosine": cosine,
            "semantic_margin": margin,
        })
    client_q25 = summary_quantiles(
        [record["anchor_cosine"] for record in records])["q25"]
    reference_q25 = summary_quantiles(reference)["q25"]
    ratio = client_q25 / max(reference_q25, 1e-12)
    common = {
        "gate_ratio": ratio,
        "client_q25": client_q25,
        "reference_q25": reference_q25,
        "gate_threshold": gate_threshold,
    }
    if ratio >= gate_threshold:
        return mixture, {**common, "gate_triggered": False,
                         "selected": len(mixture)}
    semantic = [record["semantic_margin"] for record in records
                if record["semantic_margin"] is not None]
    off_task_value = max(semantic, default=0.0) + 1.0
    signals = [
        rank_percentiles([-record["anchor_cosine"] for record in records]),
        rank_percentiles([record["loss"] for record in records]),
        rank_percentiles([off_task_value if record["semantic_margin"] is None
                          else record["semantic_margin"] for record in records]),
    ]
    scores = [sum(parts) / len(parts) for parts in zip(*signals)]
    quota = max(1, round(len(mixture) * keep_fraction /
                         target_spec.num_classes))
    selected_indices = set()
    for label in range(target_spec.num_classes):
        candidates = [index for index, record in enumerate(records)
                      if record["claimed_label"] == label]
        if len(candidates) < quota:
            raise ValueError(f"class {label} has fewer than routing quota {quota}")
        candidates.sort(key=lambda index: (scores[index],
                                           records[index]["source_id"]))
        selected_indices.update(candidates[:quota])
    selected = [item for index, item in enumerate(mixture)
                if index in selected_indices]
    return selected, {
        **common,
        "gate_triggered": True,
        "selected": len(selected),
        "quota_per_claimed_label": quota,
        "off_task_selected": 0,
        "score_record_order_sha256": canonical_sha256(
            [record["source_id"] for record in records]),
    }


def artifact_hashes(model_path: Path) -> dict:
    result = {}
    for name in ("model.safetensors", "config.json", "tokenizer.json",
                 "tokenizer.model", "tokenizer_config.json", "vocab.json",
                 "merges.txt"):
        path = model_path / name
        if path.exists():
            result[name] = sha256_file(path)
    if "model.safetensors" not in result:
        raise ValueError(f"missing model.safetensors in {model_path}")
    return result


def git_state() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            text=True, capture_output=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True,
            text=True, capture_output=True).stdout)
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"commit": commit, "dirty": dirty}


def validate_protocol(args, protocol: dict, manifest: dict, task_spec,
                      off_task_spec) -> None:
    task = protocol["task"]
    implementation = protocol["implementation"]
    if sha256_file(Path(__file__)) != implementation["runner_file_sha256"]:
        raise ValueError("runner file hash differs from protocol")
    task_contract_path = ROOT / "kafa_script/classification_task.py"
    if sha256_file(task_contract_path) != \
            implementation["classification_task_file_sha256"]:
        raise ValueError("classification-task core hash differs from protocol")
    if task_spec.task_id != task["task_id"]:
        raise ValueError("task_id differs from protocol")
    if task_spec.sha256 != task["task_spec_semantic_sha256"]:
        raise ValueError("task-spec semantic hash differs from protocol")
    if off_task_spec.sha256 != task["off_task_spec_semantic_sha256"]:
        raise ValueError("off-task-spec semantic hash differs from protocol")
    if sha256_file(args.manifest) != task["manifest_file_sha256"]:
        raise ValueError("manifest file hash differs from protocol")
    expected_hashes = {
        args.target_train: task["target_train_sha256"],
        args.off_task_train: task["off_task_train_sha256"],
    }
    if manifest["datasets"]["target_official_test"]["sha256"] != \
            task["target_official_test_sha256"]:
        raise ValueError("official-test manifest hash differs from protocol")
    # The development branch must not open the official test, even to rehash
    # it.  The official branch verifies raw bytes immediately before parsing.
    if args.evaluation_split == "official_test":
        expected_hashes[args.official_test] = task["target_official_test_sha256"]
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"dataset hash differs from protocol: {path}")
    fixed = protocol["fixed_configuration"]
    observed = observed_fixed_configuration(args, manifest, task_spec)
    validate_fixed_configuration(observed, fixed)
    section = "development" if args.evaluation_split == "development" \
        else "official_evaluation"
    if args.seed not in protocol[section]["seeds"]:
        raise ValueError(f"seed {args.seed} is not frozen for {section}")
    if args.method not in protocol[section]["comparators"]:
        raise ValueError(f"method {args.method} is not frozen for {section}")


def observed_fixed_configuration(args, manifest: dict, task_spec) -> dict:
    return {
        "server_model": str(args.server_model),
        "client_models": [str(path) for path in args.client_models],
        "rounds": args.rounds,
        "client_steps_per_round": args.client_steps,
        "server_steps_per_round": args.server_steps,
        "client_batch_size": args.batch_size,
        "server_batch_size": args.server_batch_size,
        "evaluation_batch_size": args.eval_batch_size,
        "anchor_score_batch_size": args.anchor_score_batch_size,
        "client_learning_rate": args.learning_rate,
        "server_learning_rate": args.server_learning_rate,
        "max_length": args.max_length,
        "teacher_aggregation": "uniform",
        "teacher_temperature": args.teacher_temperature,
        "anchor_supervision_weight": args.anchor_supervision_weight,
        "routed_distillation_weight": args.distillation_weight,
        "conflict_ratio": args.conflict_ratio,
        "off_task_ratio": args.off_task_ratio,
        "routing_keep_fraction": args.routing_keep_fraction,
        "routing_gate_threshold": args.routing_gate_threshold,
        "contamination_assignment": "class_balanced_deterministic_apportionment",
        "routing_quota_per_claimed_class": round(
            manifest["splits"]["client_0"]["count"] *
            args.routing_keep_fraction / task_spec.num_classes),
        "anchor_examples": manifest["splits"]["anchor"]["count"],
        "development_examples": manifest["splits"]["development"]["count"],
        "client_examples_each": manifest["splits"]["client_0"]["count"],
        "official_test_examples": manifest["datasets"]["target_official_test"]["count"],
        "conflict_permutation": list(task_spec.conflict_permutation),
    }


def validate_fixed_configuration(observed: dict, fixed: dict) -> None:
    if observed != fixed:
        differences = {key: {"expected": fixed.get(key), "observed": value}
                       for key, value in observed.items()
                       if fixed.get(key) != value}
        differences.update({key: {"expected": value, "observed": None}
                            for key, value in fixed.items()
                            if key not in observed})
        raise ValueError(f"fixed configuration mismatch: {differences}")


def load_evaluation_rows(evaluation_split: str, development_rows: list[dict],
                         official_path: Path, expected_sha256: str,
                         expected_count: int, *, hash_file=sha256_file,
                         official_loader=load_trec_official) -> list[dict]:
    """Enforce that development never touches official-test raw bytes."""
    if evaluation_split == "development":
        return development_rows
    if evaluation_split != "official_test":
        raise ValueError(f"unknown evaluation split: {evaluation_split}")
    if hash_file(official_path) != expected_sha256:
        raise ValueError("official-test hash differs from frozen protocol")
    rows = official_loader(official_path)
    if len(rows) != expected_count:
        raise ValueError(
            f"official-test count differs: expected {expected_count}, "
            f"got {len(rows)}")
    return rows


def adapter_record_ids(items: list[dict]) -> str:
    return canonical_sha256([item["source_id"] for item in items])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-spec", type=Path, default=Path(
        "kafa_script/task_specs/trec_coarse_server_v1.json"))
    parser.add_argument("--off-task-spec", type=Path, default=Path(
        "kafa_script/task_specs/imdb_sentiment_server_v1.json"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "kafa_script/trec_server_manifest_v1.json"))
    parser.add_argument("--protocol", type=Path, default=Path(
        "kafa_script/trec_server_transfer_protocol_v1.json"))
    parser.add_argument("--target-train", type=Path, default=Path(
        ".cache/datasets/trec/train_5500.label"))
    parser.add_argument("--official-test", type=Path, default=Path(
        ".cache/datasets/trec/TREC_10.label"))
    parser.add_argument("--off-task-train", type=Path, default=Path(
        ".cache/datasets/imdb/train.jsonl"))
    parser.add_argument("--server-model", type=Path, default=Path(
        ".cache/models/Qwen2.5-1.5B"))
    parser.add_argument("--client-models", type=Path, nargs="+", default=[
        Path(".cache/models/Qwen2.5-0.5B"),
        Path(".cache/models/SmolLM2-360M"),
        Path(".cache/models/TinyLlama-1.1B-Chat-v1.0")])
    parser.add_argument("--method", required=True,
                        choices=["public_anchor_only", "routed_distillation"])
    parser.add_argument("--evaluation-split", required=True,
                        choices=["development", "official_test"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--client-steps", type=int, default=40)
    parser.add_argument("--server-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--server-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--anchor-score-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--server-learning-rate", type=float, default=1e-4)
    parser.add_argument("--anchor-supervision-weight", type=float, default=1.0)
    parser.add_argument("--distillation-weight", type=float, default=0.25)
    parser.add_argument("--teacher-temperature", type=float, default=0.1)
    parser.add_argument("--routing-keep-fraction", type=float, default=0.5)
    parser.add_argument("--routing-gate-threshold", type=float, default=0.8)
    parser.add_argument("--conflict-ratio", type=float, default=0.4)
    parser.add_argument("--off-task-ratio", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is required for the frozen benchmark")
    if len(args.client_models) != 3:
        parser.error("the frozen protocol requires exactly three clients")
    if min(args.rounds, args.client_steps, args.server_steps,
           args.batch_size, args.server_batch_size, args.eval_batch_size,
           args.anchor_score_batch_size, args.max_length) <= 0:
        parser.error("step, batch, round, and length values must be positive")
    if not 0 < args.routing_keep_fraction <= 1:
        parser.error("routing keep fraction must be in (0, 1]")

    task_spec = load_task_spec(args.task_spec)
    off_task_spec = load_task_spec(args.off_task_spec)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validate_protocol(args, protocol, manifest, task_spec, off_task_spec)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    started_utc = datetime.now(timezone.utc).isoformat()
    full_args = {key: ([str(item) for item in value] if isinstance(value, list)
                       else str(value) if isinstance(value, Path) else value)
                 for key, value in vars(args).items()}
    target_rows = load_trec_train(args.target_train)
    partitions = resolve_manifest_splits(
        target_rows, manifest,
        ["anchor", "development", "client_0", "client_1", "client_2"])
    anchors = partitions["anchor"]
    evaluation_rows = load_evaluation_rows(
        args.evaluation_split, partitions["development"], args.official_test,
        protocol["task"]["target_official_test_sha256"],
        protocol["fixed_configuration"]["official_test_examples"])
    client_bases = [partitions[f"client_{index}"] for index in range(3)]
    off_task_rows = load_jsonl(args.off_task_train)
    mixtures = build_client_mixtures(
        client_bases, off_task_rows, task_spec, off_task_spec, args.seed,
        args.conflict_ratio, args.off_task_ratio)

    server, server_tokenizer = make_lora(args.server_model, device)
    server_optimizer = torch.optim.AdamW(
        [parameter for parameter in server.parameters() if parameter.requires_grad],
        lr=args.server_learning_rate)
    client_states = [None] * 3
    round_records, total_payload = [], 0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for round_index in range(args.rounds):
        teacher = None
        client_records = []
        if args.method == "routed_distillation":
            probability_matrices = []
            for client_index, (path, mixture) in enumerate(zip(
                    args.client_models, mixtures)):
                model, tokenizer = make_lora(path, device)
                if client_states[client_index] is not None:
                    set_peft_model_state_dict(model, client_states[client_index])
                selected, routing = route_client_mixture(
                    model, tokenizer, mixture, anchors, task_spec,
                    off_task_spec, device, args.max_length,
                    args.routing_keep_fraction, args.routing_gate_threshold)
                mean_loss = train_client(
                    model, tokenizer, selected, task_spec, off_task_spec,
                    device, args.client_steps, args.batch_size, args.max_length,
                    args.learning_rate,
                    args.seed * 100000 + round_index * 100 + client_index)
                model.eval()
                losses = candidate_loss_matrix(
                    model, tokenizer, anchors, task_spec, device,
                    args.max_length, args.anchor_score_batch_size)
                probabilities = torch.softmax(-losses.float(), dim=1).cpu()
                payload = audit_probability_payload(
                    probabilities, expected_rows=len(anchors),
                    num_classes=task_spec.num_classes)
                wire_payload = serialized_probability_payload(probabilities)
                client_states[client_index] = clone_adapter_state(model)
                probability_matrices.append(probabilities)
                client_records.append({
                    "client_index": client_index,
                    "model": str(path),
                    "model_artifacts": artifact_hashes(path),
                    "base_target_ids_sha256": adapter_record_ids(
                        client_bases[client_index]),
                    "mixture_assignment_sha256": mixture_assignment_sha256(mixture),
                    "selected_ids_sha256": adapter_record_ids(selected),
                    "mixture_kind_counts": dict(sorted(Counter(
                        item["kind"] for item in mixture).items())),
                    "selected_kind_counts_audit_only": dict(sorted(Counter(
                        item["kind"] for item in selected).items())),
                    "mean_local_loss": mean_loss,
                    "routing": routing,
                    "probability_payload": payload,
                    "serialized_probability_payload": wire_payload,
                    "adapter_bytes": state_bytes(client_states[client_index]),
                })
                del model
                torch.cuda.empty_cache()
            teacher = sum(probability_matrices) / len(probability_matrices)
            total_payload += sum(record["serialized_probability_payload"]["payload_bytes"]
                                 for record in client_records)

        generator = torch.Generator().manual_seed(args.seed + round_index * 1009)
        order = torch.randperm(len(anchors), generator=generator)
        cursor, objectives = 0, []
        server.train()
        for step_index in range(args.server_steps):
            indices = order[cursor:cursor + args.server_batch_size]
            if len(indices) < args.server_batch_size:
                order = torch.randperm(
                    len(anchors), generator=torch.Generator().manual_seed(
                        args.seed + round_index * 1009 + step_index + 1))
                cursor = 0
                indices = order[:args.server_batch_size]
            cursor += args.server_batch_size
            rows = [anchors[int(index)] for index in indices]
            hard = torch.tensor([int(row[task_spec.label_field]) for row in rows],
                                device=device)
            server_optimizer.zero_grad(set_to_none=True)
            losses = candidate_loss_matrix(
                server, server_tokenizer, rows, task_spec, device,
                args.max_length, args.server_batch_size, requires_grad=True)
            anchored = F.cross_entropy(-losses, hard)
            objective = args.anchor_supervision_weight * anchored
            if teacher is not None:
                target = teacher[indices].to(device)
                distill = -(target * F.log_softmax(-losses, dim=1)).sum(1).mean()
                objective = objective + args.distillation_weight * distill
            objective.backward()
            server_optimizer.step()
            objectives.append(float(objective.detach()))
        round_records.append({
            "round": round_index + 1,
            "client_records": client_records,
            "mean_server_objective": sum(objectives) / len(objectives),
        })

    final = evaluate(server, server_tokenizer, evaluation_rows, task_spec,
                     device, args.max_length, args.eval_batch_size)
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": "2.0",
        "evidence_label": ("benchmark_development" if
                           args.evaluation_split == "development" else
                           "frozen_official_test"),
        "paper_result": args.evaluation_split == "official_test",
        "task_id": task_spec.task_id,
        "method": args.method,
        "seed": args.seed,
        "started_utc": started_utc,
        "protocol_sha256": sha256_file(args.protocol),
        "runner_file_sha256": sha256_file(Path(__file__)),
        "classification_task_file_sha256": sha256_file(
            ROOT / "kafa_script/classification_task.py"),
        "task_spec_semantic_sha256": task_spec.sha256,
        "task_spec_file_sha256": sha256_file(args.task_spec),
        "off_task_spec_semantic_sha256": off_task_spec.sha256,
        "manifest_sha256": sha256_file(args.manifest),
        "full_args": full_args,
        "full_args_sha256": canonical_sha256(full_args),
        "git": git_state(),
        "datasets": {
            "target_train_sha256": sha256_file(args.target_train),
            "target_official_test_sha256": protocol["task"][
                "target_official_test_sha256"],
            "target_official_test_hash_source": (
                "verified_raw_bytes" if args.evaluation_split == "official_test"
                else "frozen_protocol_not_reopened_in_development"),
            "off_task_train_sha256": sha256_file(args.off_task_train),
        },
        "model_artifacts": {
            "server": artifact_hashes(args.server_model),
            "clients": [artifact_hashes(path) for path in args.client_models],
        },
        "split_source_ids_sha256": {
            "anchor": adapter_record_ids(anchors),
            "evaluation": adapter_record_ids(evaluation_rows),
            **{f"client_{index}": adapter_record_ids(base)
               for index, base in enumerate(client_bases)},
        },
        "evaluation_split": args.evaluation_split,
        "evaluation_examples": len(evaluation_rows),
        "anchor_examples": len(anchors),
        "round_records": round_records,
        "final": final,
        "total_client_to_server_payload_bytes": total_payload,
        "wall_seconds": elapsed,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "evidence_boundaries": protocol["evidence_boundaries"],
    }
    if not all(math.isfinite(value) for key, value in final.items()
               if key in {"accuracy", "macro_f1", "canonical_label_nll", "ece"}):
        raise ValueError("non-finite final metric")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
