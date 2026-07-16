"""Prospective IMDb untouched-test role-reversal server ablation."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.classification_task import (  # noqa: E402
    audit_probability_payload, canonical_sha256, load_task_spec,
    mixture_assignment_sha256)
from kafa_script import run_generic_classification_distillation as core  # noqa: E402


METHODS = {
    "public_anchor_only": {
        "conflict_ratio": 0.4,
        "off_task_ratio": 0.2,
        "selection": "no_client_path",
    },
    "clean_unfiltered_distillation": {
        "conflict_ratio": 0.0,
        "off_task_ratio": 0.0,
        "selection": "all_examples",
    },
    "severe_unfiltered_distillation": {
        "conflict_ratio": 0.4,
        "off_task_ratio": 0.2,
        "selection": "all_examples",
    },
    "severe_routed_distillation": {
        "conflict_ratio": 0.4,
        "off_task_ratio": 0.2,
        "selection": "relative_balanced_routing",
    },
}


def load_jsonl_preserve_rows(path: Path) -> list[dict]:
    """Load every official row, assigning a line-specific evaluation ID."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = json.loads(line)
            rows.append({
                **raw,
                "source_id": canonical_sha256({
                    "line": line_number,
                    "record": raw,
                }),
                "line": line_number,
            })
    return rows


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("name") != "imdb_server_role_reversal_protocol_v1":
        raise ValueError("unexpected IMDb role-reversal protocol")
    return protocol


def validate_contract(args, protocol: dict, target_spec, off_task_spec,
                      manifest: dict) -> dict:
    for relative, expected in protocol["implementation"][
            "file_sha256"].items():
        observed = core.sha256_file(ROOT / relative)
        if observed != expected:
            raise ValueError(f"contract hash mismatch for {relative}: {observed}")
    expected_files = {
        args.manifest: protocol["data"]["manifest_sha256"],
        args.target_train: protocol["data"]["target_train_sha256"],
        args.off_task_train: protocol["data"]["off_task_train_sha256"],
    }
    for path, expected in expected_files.items():
        if core.sha256_file(path) != expected:
            raise ValueError(f"data hash mismatch for {path}")
    if target_spec.sha256 != protocol["task"]["target_spec_semantic_sha256"]:
        raise ValueError("target task semantic hash mismatch")
    if off_task_spec.sha256 != protocol["task"]["off_task_spec_semantic_sha256"]:
        raise ValueError("off-task semantic hash mismatch")
    if manifest["task_spec_semantic_sha256"] != target_spec.sha256:
        raise ValueError("manifest task semantic link mismatch")
    if manifest["datasets"]["target_official_test"]["sha256"] != protocol[
            "data"]["official_test_sha256"]:
        raise ValueError("manifest official-test hash mismatch")
    for relative, expected in protocol["model_weight_sha256"].items():
        if core.sha256_file(ROOT / relative) != expected:
            raise ValueError(f"model weight mismatch for {relative}")
    phase = "development" if args.evaluation_split == "development" else \
        "official_evaluation"
    if args.seed not in protocol[phase]["seeds"]:
        raise ValueError(f"seed {args.seed} is not frozen for {phase}")
    if args.method not in protocol[phase]["comparators"]:
        raise ValueError(f"method {args.method} is not frozen for {phase}")
    if METHODS[args.method] != protocol["method_configuration"][args.method]:
        raise ValueError("method configuration differs from protocol")
    if args.evaluation_split == "development":
        if args.official_test.exists():
            raise ValueError(
                "development requires a nonexistent official-test sentinel")
        if str(args.official_test) != protocol["development"][
                "nonexistent_test_sentinel"]:
            raise ValueError("development sentinel path differs from protocol")
    else:
        if str(args.official_test) != protocol["data"]["official_test_path"]:
            raise ValueError("official-test path differs from protocol")
    return protocol["fixed_configuration"]


def train_server_round(server, tokenizer, optimizer, anchors, task_spec,
                       teacher, fixed: dict, seed: int, round_index: int,
                       device) -> float:
    order = torch.randperm(len(anchors), generator=torch.Generator().manual_seed(
        seed + round_index * 1009))
    cursor, objectives = 0, []
    server.train()
    for step_index in range(fixed["server_steps_per_round"]):
        batch_size = fixed["server_batch_size"]
        indices = order[cursor:cursor + batch_size]
        if len(indices) < batch_size:
            order = torch.randperm(
                len(anchors), generator=torch.Generator().manual_seed(
                    seed + round_index * 1009 + step_index + 1))
            cursor = 0
            indices = order[:batch_size]
        cursor += batch_size
        rows = [anchors[int(index)] for index in indices]
        hard = torch.tensor([int(row[task_spec.label_field]) for row in rows],
                            device=device)
        optimizer.zero_grad(set_to_none=True)
        losses = core.candidate_loss_matrix(
            server, tokenizer, rows, task_spec, device, fixed["max_length"],
            batch_size, requires_grad=True)
        objective = fixed["anchor_supervision_weight"] * F.cross_entropy(
            -losses, hard)
        if teacher is not None:
            target = teacher[indices].to(device)
            distill = -(target * F.log_softmax(-losses, dim=1)).sum(1).mean()
            objective = objective + fixed["distillation_weight"] * distill
        objective.backward()
        optimizer.step()
        objectives.append(float(objective.detach()))
    return sum(objectives) / len(objectives)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path(
        "kafa_script/imdb_server_role_reversal_protocol_v1.json"))
    parser.add_argument("--task-spec", type=Path, default=Path(
        "kafa_script/task_specs/imdb_sentiment_server_v1.json"))
    parser.add_argument("--off-task-spec", type=Path, default=Path(
        "kafa_script/task_specs/agnews_offtask_v1.json"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "kafa_script/imdb_server_manifest_v1.json"))
    parser.add_argument("--target-train", type=Path, default=Path(
        ".cache/datasets/imdb/train.jsonl"))
    parser.add_argument("--official-test", type=Path, required=True)
    parser.add_argument("--off-task-train", type=Path, default=Path(
        ".cache/datasets/agnews/train.jsonl"))
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--evaluation-split", required=True,
                        choices=["development", "official_test"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required for the frozen benchmark")

    protocol = load_protocol(args.protocol)
    target_spec = load_task_spec(args.task_spec)
    off_task_spec = load_task_spec(args.off_task_spec)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    fixed = validate_contract(args, protocol, target_spec, off_task_spec,
                              manifest)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    started_utc = datetime.now(timezone.utc).isoformat()

    target_rows = core.load_jsonl(args.target_train)
    partitions = core.resolve_manifest_splits(
        target_rows, manifest,
        ["anchor", "development", "client_0", "client_1", "client_2"])
    anchors = partitions["anchor"]
    if args.evaluation_split == "development":
        evaluation_rows = partitions["development"]
        test_hash_source = "frozen_manifest_not_opened"
    else:
        if core.sha256_file(args.official_test) != protocol["data"][
                "official_test_sha256"]:
            raise ValueError("official-test hash mismatch")
        evaluation_rows = load_jsonl_preserve_rows(args.official_test)
        if len(evaluation_rows) != fixed["official_test_examples"]:
            raise ValueError("official-test count mismatch")
        test_hash_source = "verified_raw_bytes"
    client_bases = [partitions[f"client_{index}"] for index in range(3)]
    off_task_rows = core.load_jsonl(args.off_task_train)
    config = METHODS[args.method]
    mixtures = core.build_client_mixtures(
        client_bases, off_task_rows, target_spec, off_task_spec, args.seed,
        config["conflict_ratio"], config["off_task_ratio"])

    server_path = ROOT / fixed["server_model"]
    client_paths = [ROOT / path for path in fixed["client_models"]]
    server, server_tokenizer = core.make_lora(server_path, device)
    server_optimizer = torch.optim.AdamW(
        [parameter for parameter in server.parameters()
         if parameter.requires_grad], lr=fixed["server_learning_rate"])
    client_states = [None] * 3
    round_records, total_payload = [], 0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for round_index in range(fixed["rounds"]):
        probability_matrices, client_records = [], []
        execute_clients = config["selection"] != "no_client_path"
        iterator = enumerate(zip(client_paths, mixtures)) if execute_clients else []
        for client_index, (path, mixture) in iterator:
            model, tokenizer = core.make_lora(path, device)
            if client_states[client_index] is not None:
                core.set_peft_model_state_dict(model, client_states[client_index])
            if config["selection"] == "relative_balanced_routing":
                selected, selection = core.route_client_mixture(
                    model, tokenizer, mixture, anchors, target_spec,
                    off_task_spec, device, fixed["max_length"],
                    fixed["routing_keep_fraction"],
                    fixed["routing_gate_threshold"])
                selection = {"policy": "relative_balanced_routing", **selection}
            else:
                selected = mixture
                selection = {"policy": "all_examples", "selected": len(selected),
                             "gate_triggered": False}
            mean_loss = core.train_client(
                model, tokenizer, selected, target_spec, off_task_spec, device,
                fixed["client_steps_per_round"], fixed["client_batch_size"],
                fixed["max_length"], fixed["client_learning_rate"],
                args.seed * 100000 + round_index * 100 + client_index)
            model.eval()
            losses = core.candidate_loss_matrix(
                model, tokenizer, anchors, target_spec, device,
                fixed["max_length"], fixed["anchor_score_batch_size"])
            probabilities = torch.softmax(-losses.float(), dim=1).cpu()
            probability_payload = audit_probability_payload(
                probabilities, expected_rows=len(anchors),
                num_classes=target_spec.num_classes)
            wire_payload = core.serialized_probability_payload(probabilities)
            client_states[client_index] = core.clone_adapter_state(model)
            probability_matrices.append(probabilities)
            client_records.append({
                "client_index": client_index,
                "model": str(path.relative_to(ROOT)),
                "model_artifacts": core.artifact_hashes(path),
                "base_target_ids_sha256": core.adapter_record_ids(
                    client_bases[client_index]),
                "mixture_assignment_sha256": mixture_assignment_sha256(mixture),
                "selected_ids_sha256": core.adapter_record_ids(selected),
                "mixture_kind_counts": dict(sorted(Counter(
                    item["kind"] for item in mixture).items())),
                "selected_kind_counts_audit_only": dict(sorted(Counter(
                    item["kind"] for item in selected).items())),
                "mean_local_loss": mean_loss,
                "selection": selection,
                "probability_payload": probability_payload,
                "serialized_probability_payload": wire_payload,
                "adapter_bytes": core.state_bytes(client_states[client_index]),
            })
            del model
            torch.cuda.empty_cache()
        teacher = (sum(probability_matrices) / len(probability_matrices)
                   if probability_matrices else None)
        total_payload += sum(
            record["serialized_probability_payload"]["payload_bytes"]
            for record in client_records)
        objective = train_server_round(
            server, server_tokenizer, server_optimizer, anchors, target_spec,
            teacher, fixed, args.seed, round_index, device)
        round_records.append({
            "round": round_index + 1,
            "client_records": client_records,
            "teacher_weights": ([] if not probability_matrices else
                                [1 / len(probability_matrices)] *
                                len(probability_matrices)),
            "mean_server_objective": objective,
        })

    final = core.evaluate(server, server_tokenizer, evaluation_rows,
                          target_spec, device, fixed["max_length"],
                          fixed["evaluation_batch_size"])
    elapsed = time.perf_counter() - started
    full_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    result = {
        "schema_version": "imdb_server_role_reversal_v1",
        "evidence_label": ("prospective_role_reversal_development" if
                           args.evaluation_split == "development" else
                           "prospective_untouched_test_role_reversal"),
        "paper_result": args.evaluation_split == "official_test",
        "task_id": target_spec.task_id,
        "method": args.method,
        "method_configuration": config,
        "seed": args.seed,
        "started_utc": started_utc,
        "protocol_sha256": core.sha256_file(args.protocol),
        "runner_file_sha256": core.sha256_file(Path(__file__)),
        "classification_task_file_sha256": core.sha256_file(
            ROOT / "kafa_script/classification_task.py"),
        "task_spec_semantic_sha256": target_spec.sha256,
        "off_task_spec_semantic_sha256": off_task_spec.sha256,
        "manifest_sha256": core.sha256_file(args.manifest),
        "full_args": full_args,
        "full_args_sha256": canonical_sha256(full_args),
        "fixed_configuration": fixed,
        "git": core.git_state(),
        "datasets": {
            "target_train_sha256": core.sha256_file(args.target_train),
            "target_official_test_sha256": protocol["data"][
                "official_test_sha256"],
            "target_official_test_hash_source": test_hash_source,
            "off_task_train_sha256": core.sha256_file(args.off_task_train),
        },
        "model_artifacts": {
            "server": core.artifact_hashes(server_path),
            "clients": [core.artifact_hashes(path) for path in client_paths],
        },
        "split_source_ids_sha256": {
            "anchor": core.adapter_record_ids(anchors),
            "evaluation": core.adapter_record_ids(evaluation_rows),
            **{f"client_{index}": core.adapter_record_ids(base)
               for index, base in enumerate(client_bases)},
        },
        "mixture_assignment_sha256": [
            mixture_assignment_sha256(mixture) for mixture in mixtures],
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
    if not all(math.isfinite(float(final[key])) for key in
               ("accuracy", "macro_f1", "canonical_label_nll", "ece")):
        raise ValueError("non-finite final metric")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
