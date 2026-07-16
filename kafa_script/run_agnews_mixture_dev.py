"""Real-data mechanism study for within-client task/data mixtures."""

import argparse
import hashlib
import io
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.run_agnews_local_dev import (  # noqa: E402
    evaluate, select_client_train, select_manifest_rows)
from kafa_script.run_real_data_plumbing_smoke import (  # noqa: E402
    LABELS, PROMPT, collate, load_jsonl, sha256_file)

SENTIMENT_PROMPT = (
    "Classify the movie review sentiment into exactly one category: "
    "Negative or Positive.\nReview: {text}\nSentiment:"
)
SENTIMENT_LABELS = ("Negative", "Positive")


def stable_rank(namespace, seed, record_id):
    return hashlib.sha256(
        f"mixture-dev-v1|{namespace}|{seed}|{record_id}".encode()).hexdigest()


def select_imdb(rows, seed, count):
    ranked = sorted((stable_rank("imdb", seed, row["_id"]), row)
                    for row in rows)
    return [row for _, row in ranked[:count]]


def build_mixture(ag_rows, imdb_rows, seed, conflict_ratio, off_task_ratio):
    total = len(ag_rows)
    n_conflict = round(total * conflict_ratio)
    n_off_task = round(total * off_task_ratio)
    if n_conflict + n_off_task >= total:
        raise ValueError("conflict and off-task counts must sum below total")
    positions = sorted(range(total), key=lambda index: stable_rank(
        "position", seed, ag_rows[index]["_id"]))
    off_positions = set(positions[:n_off_task])
    conflict_positions = set(positions[n_off_task:n_off_task + n_conflict])
    chosen_imdb = select_imdb(imdb_rows, seed, n_off_task)
    imdb_iterator = iter(chosen_imdb)
    mixture = []
    for index, row in enumerate(ag_rows):
        if index in off_positions:
            source = next(imdb_iterator)
            mixture.append({
                "kind": "off_task",
                "source_id": source["_id"],
                "text": source["text"],
                "answer": SENTIMENT_LABELS[int(source["label"])],
            })
        elif index in conflict_positions:
            original = int(row["label"])
            corrupted = (original + 1) % len(LABELS)
            mixture.append({
                "kind": "conflict",
                "source_id": row["_id"],
                "text": row["text"],
                "original_label": original,
                "training_label": corrupted,
                "answer": LABELS[corrupted],
            })
        else:
            label = int(row["label"])
            mixture.append({
                "kind": "clean",
                "source_id": row["_id"],
                "text": row["text"],
                "original_label": label,
                "training_label": label,
                "answer": LABELS[label],
            })
    return mixture


def encode_mixture_example(tokenizer, item, max_length):
    template = SENTIMENT_PROMPT if item["kind"] == "off_task" else PROMPT
    prompt_ids = tokenizer(template.format(text=item["text"]),
                           add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(" " + item["answer"],
                           add_special_tokens=False)["input_ids"]
    prompt_ids = prompt_ids[-(max_length - len(answer_ids)):]
    return {
        "input_ids": prompt_ids + answer_ids,
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


def rank_percentiles(values):
    count = len(values)
    if count <= 1:
        return [0.0] * count
    order = sorted(range(count), key=values.__getitem__)
    result = [0.0] * count
    begin = 0
    while begin < count:
        end = begin + 1
        while end < count and values[order[end]] == values[order[begin]]:
            end += 1
        percentile = ((begin + end - 1) / 2) / (count - 1)
        for position in range(begin, end):
            result[order[position]] = percentile
        begin = end
    return result


def select_for_training(mixture, selection, keep_fraction, diagnostics_path,
                        gate_quantile=0.25, gate_threshold=0.7):
    if selection == "none":
        return mixture, None
    if selection == "oracle_clean":
        selected = [item for item in mixture if item["kind"] == "clean"]
        return selected, {
            "uses_hidden_mixture_labels": True,
            "purpose": "unattainable clean-only upper reference",
        }
    if diagnostics_path is None:
        raise ValueError(f"{selection} selection requires --diagnostics")
    diagnostics = json.loads(diagnostics_path.read_text())
    records = diagnostics["records"]
    by_id = {record["source_id"]: record for record in records}
    mixture_ids = [item["source_id"] for item in mixture]
    if len(by_id) != len(records) or set(by_id) != set(mixture_ids):
        raise ValueError("diagnostic records do not match this mixture")
    ordered = [by_id[record_id] for record_id in mixture_ids]
    semantic_values = [record["semantic_inconsistency_margin"]
                       for record in ordered
                       if record["semantic_inconsistency_margin"] is not None]
    off_task_semantic = max(semantic_values, default=0.0) + 1.0
    signals = [
        rank_percentiles([-record["anchor_cosine"] for record in ordered]),
        rank_percentiles([record["loss"] for record in ordered]),
        rank_percentiles([
            off_task_semantic if record["semantic_inconsistency_margin"] is None
            else record["semantic_inconsistency_margin"]
            for record in ordered]),
    ]
    scores = [sum(parts) / len(parts) for parts in zip(*signals)]
    keep = max(1, round(len(mixture) * keep_fraction))
    gate_values = sorted(record["anchor_cosine"] for record in ordered)
    gate_index = round(gate_quantile * (len(gate_values) - 1))
    raw_gate_statistic = gate_values[gate_index]
    if selection == "adaptive_balanced_relative_rank":
        reference = diagnostics["anchor_reference_cosine"]["q25"]
        gate_statistic = raw_gate_statistic / max(reference, 1e-12)
        gate_name = "client_q25_over_anchor_reference_q25"
    else:
        gate_statistic = raw_gate_statistic
        gate_name = "client_raw_anchor_cosine_q25"
    gate_triggered = (selection == "rank_ensemble" or
                      gate_statistic < gate_threshold)
    if not gate_triggered:
        return mixture, {
            "uses_hidden_mixture_labels": False,
            "keep_fraction": 1.0,
            "kept": len(mixture),
            "gate_quantile": gate_quantile,
            "gate_statistic": gate_statistic,
            "gate_threshold": gate_threshold,
            "gate_triggered": False,
            "diagnostics_path": str(diagnostics_path),
        }
    if selection in {"adaptive_balanced_rank",
                     "adaptive_balanced_relative_rank"}:
        per_label = max(1, round(keep / 4))
        accepted = set()
        for label in range(4):
            candidates = [index for index, record in enumerate(ordered)
                          if record["claimed_label"] == label]
            candidates.sort(key=lambda index: (scores[index],
                                               mixture_ids[index]))
            accepted.update(candidates[:per_label])
    else:
        ranked = sorted(range(len(mixture)),
                        key=lambda index: (scores[index], mixture_ids[index]))
        accepted = set(ranked[:keep])
    selected = [item for index, item in enumerate(mixture)
                if index in accepted]
    return selected, {
        "uses_hidden_mixture_labels": False,
        "keep_fraction": keep_fraction,
        "kept": len(selected),
        "gate_quantile": gate_quantile,
        "gate_statistic": gate_statistic,
        "raw_gate_statistic": raw_gate_statistic,
        "gate_name": gate_name,
        "gate_threshold": gate_threshold,
        "gate_triggered": True,
        "class_balanced": selection in {"adaptive_balanced_rank",
                                         "adaptive_balanced_relative_rank"},
        "diagnostics_path": str(diagnostics_path),
        "diagnostics_record_order_sha256": diagnostics[
            "record_order_sha256"],
        "score": "mean within-client percentile rank of negative anchor cosine, loss, and semantic inconsistency",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path("kafa_script/benchmark_manifest_v1.json"))
    parser.add_argument("--agnews-data", type=Path,
                        default=Path(".cache/datasets/agnews/train.jsonl"))
    parser.add_argument("--imdb-data", type=Path,
                        default=Path(".cache/datasets/imdb/train.jsonl"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--conflict-ratio", type=float, default=0.0)
    parser.add_argument("--off-task-ratio", type=float, default=0.0)
    parser.add_argument("--train-per-class", type=int, default=64)
    parser.add_argument("--eval-per-class", type=int, default=200)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--selection",
                        choices=["none", "oracle_clean", "rank_ensemble",
                                 "adaptive_balanced_rank",
                                 "adaptive_balanced_relative_rank"],
                        default="none")
    parser.add_argument("--keep-fraction", type=float, default=0.5)
    parser.add_argument("--gate-quantile", type=float, default=0.25)
    parser.add_argument("--gate-threshold", type=float, default=0.7)
    parser.add_argument("--diagnostics", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.conflict_ratio, args.off_task_ratio) < 0:
        parser.error("mixture ratios must be non-negative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    manifest = json.loads(args.manifest.read_text())
    agnews = load_jsonl(args.agnews_data)
    imdb = load_jsonl(args.imdb_data)
    base_rows = select_client_train(agnews, manifest, args.seed,
                                    args.train_per_class)
    mixture = build_mixture(base_rows, imdb, args.seed, args.conflict_ratio,
                            args.off_task_ratio)
    train_mixture, selection_metadata = select_for_training(
        mixture, args.selection, args.keep_fraction, args.diagnostics,
        args.gate_quantile, args.gate_threshold)
    if not 0 < args.keep_fraction <= 1:
        parser.error("--keep-fraction must be in (0, 1]")
    eval_rows = select_manifest_rows(agnews, manifest, "development",
                                     args.eval_per_class)

    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [encode_mixture_example(tokenizer, item, args.max_length)
               for item in train_mixture]
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(encoded, batch_size=args.batch_size, shuffle=True,
                        generator=generator,
                        collate_fn=lambda batch: collate(tokenizer, batch))
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate)
    torch.cuda.reset_peak_memory_stats()
    initial = evaluate(model, tokenizer, eval_rows, device, args.max_length,
                       args.eval_batch_size)
    started = time.perf_counter()
    iterator = iter(loader)
    model.train()
    last_loss = math.nan
    for _ in range(args.steps):
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
        last_loss = float(loss.detach())
    train_seconds = time.perf_counter() - started
    final = evaluate(model, tokenizer, eval_rows, device, args.max_length,
                     args.eval_batch_size)
    buffer = io.BytesIO()
    torch.save(get_peft_model_state_dict(model), buffer)
    counts = Counter(item["kind"] for item in mixture)
    result = {
        "evidence_label": "benchmark_dev",
        "paper_result": False,
        "study": "within_client_real_mixture_mechanism",
        "method": ("naive_local_lora" if args.selection == "none" else
                   f"local_lora_{args.selection}"),
        "seed": args.seed,
        "model": str(args.model),
        "model_weight_sha256": sha256_file(args.model / "model.safetensors"),
        "manifest_sha256": sha256_file(args.manifest),
        "requested_conflict_ratio": args.conflict_ratio,
        "requested_off_task_ratio": args.off_task_ratio,
        "realized_counts": dict(sorted(counts.items())),
        "train_examples": len(mixture),
        "selected_train_examples": len(train_mixture),
        "selected_realized_counts": dict(sorted(Counter(
            item["kind"] for item in train_mixture).items())),
        "selection_metadata": selection_metadata,
        "eval_examples": len(eval_rows),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "last_batch_train_loss": last_loss,
        "initial": initial,
        "final": final,
        "train_wall_seconds": train_seconds,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "serialized_adapter_bytes": buffer.tell(),
        "base_agnews_ids_sha256": hashlib.sha256("\n".join(sorted(
            row["_id"] for row in base_rows)).encode()).hexdigest(),
        "mixture_assignment_sha256": hashlib.sha256("\n".join(
            f"{item['kind']}:{item['source_id']}:{item['answer']}"
            for item in mixture).encode()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
