#!/usr/bin/env python3
"""Matched numeric-random and client-small-loss GSM8K selection baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.run_gsm8k_anchor_teacher_router_dev import loss_for, make_lora, train
from kafa_script.run_gsm8k_local_dev import evaluate, load_jsonl, normalize_numeric
from kafa_script.run_gsm8k_mixture_dev import build_mixture
from kafa_script.run_real_data_plumbing_smoke import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-model", type=Path, required=True)
    parser.add_argument("--selector", choices=["numeric_random", "client_small_loss"], required=True)
    parser.add_argument("--manifest", type=Path, default=Path("kafa_script/gsm8k_benchmark_manifest_v1.json"))
    parser.add_argument("--train-data", type=Path, default=Path(".cache/datasets/gsm8k/processed/train.jsonl"))
    parser.add_argument("--squad-data", type=Path, default=Path(".cache/datasets/squad/test.jsonl"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--client-index", type=int, default=0)
    parser.add_argument("--conflict-ratio", type=float, default=.4)
    parser.add_argument("--off-task-ratio", type=float, default=.2)
    parser.add_argument("--client-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--keep-count", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    manifest = json.loads(args.manifest.read_text())
    rows = load_jsonl(args.train_data)
    by_index = {row["source_index"]: row for row in rows}
    base = [by_index[i] for i in manifest["partitions"]["client_pools"][args.client_index]]
    dev = [by_index[i] for i in manifest["partitions"]["development"][:args.eval_size]]
    squad = [json.loads(line) for line in args.squad_data.read_text().split("\n") if line]
    mixture = build_mixture(base, squad, args.seed, args.client_index,
                            args.conflict_ratio, args.off_task_ratio)
    eligible = [i for i, item in enumerate(mixture)
                if normalize_numeric(item["answer"]) is not None]
    if len(eligible) < args.keep_count:
        raise ValueError("not enough numeric-interface examples")

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = make_lora(args.client_model, device)
    initial = evaluate(model, tokenizer, dev, device, args.max_length,
                       args.eval_batch_size, args.max_new_tokens)
    selection_started = time.perf_counter()
    if args.selector == "numeric_random":
        order = eligible.copy()
        random.Random(args.seed + 1_000_003).shuffle(order)
        chosen = order[:args.keep_count]
        score_range = None
    else:
        scored = [(loss_for(model, tokenizer, mixture[index], device,
                            args.max_length), index) for index in eligible]
        selected_scores = sorted(scored, key=lambda pair: (pair[0], mixture[pair[1]]["source_id"]))[:args.keep_count]
        chosen = [index for _, index in selected_scores]
        score_range = [min(score for score, _ in selected_scores),
                       max(score for score, _ in selected_scores)]
    selection_wall = time.perf_counter() - selection_started
    selected = [mixture[index] for index in chosen]
    train_started = time.perf_counter()
    mean_loss = train(model, tokenizer, selected, device, args.client_steps,
                      args.batch_size, args.max_length, args.learning_rate,
                      args.seed)
    train_wall = time.perf_counter() - train_started
    final = evaluate(model, tokenizer, dev, device, args.max_length,
                     args.eval_batch_size, args.max_new_tokens)
    result = {
        "evidence_label": "benchmark_dev", "paper_result": False,
        "method": f"gsm8k_{args.selector}_numeric_rank", "seed": args.seed,
        "client_model": str(args.client_model),
        "client_weight_sha256": sha256_file(args.client_model / "model.safetensors"),
        "manifest_sha256": sha256_file(args.manifest),
        "squad_sha256": sha256_file(args.squad_data),
        "client_index": args.client_index,
        "mixture_counts": dict(sorted(Counter(item["kind"] for item in mixture).items())),
        "selection": {
            "selector": args.selector, "keep_count": args.keep_count,
            "numeric_eligible": len(eligible),
            "selected_kind_counts": dict(sorted(Counter(item["kind"] for item in selected).items())),
            "selected_source_ids_sha256": hashlib.sha256("\n".join(item["source_id"] for item in selected).encode()).hexdigest(),
            "selected_loss_range": score_range,
            "wall_seconds": selection_wall,
        },
        "development_examples": len(dev), "client_steps": args.client_steps,
        "initial": initial, "final": final, "client_mean_train_loss": mean_loss,
        "client_train_wall_seconds": train_wall,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "limitations": ["Development-only; official GSM8K test is not loaded.",
                        "Contamination kind is recorded only for post-selection audit."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
