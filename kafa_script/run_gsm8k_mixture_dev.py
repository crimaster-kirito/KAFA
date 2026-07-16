#!/usr/bin/env python3
"""Development-only GSM8K clean/severe-mixture local baseline.

The mixture construction is visible and deterministic: target questions retain
their prompt, conflicts swap only final numeric answers within a client pool,
and off-task examples are frozen SQuAD extractive QA pairs.  Evaluation always
uses clean GSM8K development questions and generative numeric exact match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.run_gsm8k_local_dev import (  # noqa: E402
    PROMPT, evaluate, load_jsonl, normalize_numeric)
from kafa_script.run_real_data_plumbing_smoke import collate, sha256_file  # noqa: E402

OFFTASK_PROMPT = "Answer the question using the supplied context.\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"


def encode_supervised(tokenizer, prompt: str, answer: str, max_length: int) -> dict:
    prefix = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    suffix = tokenizer(" " + answer + tokenizer.eos_token,
                       add_special_tokens=False)["input_ids"]
    if len(suffix) >= max_length:
        raise ValueError("max_length too small for answer")
    prefix = prefix[-(max_length - len(suffix)):]
    return {"input_ids": prefix + suffix, "labels": [-100] * len(prefix) + suffix}


def select_off_task(rows: list[dict], count: int, seed: int, client_index: int) -> list[dict]:
    ranked = sorted(
        (hashlib.sha256(f"stage-b-off-task-v1|{seed}|{client_index}|{row['id']}".encode()).hexdigest(), row)
        for row in rows
    )
    chosen = []
    for _, row in ranked[:count]:
        answers = row.get("answers", {}).get("text", [])
        if not answers:
            raise ValueError(f"SQuAD row without answer: {row.get('id')}")
        chosen.append({
            "kind": "off_task", "source_id": str(row["id"]),
            "prompt": OFFTASK_PROMPT.format(context=row["context"], question=row["question"]),
            "answer": str(answers[0]),
        })
    return chosen


def build_mixture(base: list[dict], off_task_rows: list[dict], seed: int,
                  client_index: int, conflict_ratio: float,
                  off_task_ratio: float) -> list[dict]:
    if conflict_ratio < 0 or off_task_ratio < 0 or conflict_ratio + off_task_ratio >= 1:
        raise ValueError("ratios must be non-negative and sum to less than one")
    total = len(base)
    conflict_count = round(total * conflict_ratio)
    off_count = round(total * off_task_ratio)
    clean_count = total - conflict_count - off_count
    ranked = sorted(base, key=lambda row: hashlib.sha256(
        f"stage-b-mix-v1|{seed}|{client_index}|{row['source_index']}".encode()).hexdigest())
    clean_rows = ranked[:clean_count]
    conflict_rows = ranked[clean_count:clean_count + conflict_count]
    items = [
        {"kind": "clean", "source_id": str(row["source_index"]),
         "prompt": PROMPT.format(question=row["question"]), "answer": row["final_answer"]}
        for row in clean_rows
    ]
    if conflict_rows:
        answers = [row["final_answer"] for row in conflict_rows]
        rotated = answers[1:] + answers[:1]
        if len(conflict_rows) == 1 or any(a == b for a, b in zip(answers, rotated)):
            # A deterministic global shift of the whole base avoids accidental
            # identity even when numerical answers repeat among selected rows.
            all_answers = [row["final_answer"] for row in ranked]
            rotated = [next(answer for answer in all_answers if answer != row["final_answer"])
                       for row in conflict_rows]
        items.extend(
            {"kind": "conflict", "source_id": str(row["source_index"]),
             "prompt": PROMPT.format(question=row["question"]), "answer": answer}
            for row, answer in zip(conflict_rows, rotated)
        )
    items.extend(select_off_task(off_task_rows, off_count, seed, client_index))
    random.Random(seed + 1009 * client_index).shuffle(items)
    if len(items) != total or Counter(item["kind"] for item in items) != Counter(
            {"clean": clean_count, "conflict": conflict_count, "off_task": off_count}):
        raise AssertionError("mixture count invariant failed")
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path("kafa_script/gsm8k_benchmark_manifest_v1.json"))
    parser.add_argument("--train-data", type=Path,
                        default=Path(".cache/datasets/gsm8k/processed/train.jsonl"))
    parser.add_argument("--test-data", type=Path,
                        default=Path(".cache/datasets/gsm8k/processed/test.jsonl"))
    parser.add_argument("--secondary-manifest", type=Path,
                        default=Path("kafa_script/gsm8k_secondary_holdout_manifest_v1.json"))
    parser.add_argument("--squad-data", type=Path,
                        default=Path(".cache/datasets/squad/test.jsonl"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--client-index", type=int, default=0)
    parser.add_argument("--conflict-ratio", type=float, default=0.4)
    parser.add_argument("--off-task-ratio", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--evaluation-split", choices=["development", "secondary_holdout", "official_test"],
                        default="development")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    total_started = time.perf_counter()
    if args.steps <= 0:
        parser.error("steps must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    manifest = json.loads(args.manifest.read_text())
    pools = manifest["partitions"]["client_pools"]
    if not 0 <= args.client_index < len(pools):
        parser.error("client-index outside manifest client pools")
    train = load_jsonl(args.train_data)
    by_index = {row["source_index"]: row for row in train}
    base = [by_index[index] for index in pools[args.client_index]]
    if args.evaluation_split == "development":
        dev_indices = manifest["partitions"]["development"][:args.eval_size]
        dev = [by_index[index] for index in dev_indices]
    elif args.evaluation_split == "official_test":
        dev_indices = []
        dev = load_jsonl(args.test_data)
    else:
        secondary = json.loads(args.secondary_manifest.read_text())
        dev_indices = secondary["source_indices"]
        dev = [by_index[index] for index in dev_indices]
    squad = [json.loads(line) for line in args.squad_data.read_text().split("\n") if line]
    mixture = build_mixture(base, squad, args.seed, args.client_index,
                            args.conflict_ratio, args.off_task_ratio)
    if set(row["source_index"] for row in base) & set(dev_indices):
        raise AssertionError("client train and development rows overlap")

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Batch greedy generation for decoder-only models requires left padding;
    # our training collator pads explicitly and is unaffected by this setting.
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                                  dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                                              target_modules=["q_proj", "v_proj"],
                                              task_type="CAUSAL_LM")).to(device)
    encoded = [encode_supervised(tokenizer, item["prompt"], item["answer"], args.max_length)
               for item in mixture]
    loader = DataLoader(encoded, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed),
                        collate_fn=lambda batch: collate(tokenizer, batch))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.learning_rate)
    torch.cuda.reset_peak_memory_stats()
    initial = None if args.skip_initial_eval else evaluate(
        model, tokenizer, dev, device, args.max_length,
        args.eval_batch_size, args.max_new_tokens)
    iterator, losses = iter(loader), []
    started = time.perf_counter()
    model.train()
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
        losses.append(float(loss.detach()))
    wall = time.perf_counter() - started
    final = evaluate(model, tokenizer, dev, device, args.max_length,
                     args.eval_batch_size, args.max_new_tokens)
    total_wall = time.perf_counter() - total_started
    result = {
        "evidence_label": "benchmark_dev" if args.evaluation_split == "development" else "benchmark_holdout", "paper_result": False,
        "method": "gsm8k_local_naive_mixture", "seed": args.seed,
        "model": str(args.model), "model_weight_sha256": sha256_file(args.model / "model.safetensors"),
        "manifest_sha256": sha256_file(args.manifest), "squad_sha256": sha256_file(args.squad_data),
        "evaluation_manifest_sha256": sha256_file(args.secondary_manifest) if args.evaluation_split == "secondary_holdout" else None,
        "client_index": args.client_index, "mixture_counts": dict(sorted(Counter(item["kind"] for item in mixture).items())),
        "conflict_ratio": args.conflict_ratio, "off_task_ratio": args.off_task_ratio,
        "evaluation_split": args.evaluation_split, "evaluation_examples": len(dev), "steps": args.steps, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens, "mean_train_loss": sum(losses) / len(losses),
        "initial": initial, "final": final, "train_wall_seconds": wall,
        "total_wall_seconds": total_wall,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "mixture_source_ids_sha256": hashlib.sha256("\n".join(item["source_id"] for item in mixture).encode()).hexdigest(),
        "limitations": ["Official-test labels are loaded only when explicitly requested under a frozen final protocol.", "No routing is applied in this naive mixture baseline."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
