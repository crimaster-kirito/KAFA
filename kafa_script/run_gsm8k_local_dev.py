#!/usr/bin/env python3
"""Development-only local GSM8K adaptation and generative exact-match scoring.

This is intentionally separate from AG News candidate-label scoring.  A model
is trained to emit a final number and evaluated by greedy generation; it never
loads the GSM8K official test file.  It is a Stage-B plumbing/baseline runner,
not a federated result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.run_real_data_plumbing_smoke import collate, sha256_file  # noqa: E402


PROMPT = "Solve the following grade-school math problem. Reply with only the final numeric answer.\n\nQuestion: {question}\nAnswer:"
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")


def load_jsonl(path: Path) -> list[dict]:
    # GSM8K text can contain Unicode characters that Python regards as line
    # separators. JSONL records are delimited by literal LF bytes only.
    return [json.loads(line) for line in path.read_text().split("\n") if line]


def normalize_numeric(value: str) -> str | None:
    """Return a canonical finite decimal string, or None for non-numeric text."""
    value = value.strip().replace(",", "").replace("$", "")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        return None
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    number = number.normalize()
    rendered = format(number, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in {"", "-0"} else rendered


def parse_generated_answer(text: str) -> str | None:
    """Prefer a delimiter answer, otherwise use the final standalone number."""
    if "####" in text:
        candidate = normalize_numeric(text.rsplit("####", 1)[1].strip())
        if candidate is not None:
            return candidate
    matches = NUMBER.findall(text.replace(",", ""))
    return normalize_numeric(matches[-1]) if matches else None


def encode_example(tokenizer, row: dict, max_length: int) -> dict[str, list[int]]:
    prompt = tokenizer(PROMPT.format(question=row["question"]),
                       add_special_tokens=True)["input_ids"]
    answer = tokenizer(" " + row["final_answer"] + tokenizer.eos_token,
                       add_special_tokens=False)["input_ids"]
    if len(answer) >= max_length:
        raise ValueError("max_length too small for answer")
    prompt = prompt[-(max_length - len(answer)):]
    return {"input_ids": prompt + answer, "labels": [-100] * len(prompt) + answer}


@torch.no_grad()
def evaluate(model, tokenizer, rows: list[dict], device: torch.device,
             max_prompt_length: int, batch_size: int, max_new_tokens: int) -> dict:
    model.eval()
    predictions: list[str | None] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        encoded = tokenizer([PROMPT.format(question=row["question"]) for row in batch_rows],
                            add_special_tokens=True, truncation=True,
                            max_length=max_prompt_length, padding=True,
                            return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generated = model.generate(**encoded, do_sample=False,
                                   max_new_tokens=max_new_tokens,
                                   pad_token_id=tokenizer.pad_token_id,
                                   eos_token_id=tokenizer.eos_token_id)
        prompt_width = encoded["input_ids"].shape[1]
        continuations = tokenizer.batch_decode(generated[:, prompt_width:],
                                                skip_special_tokens=True)
        predictions.extend(parse_generated_answer(text) for text in continuations)
    targets = [normalize_numeric(row["final_answer"]) for row in rows]
    correct = sum(prediction == target for prediction, target in zip(predictions, targets))
    invalid = sum(prediction is None for prediction in predictions)
    return {
        "exact_match": correct / len(rows),
        "invalid_output_rate": invalid / len(rows),
        "examples": len(rows),
        "prediction_sha256": hashlib.sha256("\n".join(
            prediction if prediction is not None else "<invalid>"
            for prediction in predictions).encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path("kafa_script/gsm8k_benchmark_manifest_v1.json"))
    parser.add_argument("--train-data", type=Path,
                        default=Path(".cache/datasets/gsm8k/processed/train.jsonl"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--client-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0 or args.eval_size <= 0 or args.max_new_tokens <= 0:
        parser.error("steps, eval-size, and max-new-tokens must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    manifest = json.loads(args.manifest.read_text())
    pools = manifest["partitions"]["client_pools"]
    if not 0 <= args.client_index < len(pools):
        parser.error("client-index outside manifest client pools")
    rows = load_jsonl(args.train_data)
    by_index = {row["source_index"]: row for row in rows}
    train_rows = [by_index[index] for index in pools[args.client_index]]
    dev_indices = manifest["partitions"]["development"][:args.eval_size]
    eval_rows = [by_index[index] for index in dev_indices]
    if set(row["source_index"] for row in train_rows) & set(dev_indices):
        raise AssertionError("client train and development rows overlap")

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Generation of decoder-only models with a batch requires left padding.
    # Training collation is explicit and remains right padded with labels -100.
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")).to(device)
    encoded = [encode_example(tokenizer, row, args.max_length) for row in train_rows]
    loader = DataLoader(encoded, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed),
                        collate_fn=lambda batch: collate(tokenizer, batch))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate)
    torch.cuda.reset_peak_memory_stats()
    initial = evaluate(model, tokenizer, eval_rows, device, args.max_length,
                       args.eval_batch_size, args.max_new_tokens)
    started = time.perf_counter()
    iterator = iter(loader)
    losses = []
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
    wall_seconds = time.perf_counter() - started
    final = evaluate(model, tokenizer, eval_rows, device, args.max_length,
                     args.eval_batch_size, args.max_new_tokens)
    result = {
        "evidence_label": "benchmark_dev",
        "paper_result": False,
        "method": "gsm8k_local_final_answer_adaptation",
        "seed": args.seed,
        "model": str(args.model),
        "model_weight_sha256": sha256_file(args.model / "model.safetensors"),
        "manifest_sha256": sha256_file(args.manifest),
        "client_index": args.client_index,
        "train_examples": len(train_rows),
        "development_examples": len(eval_rows),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "mean_train_loss": sum(losses) / len(losses),
        "initial": initial,
        "final": final,
        "train_wall_seconds": wall_seconds,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "train_source_indices_sha256": hashlib.sha256("\n".join(
            str(row["source_index"]) for row in train_rows).encode()).hexdigest(),
        "limitations": [
            "Development-only runner; it cannot read or score the official GSM8K test split.",
            "It supervises final answers rather than rationales and does not establish reasoning faithfulness.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
