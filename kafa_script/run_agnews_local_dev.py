"""Frozen real-data local-only development baseline for KAFA."""

import argparse
import hashlib
import io
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.run_real_data_plumbing_smoke import (  # noqa: E402
    LABELS, PROMPT, collate, encode_training_example, load_jsonl, sha256_file)


def select_client_train(rows, manifest, seed, per_class):
    excluded = {
        record["id"]
        for split in ("anchor", "development")
        for record in manifest["splits"][split]["records"]
    }
    grouped = defaultdict(list)
    for row in rows:
        if row["_id"] in excluded:
            continue
        rank = hashlib.sha256(
            f"local-dev-v1|{seed}|{row['_id']}".encode()).hexdigest()
        grouped[int(row["label"])].append((rank, row))
    selected = []
    for label in range(len(LABELS)):
        selected.extend(row for _, row in sorted(grouped[label])[:per_class])
    return selected


def select_manifest_rows(rows, manifest, split, per_class=None):
    requested = manifest["splits"][split]["records"]
    by_id = {row["_id"]: row for row in rows}
    selected, counts = [], defaultdict(int)
    for record in requested:
        label = int(record["label"])
        if per_class is None or counts[label] < per_class:
            selected.append(by_id[record["id"]])
            counts[label] += 1
    return selected


def candidate_example(tokenizer, row, label, max_length):
    prompt = tokenizer(PROMPT.format(text=row["text"]),
                       add_special_tokens=True)["input_ids"]
    answer = tokenizer(" " + LABELS[label],
                       add_special_tokens=False)["input_ids"]
    prompt = prompt[-(max_length - len(answer)):]
    return {
        "input_ids": prompt + answer,
        "labels": [-100] * len(prompt) + answer,
    }


@torch.no_grad()
def evaluate(model, tokenizer, rows, device, max_length, batch_size):
    model.eval()
    candidates = [
        candidate_example(tokenizer, row, label, max_length)
        for row in rows for label in range(len(LABELS))
    ]
    loader = DataLoader(candidates,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=lambda batch: collate(tokenizer, batch))
    losses = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"]).logits[:, :-1]
        targets = batch["input_ids"][:, 1:]
        mask = batch["labels"][:, 1:] != -100
        token_losses = F.cross_entropy(logits.transpose(1, 2), targets,
                                       reduction="none")
        sequence_loss = ((token_losses * mask).sum(1) /
                         mask.sum(1).clamp_min(1))
        losses.extend(sequence_loss.float().cpu().tolist())
    matrix = torch.tensor(losses).reshape(len(rows), len(LABELS))
    probabilities = torch.softmax(-matrix, dim=1)
    predictions = probabilities.argmax(1)
    targets = torch.tensor([int(row["label"]) for row in rows])
    accuracy = float((predictions == targets).float().mean())
    recalls = []
    for label in range(len(LABELS)):
        label_mask = targets == label
        recalls.append(float((predictions[label_mask] == label).float().mean()))
    macro_f1 = 0.0
    for label in range(len(LABELS)):
        tp = ((predictions == label) & (targets == label)).sum().float()
        fp = ((predictions == label) & (targets != label)).sum().float()
        fn = ((predictions != label) & (targets == label)).sum().float()
        precision = tp / (tp + fp).clamp_min(1)
        recall = tp / (tp + fn).clamp_min(1)
        macro_f1 += float(2 * precision * recall /
                          (precision + recall).clamp_min(1e-12)) / len(LABELS)
    true_nll = float(matrix[torch.arange(len(rows)), targets].mean())
    confidence = probabilities.max(1).values
    correct = (predictions == targets).float()
    ece = 0.0
    for lower in torch.linspace(0, 0.9, 10):
        upper = lower + 0.1
        bin_mask = (confidence >= lower) & (confidence < upper)
        if bin_mask.any():
            ece += float(bin_mask.float().mean() *
                         (correct[bin_mask].mean() -
                          confidence[bin_mask].mean()).abs())
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "label_nll": true_nll,
        "ece": ece,
        "per_class_recall": recalls,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path("kafa_script/benchmark_manifest_v1.json"))
    parser.add_argument("--train-data", type=Path,
                        default=Path(".cache/datasets/agnews/train.jsonl"))
    parser.add_argument("--test-data", type=Path,
                        default=Path(".cache/datasets/agnews/test.jsonl"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-per-class", type=int, default=64)
    parser.add_argument("--eval-per-class", type=int, default=200)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    manifest = json.loads(args.manifest.read_text())
    all_train = load_jsonl(args.train_data)
    train_rows = select_client_train(all_train, manifest, args.seed,
                                     args.train_per_class)
    eval_rows = select_manifest_rows(all_train, manifest, "development",
                                     args.eval_per_class)
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [encode_training_example(tokenizer, row, args.max_length)
               for row in train_rows]
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(encoded,
                        batch_size=args.batch_size,
                        shuffle=True,
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
    result = {
        "evidence_label": "benchmark_dev",
        "paper_result": False,
        "method": "local_only",
        "seed": args.seed,
        "model": str(args.model),
        "model_weight_sha256": sha256_file(args.model / "model.safetensors"),
        "manifest_sha256": sha256_file(args.manifest),
        "train_examples": len(train_rows),
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
        "train_record_ids_sha256": hashlib.sha256(
            "\n".join(sorted(row["_id"] for row in train_rows)).encode()
        ).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
