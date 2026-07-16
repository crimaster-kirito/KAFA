#!/usr/bin/env python3
"""Causal-LM naive and Co-Teaching baselines on COMPRISE natural label noise."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def prompt(text: str, labels: tuple[str, ...]) -> str:
    return ("Classify the news headline into exactly one category: "
            f"{', '.join(labels)}.\nHeadline: {text}\nCategory:")


def encode(tokenizer, text: str, answer: str, labels: tuple[str, ...],
           max_length: int) -> dict[str, list[int]]:
    prefix = tokenizer(prompt(text, labels), add_special_tokens=True)["input_ids"]
    suffix = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
    prefix = prefix[-max(1, max_length - len(suffix)):]
    return {"input_ids": prefix + suffix,
            "labels": [-100] * len(prefix) + suffix}


def collate(tokenizer, rows):
    width = max(len(row["input_ids"]) for row in rows)
    input_ids, attention_mask, labels = [], [], []
    for row in rows:
        padding = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [tokenizer.pad_token_id] * padding)
        attention_mask.append([1] * len(row["input_ids"]) + [0] * padding)
        labels.append(row["labels"] + [-100] * padding)
    return {"input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels)}


def make_model(path: Path, device, seed: int):
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype=torch.bfloat16)
    return get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM")).to(device)


def per_example_loss(model, batch):
    logits = model(input_ids=batch["input_ids"],
                   attention_mask=batch["attention_mask"]).logits[:, :-1]
    targets = batch["input_ids"][:, 1:]
    mask = batch["labels"][:, 1:] != -100
    token_loss = F.cross_entropy(logits.transpose(1, 2), targets,
                                 reduction="none")
    return (token_loss * mask).sum(1) / mask.sum(1).clamp_min(1)


def metrics(probabilities: torch.Tensor, targets: torch.Tensor,
            label_count: int) -> dict:
    predictions = probabilities.argmax(1)
    accuracy = float((predictions == targets).float().mean())
    f1s, recalls = [], []
    for label in range(label_count):
        positive = targets == label
        tp = ((predictions == label) & positive).sum().float()
        fp = ((predictions == label) & ~positive).sum().float()
        fn = ((predictions != label) & positive).sum().float()
        precision = tp / (tp + fp).clamp_min(1)
        recall = tp / (tp + fn).clamp_min(1)
        f1s.append(float(2 * precision * recall /
                         (precision + recall).clamp_min(1e-12)))
        recalls.append(float(recall))
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1s) / label_count,
        "label_nll": float(-probabilities[
            torch.arange(len(targets)), targets].clamp_min(1e-12).log().mean()),
        "per_class_recall": recalls,
    }


@torch.no_grad()
def evaluate(models, tokenizer, rows, labels, device, max_length, batch_size):
    candidates = [encode(tokenizer, row["news_title"], label, labels, max_length)
                  for row in rows for label in labels]
    loader = DataLoader(candidates, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda batch: collate(tokenizer, batch))
    all_probabilities = []
    for model in models:
        model.eval()
        losses = []
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            losses.extend(per_example_loss(model, batch).float().cpu().tolist())
        matrix = torch.tensor(losses).reshape(len(rows), len(labels))
        all_probabilities.append(torch.softmax(-matrix, dim=1))
    probabilities = torch.stack(all_probabilities).mean(0)
    label_to_index = {label: index for index, label in enumerate(labels)}
    targets = torch.tensor([label_to_index[row["label"].lower()] for row in rows])
    return metrics(probabilities, targets, len(labels))


def adapter_bytes(model) -> int:
    buffer = io.BytesIO()
    torch.save(get_peft_model_state_dict(model), buffer)
    return buffer.tell()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="COMPRISE_Data09_H-VOA-Topics_V1.0")
    parser.add_argument("--dataset-doi", default="10.5281/zenodo.5997635")
    parser.add_argument("--method", choices=("naive", "coteach"), required=True)
    parser.add_argument("--evaluation-split", choices=("dev", "test"), default="dev")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--forget-rate", type=float, default=0.5036674816625917)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.forget_rate < 1:
        parser.error("forget-rate must be in [0,1)")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    weak_path = args.data_root / "train_clean_noisy_labels.tsv"
    clean_path = args.data_root / "train_clean.tsv"
    eval_path = args.data_root / f"{args.evaluation_split}.tsv"
    weak_rows, clean_rows, eval_rows = map(read_tsv,
                                           (weak_path, clean_path, eval_path))
    clean_by_text = {row["news_title"]: row["label"].lower()
                     for row in clean_rows}
    if len(weak_rows) != len(clean_rows) or any(
            row["news_title"] not in clean_by_text for row in weak_rows):
        raise ValueError("clean/noisy training rows do not align")
    labels = tuple(sorted({row["label"].lower() for row in clean_rows}))
    noise_rate = sum(row["label"].lower() != clean_by_text[row["news_title"]]
                     for row in weak_rows) / len(weak_rows)
    if args.method == "coteach" and not math.isclose(
            args.forget_rate, noise_rate, abs_tol=1e-12):
        raise ValueError("Co-Teaching protocol requires the measured noise rate")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kind_names = ("clean_label_match", "noisy_label_conflict")
    encoded = []
    for row in weak_rows:
        item = encode(tokenizer, row["news_title"], row["label"].lower(),
                      labels, args.max_length)
        item["kind_index"] = int(
            row["label"].lower() != clean_by_text[row["news_title"]])
        encoded.append(item)

    def batch_collate(batch):
        result = collate(tokenizer, batch)
        result["kind_index"] = torch.tensor([row["kind_index"] for row in batch])
        return result

    loader = DataLoader(encoded, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed),
                        collate_fn=batch_collate)
    model_count = 1 if args.method == "naive" else 2
    models = [make_model(args.model, device,
                         args.seed if peer == 0 else args.seed + 100000)
              for peer in range(model_count)]
    optimizers = [torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate) for model in models]
    iterator = iter(loader)
    selected_kinds = [Counter() for _ in models]
    objective_history = [[] for _ in models]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        kinds = batch.pop("kind_index")
        batch = {key: value.to(device) for key, value in batch.items()}
        losses = [per_example_loss(model, batch) for model in models]
        if args.method == "naive":
            selections = [torch.arange(len(kinds), device=device)]
        else:
            remember = max(1, round(len(kinds) * (1 - args.forget_rate)))
            own_small = [torch.argsort(loss.detach())[:remember] for loss in losses]
            selections = [own_small[1], own_small[0]]
        objectives = [loss[selection].mean()
                      for loss, selection in zip(losses, selections)]
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        sum(objectives).backward()
        for optimizer in optimizers:
            optimizer.step()
        for peer, (objective, selection) in enumerate(zip(objectives, selections)):
            objective_history[peer].append(float(objective.detach()))
            for index in selection.cpu().tolist():
                selected_kinds[peer][kind_names[int(kinds[index])]] += 1
    train_wall = time.perf_counter() - started
    final = evaluate(models, tokenizer, eval_rows, labels, device,
                     args.max_length, args.eval_batch_size)
    result = {
        "evidence_label": ("benchmark_dev" if args.evaluation_split == "dev"
                           else "benchmark_holdout"),
        "paper_result": args.evaluation_split == "test",
        "dataset": args.dataset_name,
        "dataset_doi": args.dataset_doi,
        "method": args.method,
        "seed": args.seed,
        "model": str(args.model),
        "model_weight_sha256": sha256(args.model / "model.safetensors"),
        "training_file_sha256": sha256(weak_path),
        "clean_training_file_sha256": sha256(clean_path),
        "evaluation_file_sha256": sha256(eval_path),
        "evaluation_split": args.evaluation_split,
        "labels": labels,
        "train_examples": len(weak_rows),
        "eval_examples": len(eval_rows),
        "measured_training_noise_rate": noise_rate,
        "forget_rate": (args.forget_rate if args.method == "coteach" else None),
        "forget_rate_uses_clean_training_labels": args.method == "coteach",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "final": final,
        "selected_kind_counts": [dict(sorted(counts.items()))
                                 for counts in selected_kinds],
        "mean_objective": [sum(values) / len(values)
                           for values in objective_history],
        "train_wall_seconds": train_wall,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "serialized_adapter_bytes": sum(adapter_bytes(model) for model in models),
        "limitations": [
            "Co-Teaching receives the measured natural-noise fraction as oracle information.",
            "Clean training labels are used only to audit selections, never in optimization.",
            "The causal label interface is not a reproduction of Delora's GPT-4o-assisted pipeline."
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
