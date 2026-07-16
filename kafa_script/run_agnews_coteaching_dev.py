#!/usr/bin/env python3
"""Co-Teaching baseline for controlled AG News label/task mixtures."""

from __future__ import annotations

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
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.run_agnews_local_dev import (  # noqa: E402
    candidate_example, evaluate, select_client_train, select_manifest_rows)
from kafa_script.run_agnews_mixture_dev import (  # noqa: E402
    build_mixture, encode_mixture_example)
from kafa_script.run_real_data_plumbing_smoke import (  # noqa: E402
    LABELS, collate, load_jsonl, sha256_file)


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


def metrics_from_probabilities(probabilities, rows):
    predictions = probabilities.argmax(1)
    targets = torch.tensor([int(row["label"]) for row in rows])
    accuracy = float((predictions == targets).float().mean())
    macro_f1 = 0.0
    recalls = []
    for label in range(len(LABELS)):
        label_mask = targets == label
        recalls.append(float((predictions[label_mask] == label).float().mean()))
        tp = ((predictions == label) & (targets == label)).sum().float()
        fp = ((predictions == label) & (targets != label)).sum().float()
        fn = ((predictions != label) & (targets == label)).sum().float()
        precision = tp / (tp + fp).clamp_min(1)
        recall = tp / (tp + fn).clamp_min(1)
        macro_f1 += float(2 * precision * recall /
                          (precision + recall).clamp_min(1e-12)) / len(LABELS)
    confidence = probabilities.max(1).values
    correct = (predictions == targets).float()
    ece = 0.0
    for lower in torch.linspace(0, .9, 10):
        upper = lower + .1
        in_bin = (confidence >= lower) & (confidence < upper)
        if in_bin.any():
            ece += float(in_bin.float().mean() *
                         (correct[in_bin].mean() - confidence[in_bin].mean()).abs())
    nll = float(-probabilities[torch.arange(len(rows)), targets].clamp_min(1e-12).log().mean())
    return {"accuracy": accuracy, "macro_f1": macro_f1,
            "label_nll": nll, "ece": ece,
            "per_class_recall": recalls}


@torch.no_grad()
def ensemble_evaluate(models, tokenizer, rows, device, max_length, batch_size):
    candidates = [candidate_example(tokenizer, row, label, max_length)
                  for row in rows for label in range(len(LABELS))]
    loader = DataLoader(candidates, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda batch: collate(tokenizer, batch))
    model_probabilities = []
    for model in models:
        model.eval()
        losses = []
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            losses.extend(per_example_loss(model, batch).float().cpu().tolist())
        matrix = torch.tensor(losses).reshape(len(rows), len(LABELS))
        model_probabilities.append(torch.softmax(-matrix, dim=1))
    probabilities = torch.stack(model_probabilities).mean(0)
    return metrics_from_probabilities(probabilities, rows)


def adapter_bytes(model) -> int:
    buffer = io.BytesIO()
    torch.save(get_peft_model_state_dict(model), buffer)
    return buffer.tell()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("kafa_script/benchmark_manifest_v1.json"))
    parser.add_argument("--agnews-data", type=Path, default=Path(".cache/datasets/agnews/train.jsonl"))
    parser.add_argument("--imdb-data", type=Path, default=Path(".cache/datasets/imdb/train.jsonl"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--conflict-ratio", type=float, default=.4)
    parser.add_argument("--off-task-ratio", type=float, default=0.0)
    parser.add_argument("--forget-rate", type=float, required=True)
    parser.add_argument("--train-per-class", type=int, default=64)
    parser.add_argument("--eval-per-class", type=int, default=200)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.forget_rate < 1:
        parser.error("forget-rate must be in [0,1)")
    if args.conflict_ratio + args.off_task_ratio >= 1:
        parser.error("mixture ratios must sum below one")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    manifest = json.loads(args.manifest.read_text())
    agnews = load_jsonl(args.agnews_data)
    imdb = load_jsonl(args.imdb_data)
    base = select_client_train(agnews, manifest, args.seed, args.train_per_class)
    mixture = build_mixture(base, imdb, args.seed, args.conflict_ratio,
                            args.off_task_ratio)
    eval_rows = select_manifest_rows(agnews, manifest, "development",
                                     args.eval_per_class)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = []
    kind_names = sorted({item["kind"] for item in mixture})
    kind_to_index = {name: index for index, name in enumerate(kind_names)}
    for item in mixture:
        row = encode_mixture_example(tokenizer, item, args.max_length)
        row["kind_index"] = kind_to_index[item["kind"]]
        encoded.append(row)

    def batch_collate(batch):
        tensors = collate(tokenizer, batch)
        tensors["kind_index"] = torch.tensor([item["kind_index"] for item in batch])
        return tensors

    loader = DataLoader(encoded, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed),
                        collate_fn=batch_collate)
    models = [make_model(args.model, device, args.seed * 100 + peer)
              for peer in (1, 2)]
    optimizers = [torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate) for model in models]
    initial = [evaluate(model, tokenizer, eval_rows, device, args.max_length,
                        args.eval_batch_size) for model in models]
    iterator = iter(loader)
    peer_loss_history = [[], []]
    selected_kinds = [Counter(), Counter()]
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
        remember = max(1, round(len(kinds) * (1 - args.forget_rate)))
        own_small = [torch.argsort(loss.detach())[:remember] for loss in losses]
        exchanged = [own_small[1], own_small[0]]
        objectives = [losses[peer][exchanged[peer]].mean() for peer in (0, 1)]
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        (objectives[0] + objectives[1]).backward()
        for optimizer in optimizers:
            optimizer.step()
        for peer in (0, 1):
            peer_loss_history[peer].append(float(objectives[peer].detach()))
            for index in exchanged[peer].cpu().tolist():
                selected_kinds[peer][kind_names[int(kinds[index])]] += 1
    train_wall = time.perf_counter() - started
    final = [evaluate(model, tokenizer, eval_rows, device, args.max_length,
                      args.eval_batch_size) for model in models]
    ensemble = ensemble_evaluate(models, tokenizer, eval_rows, device,
                                 args.max_length, args.eval_batch_size)
    result = {
        "evidence_label": "benchmark_dev", "paper_result": False,
        "method": "co_teaching_lora", "seed": args.seed,
        "model": str(args.model),
        "model_weight_sha256": sha256_file(args.model / "model.safetensors"),
        "manifest_sha256": sha256_file(args.manifest),
        "conflict_ratio": args.conflict_ratio,
        "off_task_ratio": args.off_task_ratio,
        "forget_rate": args.forget_rate,
        "forget_rate_uses_known_contamination": True,
        "mixture_counts": dict(sorted(Counter(item["kind"] for item in mixture).items())),
        "train_examples": len(mixture), "eval_examples": len(eval_rows),
        "steps": args.steps, "batch_size": args.batch_size,
        "peer_initial": initial, "peer_final": final,
        "ensemble_final": ensemble,
        "mean_peer_objective": [sum(values) / len(values) for values in peer_loss_history],
        "exchanged_selected_kind_counts": [dict(sorted(counts.items())) for counts in selected_kinds],
        "train_wall_seconds": train_wall,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "serialized_two_adapter_bytes": sum(adapter_bytes(model) for model in models),
        "mixture_assignment_sha256": hashlib.sha256("\n".join(
            f"{item['kind']}:{item['source_id']}:{item['answer']}" for item in mixture).encode()).hexdigest(),
        "limitations": [
            "Co-Teaching uses two full client backbones and an oracle forget rate equal to known contamination.",
            "The joint condition includes off-task examples although the original method targets finite-label noise."
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
