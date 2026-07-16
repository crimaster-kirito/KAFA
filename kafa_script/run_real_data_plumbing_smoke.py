"""Real-data plumbing smoke using a tiny random Qwen2 architecture.

This verifies tokenization, causal-label masking, LoRA optimization, constrained
classification scoring, and resource accounting. The randomly initialized model
is intentionally incapable of supporting a paper performance claim.
"""

import argparse
import hashlib
import io
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoTokenizer, Qwen2Config,
                          Qwen2ForCausalLM)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


LABELS = ("World", "Sports", "Business", "Science")
PROMPT = (
    "Classify the news article into exactly one category: "
    "World, Sports, Business, or Science.\nArticle: {text}\nCategory:"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        rows = []
        for line in handle:
            row = json.loads(line)
            payload = json.dumps(row, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":"))
            row["_id"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            rows.append(row)
        return rows


def balanced_subset(rows, per_class):
    selected = []
    counts = {label: 0 for label in range(len(LABELS))}
    for row in rows:
        label = int(row["label"])
        if counts[label] < per_class:
            selected.append(row)
            counts[label] += 1
        if all(value == per_class for value in counts.values()):
            break
    if not all(value == per_class for value in counts.values()):
        raise ValueError(f"insufficient balanced data: {counts}")
    return selected


def manifest_subset(rows, manifest, split):
    requested = [record["id"] for record in manifest["splits"][split]["records"]]
    by_id = {row["_id"]: row for row in rows}
    missing = [record_id for record_id in requested if record_id not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} manifest IDs missing from {split}")
    return [by_id[record_id] for record_id in requested]


def encode_training_example(tokenizer, row, max_length):
    prompt_ids = tokenizer(PROMPT.format(text=row["text"]),
                           add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(" " + LABELS[int(row["label"])],
                           add_special_tokens=False)["input_ids"]
    # Preserve the answer and the end of long articles.
    prompt_ids = prompt_ids[-(max_length - len(answer_ids)):]
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return {"input_ids": input_ids, "labels": labels}


def collate(tokenizer, batch):
    width = max(len(item["input_ids"]) for item in batch)
    input_ids, labels, attention_mask = [], [], []
    for item in batch:
        padding = width - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * padding)
        labels.append(item["labels"] + [-100] * padding)
        attention_mask.append([1] * len(item["input_ids"]) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }


@torch.no_grad()
def mean_loss(model, loader, device):
    model.eval()
    losses = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        losses.append(float(model(**batch).loss))
    return sum(losses) / len(losses)


@torch.no_grad()
def label_nll(model, tokenizer, row, label, device, max_length):
    prompt_ids = tokenizer(PROMPT.format(text=row["text"]),
                           add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(" " + LABELS[label],
                           add_special_tokens=False)["input_ids"]
    prompt_ids = prompt_ids[-(max_length - len(answer_ids)):]
    ids = torch.tensor([prompt_ids + answer_ids], device=device)
    logits = model(input_ids=ids).logits[:, :-1]
    targets = ids[:, 1:]
    token_loss = F.cross_entropy(logits.transpose(1, 2), targets,
                                 reduction="none")
    answer_start = max(len(prompt_ids) - 1, 0)
    return float(token_loss[0, answer_start:].mean())


@torch.no_grad()
def classification_accuracy(model, tokenizer, rows, device, max_length):
    model.eval()
    predictions = []
    for row in rows:
        losses = [
            label_nll(model, tokenizer, row, label, device, max_length)
            for label in range(len(LABELS))
        ]
        predictions.append(min(range(len(losses)), key=losses.__getitem__))
    correct = sum(pred == int(row["label"])
                  for pred, row in zip(predictions, rows))
    return correct / len(rows), predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path,
                        default=Path(".cache/models/Qwen2.5-0.5B"))
    parser.add_argument("--pretrained-model", type=Path, default=None,
                        help="Local pretrained checkpoint; omit for random plumbing model.")
    parser.add_argument("--train-data", type=Path,
                        default=Path(".cache/datasets/agnews/train.jsonl"))
    parser.add_argument("--test-data", type=Path,
                        default=Path(".cache/datasets/agnews/test.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("kafa_script/analysis_outputs/real_plumbing_smoke.json"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("kafa_script/benchmark_manifest_v1.json"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"],
                        default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,
                                              local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_rows = manifest_subset(load_jsonl(args.train_data), manifest,
                                 "smoke_train")
    test_rows = manifest_subset(load_jsonl(args.test_data), manifest,
                                "smoke_test")
    encoded = [
        encode_training_example(tokenizer, row, args.max_length)
        for row in train_rows
    ]
    loader = DataLoader(encoded,
                        batch_size=4,
                        shuffle=False,
                        collate_fn=lambda batch: collate(tokenizer, batch))

    if args.pretrained_model is None:
        config = Qwen2Config(
            vocab_size=len(tokenizer),
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=args.max_length,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = Qwen2ForCausalLM(config)
        model_description = "randomly initialized tiny Qwen2 (2 layers, hidden 64)"
        evidence_label = "plumbing_only_random_model"
    else:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.pretrained_model,
            local_files_only=True,
            dtype=dtype,
        )
        model_description = str(args.pretrained_model)
        evidence_label = "smoke_real"
    model = get_peft_model(
        model,
        LoraConfig(r=4,
                   lora_alpha=8,
                   lora_dropout=0.0,
                   target_modules=["q_proj", "v_proj"],
                   task_type="CAUSAL_LM"),
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-3)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    initial_loss = mean_loss(model, loader, device)
    initial_accuracy, _ = classification_accuracy(model, tokenizer, test_rows,
                                                   device, args.max_length)
    started = time.perf_counter()
    iterator = iter(loader)
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
    elapsed = time.perf_counter() - started
    final_loss = mean_loss(model, loader, device)
    final_accuracy, predictions = classification_accuracy(
        model, tokenizer, test_rows, device, args.max_length)

    buffer = io.BytesIO()
    torch.save(get_peft_model_state_dict(model), buffer)
    peak_memory = (torch.cuda.max_memory_allocated()
                   if device.type == "cuda" else 0)
    result = {
        "evidence_label": evidence_label,
        "paper_result": False,
        "seed": args.seed,
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "device": str(device),
        "real_dataset": "AG News",
        "model": model_description,
        "tokenizer": str(args.tokenizer),
        "model_weight_sha256": (
            sha256_file(args.pretrained_model / "model.safetensors")
            if args.pretrained_model is not None else None),
        "train_examples": len(train_rows),
        "test_examples": len(test_rows),
        "steps": args.steps,
        "initial_train_loss": initial_loss,
        "final_train_loss": final_loss,
        "initial_test_accuracy": initial_accuracy,
        "final_test_accuracy": final_accuracy,
        "test_predictions": predictions,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "serialized_adapter_bytes": buffer.tell(),
        "train_wall_seconds": elapsed,
        "peak_allocated_gpu_bytes": peak_memory,
        "success_criterion": "finite loss decreases after LoRA optimization",
        "success": bool(torch.isfinite(torch.tensor(final_loss)) and
                        final_loss < initial_loss),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
