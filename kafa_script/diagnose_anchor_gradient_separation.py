"""Measure whether real mixed samples are separable by anchor gradients.

This is a diagnostic, not a training method. Hidden mixture labels are used
only after scores have been computed to measure separability.
"""

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.run_agnews_local_dev import (  # noqa: E402
    candidate_example, select_client_train, select_manifest_rows)
from kafa_script.run_agnews_mixture_dev import (  # noqa: E402
    build_mixture, encode_mixture_example)
from kafa_script.run_real_data_plumbing_smoke import (  # noqa: E402
    collate, encode_training_example, load_jsonl, sha256_file)


def parameter_gradients(parameters):
    return [torch.zeros_like(parameter, memory_format=torch.preserve_format)
            if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in parameters]


def gradient_cosine(left, right, eps=1e-12):
    dot = sum((a.float() * b.float()).sum() for a, b in zip(left, right))
    left_sq = sum(a.float().square().sum() for a in left)
    right_sq = sum(b.float().square().sum() for b in right)
    return float(dot / (left_sq.sqrt() * right_sq.sqrt()).clamp_min(eps))


def auroc(scores, positives):
    positive_scores = [score for score, positive in zip(scores, positives)
                       if positive]
    negative_scores = [score for score, positive in zip(scores, positives)
                       if not positive]
    if not positive_scores or not negative_scores:
        return None
    wins = 0.0
    for positive in positive_scores:
        for negative in negative_scores:
            wins += positive > negative
            wins += 0.5 * (positive == negative)
    return wins / (len(positive_scores) * len(negative_scores))


def average_precision(scores, positives):
    if not any(positives):
        return None
    ordered = sorted(zip(scores, positives), key=lambda pair: pair[0],
                     reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, positive) in enumerate(ordered, 1):
        if positive:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / sum(positives)


def rank_percentiles(values):
    """Return average-tie ranks scaled to [0, 1], lower value first."""
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


def summary_quantiles(values):
    ordered = sorted(values)
    return {
        "q10": ordered[round(0.10 * (len(ordered) - 1))],
        "q25": ordered[round(0.25 * (len(ordered) - 1))],
        "median": ordered[round(0.50 * (len(ordered) - 1))],
        "mean": sum(ordered) / len(ordered),
    }


def discrimination(records, positive_kinds):
    positives = [record["kind"] in positive_kinds for record in records]
    cosine_harm = [-record["anchor_cosine"] for record in records]
    loss_harm = [record["loss"] for record in records]
    return {
        "positive_kinds": sorted(positive_kinds),
        "positives": sum(positives),
        "negatives": len(positives) - sum(positives),
        "anchor_cosine_auroc": auroc(cosine_harm, positives),
        "anchor_cosine_auprc": average_precision(cosine_harm, positives),
        "loss_auroc": auroc(loss_harm, positives),
        "loss_auprc": average_precision(loss_harm, positives),
    }


def semantic_discrimination(records):
    subset = [record for record in records if record["kind"] != "off_task"]
    positives = [record["kind"] == "conflict" for record in subset]
    scores = [record["semantic_inconsistency_margin"] for record in subset]
    return {
        "positive_kinds": ["conflict"],
        "positives": sum(positives),
        "negatives": len(positives) - sum(positives),
        "semantic_margin_auroc": auroc(scores, positives),
        "semantic_margin_auprc": average_precision(scores, positives),
    }


def rank_ensemble_discrimination(records):
    subset = [record for record in records if record["kind"] != "off_task"]
    positives = [record["kind"] == "conflict" for record in subset]
    signals = [
        rank_percentiles([-record["anchor_cosine"] for record in subset]),
        rank_percentiles([record["loss"] for record in subset]),
        rank_percentiles([record["semantic_inconsistency_margin"]
                          for record in subset]),
    ]
    scores = [sum(parts) / len(parts) for parts in zip(*signals)]
    return {
        "positive_kinds": ["conflict"],
        "positives": sum(positives),
        "negatives": len(positives) - sum(positives),
        "rank_ensemble_auroc": auroc(scores, positives),
        "rank_ensemble_auprc": average_precision(scores, positives),
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
    parser.add_argument("--conflict-ratio", type=float, default=0.4)
    parser.add_argument("--off-task-ratio", type=float, default=0.2)
    parser.add_argument("--anchor-per-class", type=int, default=32)
    parser.add_argument("--train-per-class", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

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
    anchors = select_manifest_rows(agnews, manifest, "anchor",
                                   args.anchor_per_class)

    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")).to(device)
    model.eval()
    parameters = [parameter for parameter in model.parameters()
                  if parameter.requires_grad]

    grouped = defaultdict(list)
    for row in anchors:
        grouped[int(row["label"])].append(row)
    anchor_gradients = {}
    for label in range(4):
        encoded = [encode_training_example(tokenizer, row, args.max_length)
                   for row in grouped[label]]
        batch = collate(tokenizer, encoded)
        batch = {key: value.to(device) for key, value in batch.items()}
        model.zero_grad(set_to_none=True)
        model(**batch).loss.backward()
        anchor_gradients[label] = parameter_gradients(parameters)

    # Model-relative calibration: score known clean anchor examples against
    # their class anchor direction. This has no hidden mixture labels and
    # makes raw cosine scales comparable across model families.
    anchor_reference_scores = []
    for label in range(4):
        for row in grouped[label]:
            encoded = encode_training_example(tokenizer, row, args.max_length)
            batch = collate(tokenizer, [encoded])
            batch = {key: value.to(device) for key, value in batch.items()}
            model.zero_grad(set_to_none=True)
            model(**batch).loss.backward()
            anchor_reference_scores.append(gradient_cosine(
                parameter_gradients(parameters), anchor_gradients[label]))

    started = time.perf_counter()
    records = []
    for item in mixture:
        encoded = encode_mixture_example(tokenizer, item, args.max_length)
        batch = collate(tokenizer, [encoded])
        batch = {key: value.to(device) for key, value in batch.items()}
        model.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        sample_gradient = parameter_gradients(parameters)
        if item["kind"] == "off_task":
            similarities = [gradient_cosine(sample_gradient,
                                             anchor_gradients[label])
                            for label in range(4)]
            score = max(similarities)
            claimed_label = None
            candidate_losses = None
            semantic_margin = None
        else:
            claimed_label = int(item["training_label"])
            similarities = [gradient_cosine(
                sample_gradient, anchor_gradients[claimed_label])]
            score = similarities[0]
            candidates = [candidate_example(
                tokenizer, {"text": item["text"]}, label, args.max_length)
                          for label in range(4)]
            candidate_batch = collate(tokenizer, candidates)
            candidate_batch = {key: value.to(device)
                               for key, value in candidate_batch.items()}
            with torch.no_grad():
                logits = model(
                    input_ids=candidate_batch["input_ids"],
                    attention_mask=candidate_batch["attention_mask"]).logits[:, :-1]
                targets = candidate_batch["input_ids"][:, 1:]
                mask = candidate_batch["labels"][:, 1:] != -100
                token_losses = F.cross_entropy(
                    logits.transpose(1, 2), targets, reduction="none")
                losses = ((token_losses * mask).sum(1) /
                          mask.sum(1).clamp_min(1)).float().cpu().tolist()
            candidate_losses = losses
            semantic_margin = losses[claimed_label] - min(losses)
        records.append({
            "kind": item["kind"],
            "source_id": item["source_id"],
            "claimed_label": claimed_label,
            "loss": float(loss.detach()),
            "anchor_cosine": score,
            "all_applicable_cosines": similarities,
            "semantic_candidate_losses": candidate_losses,
            "semantic_inconsistency_margin": semantic_margin,
        })
    elapsed = time.perf_counter() - started

    by_kind = {}
    for kind in ("clean", "conflict", "off_task"):
        subset = [record for record in records if record["kind"] == kind]
        by_kind[kind] = {
            "count": len(subset),
            "mean_loss": (sum(record["loss"] for record in subset) /
                          len(subset) if subset else None),
            "mean_anchor_cosine": (sum(record["anchor_cosine"]
                                       for record in subset) / len(subset)
                                   if subset else None),
        }
    conflict_records = [record for record in records
                        if record["kind"] != "off_task"]
    result = {
        "evidence_label": "benchmark_dev",
        "paper_result": False,
        "study": "anchor_gradient_separation_diagnostic",
        "seed": args.seed,
        "model": str(args.model),
        "model_weight_sha256": sha256_file(args.model / "model.safetensors"),
        "manifest_sha256": sha256_file(args.manifest),
        "conflict_ratio": args.conflict_ratio,
        "off_task_ratio": args.off_task_ratio,
        "anchor_per_class": args.anchor_per_class,
        "anchor_reference_cosine": summary_quantiles(anchor_reference_scores),
        "sample_count": len(records),
        "scoring_seconds": elapsed,
        "by_kind": by_kind,
        "mixture_anchor_cosine": summary_quantiles(
            [record["anchor_cosine"] for record in records]),
        "harmful_vs_clean": discrimination(
            records, {"conflict", "off_task"}),
        "conflict_vs_clean_excluding_off_task": discrimination(
            conflict_records, {"conflict"}),
        "semantic_conflict_vs_clean": semantic_discrimination(records),
        "rank_ensemble_conflict_vs_clean": rank_ensemble_discrimination(records),
        "record_order_sha256": hashlib.sha256("\n".join(
            record["source_id"] for record in records).encode()).hexdigest(),
        "records": records,
        "limitations": [
            "Hidden kinds are used only for post-hoc diagnostic metrics.",
            "Off-task samples use the maximum class-anchor cosine, a conservative acceptance score.",
            "Scores are measured at the pretrained LoRA initialization and may change during training.",
            "Semantic consistency is evaluated only for AG News clean/conflict samples; off-task detection is separate."
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    summary = {key: value for key, value in result.items()
               if key != "records"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
