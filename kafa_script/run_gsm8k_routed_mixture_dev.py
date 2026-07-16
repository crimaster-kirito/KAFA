#!/usr/bin/env python3
"""Development-only GSM8K relative-anchor-gradient routing baseline."""

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
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.diagnose_anchor_gradient_separation import (  # noqa: E402
    discrimination, gradient_cosine, parameter_gradients, rank_percentiles,
    summary_quantiles)
from kafa_script.run_gsm8k_local_dev import evaluate, load_jsonl, normalize_numeric  # noqa: E402
from kafa_script.run_gsm8k_mixture_dev import (  # noqa: E402
    build_mixture, encode_supervised)
from kafa_script.run_real_data_plumbing_smoke import collate, sha256_file  # noqa: E402


def gradient_and_loss(model, tokenizer, prompt: str, answer: str, device,
                      max_length: int, parameters) -> tuple[list[torch.Tensor], float]:
    batch = collate(tokenizer, [encode_supervised(tokenizer, prompt, answer, max_length)])
    batch = {key: value.to(device) for key, value in batch.items()}
    model.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    loss.backward()
    return parameter_gradients(parameters), float(loss.detach())


def mean_gradients(gradients: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    return [sum(parts) / len(parts) for parts in zip(*gradients)]


def route(model, tokenizer, mixture: list[dict], anchors: list[dict], device,
          max_length: int, threshold: float, keep_fraction: float,
          numeric_interface_only: bool = False,
          risk_mode: str = "combined") -> tuple[list[dict], dict]:
    """Route purely from model/public-anchor signals; kinds are recorded only after selection."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    anchor_records = []
    for row in anchors:
        gradient, loss = gradient_and_loss(model, tokenizer, row["prompt"],
                                           row["answer"], device, max_length,
                                           parameters)
        anchor_records.append((gradient, loss))
    reference_gradient = mean_gradients([gradient for gradient, _ in anchor_records])
    reference_cosines = [gradient_cosine(gradient, reference_gradient)
                         for gradient, _ in anchor_records]
    records = []
    for item in mixture:
        gradient, loss = gradient_and_loss(model, tokenizer, item["prompt"],
                                           item["answer"], device, max_length,
                                           parameters)
        records.append({"source_id": item["source_id"], "anchor_cosine": gradient_cosine(gradient, reference_gradient),
                        "loss": loss, "kind": item["kind"],
                        "numeric_answer": normalize_numeric(item["answer"]) is not None})
    client_q25 = summary_quantiles([record["anchor_cosine"] for record in records])["q25"]
    reference_q25 = summary_quantiles(reference_cosines)["q25"]
    ratio = client_q25 / max(reference_q25, 1e-12)
    if ratio >= threshold:
        audit = {"conflict": discrimination(records, {"conflict"}),
                 "off_task": discrimination(records, {"off_task"}),
                 "harmful": discrimination(records, {"conflict", "off_task"})}
        return mixture, {"gate_triggered": False, "gate_ratio": ratio,
                         "client_q25": client_q25, "reference_q25": reference_q25,
                         "selected": len(mixture), "selected_kind_counts": dict(sorted(Counter(i["kind"] for i in mixture).items())),
                         "audit_not_used_for_selection": audit}
    if risk_mode == "combined":
        scores = [(a + b) / 2 for a, b in zip(
            rank_percentiles([-record["anchor_cosine"] for record in records]),
            rank_percentiles([record["loss"] for record in records]))]
    elif risk_mode == "loss_only":
        scores = rank_percentiles([record["loss"] for record in records])
    else:
        raise ValueError(f"Unknown risk mode: {risk_mode}")
    count = max(1, round(len(mixture) * keep_fraction))
    eligible = [i for i, record in enumerate(records)
                if not numeric_interface_only or record["numeric_answer"]]
    if len(eligible) < count:
        raise ValueError("numeric interface filter left too few examples")
    selected_indices = sorted(eligible, key=lambda i: (scores[i], records[i]["source_id"]))[:count]
    selected = [mixture[i] for i in selected_indices]
    audit = {"conflict": discrimination(records, {"conflict"}),
             "off_task": discrimination(records, {"off_task"}),
             "harmful": discrimination(records, {"conflict", "off_task"})}
    return selected, {"gate_triggered": True, "gate_ratio": ratio,
                      "client_q25": client_q25, "reference_q25": reference_q25,
                      "selected": len(selected), "numeric_interface_only": numeric_interface_only,
                      "risk_mode": risk_mode,
                      "eligible_before_rank": len(eligible),
                      "selected_kind_counts": dict(sorted(Counter(i["kind"] for i in selected).items())),
                      "record_order_sha256": hashlib.sha256("\n".join(r["source_id"] for r in records).encode()).hexdigest(),
                      "audit_not_used_for_selection": audit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("kafa_script/gsm8k_benchmark_manifest_v1.json"))
    parser.add_argument("--train-data", type=Path, default=Path(".cache/datasets/gsm8k/processed/train.jsonl"))
    parser.add_argument("--squad-data", type=Path, default=Path(".cache/datasets/squad/test.jsonl"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--client-index", type=int, default=0)
    parser.add_argument("--conflict-ratio", type=float, default=0.4)
    parser.add_argument("--off-task-ratio", type=float, default=0.2)
    parser.add_argument("--router-anchor-size", type=int, default=16)
    parser.add_argument("--routing-threshold", type=float, default=0.8)
    parser.add_argument("--routing-keep-fraction", type=float, default=0.5)
    parser.add_argument("--numeric-interface-only", action="store_true")
    parser.add_argument("--risk-mode", choices=["combined", "loss_only"], default="combined")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.routing_keep_fraction <= 1 or args.router_anchor_size <= 0:
        parser.error("invalid routing fraction or anchor count")
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    manifest = json.loads(args.manifest.read_text())
    train = load_jsonl(args.train_data); by_index = {row["source_index"]: row for row in train}
    pools = manifest["partitions"]["client_pools"]
    base = [by_index[i] for i in pools[args.client_index]]
    dev = [by_index[i] for i in manifest["partitions"]["development"][:args.eval_size]]
    anchor_rows = [by_index[i] for i in manifest["partitions"]["anchor"][:args.router_anchor_size]]
    anchors = [{"prompt": "Solve the following grade-school math problem. Reply with only the final numeric answer.\n\nQuestion: " + row["question"] + "\nAnswer:", "answer": row["final_answer"]} for row in anchor_rows]
    squad = [json.loads(x) for x in args.squad_data.read_text().split("\n") if x]
    mixture = build_mixture(base, squad, args.seed, args.client_index, args.conflict_ratio, args.off_task_ratio)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")).to(device)
    torch.cuda.reset_peak_memory_stats()
    initial = evaluate(model, tokenizer, dev, device, args.max_length, args.eval_batch_size, args.max_new_tokens)
    selected, routing = route(model, tokenizer, mixture, anchors, device, args.max_length, args.routing_threshold, args.routing_keep_fraction, args.numeric_interface_only, args.risk_mode)
    encoded = [encode_supervised(tokenizer, item["prompt"], item["answer"], args.max_length) for item in selected]
    loader = DataLoader(encoded, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), collate_fn=lambda b: collate(tokenizer, b))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    iterator, losses = iter(loader), []; started = time.perf_counter(); model.train()
    for _ in range(args.steps):
        try: batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}; optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss; loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    wall = time.perf_counter() - started
    final = evaluate(model, tokenizer, dev, device, args.max_length, args.eval_batch_size, args.max_new_tokens)
    result = {"evidence_label":"benchmark_dev", "paper_result":False, "method":"gsm8k_relative_anchor_gradient_rank", "seed":args.seed, "model":str(args.model), "model_weight_sha256":sha256_file(args.model / "model.safetensors"), "manifest_sha256":sha256_file(args.manifest), "squad_sha256":sha256_file(args.squad_data), "client_index":args.client_index, "mixture_counts":dict(sorted(Counter(i["kind"] for i in mixture).items())), "routing":routing, "router_anchor_size":args.router_anchor_size, "routing_threshold":args.routing_threshold, "routing_keep_fraction":args.routing_keep_fraction, "numeric_interface_only":args.numeric_interface_only, "risk_mode":args.risk_mode, "development_examples":len(dev), "steps":args.steps, "initial":initial, "final":final, "mean_train_loss":sum(losses)/len(losses), "train_wall_seconds":wall, "peak_allocated_gpu_bytes":torch.cuda.max_memory_allocated(), "limitations":["Development-only; official GSM8K test is not loaded.", "Selection is a candidate diagnostic and uses no contamination kind."]}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2)+"\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
