#!/usr/bin/env python3
"""Audit an anchor-trained independent teacher as a GSM8K trust signal."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.diagnose_anchor_gradient_separation import auroc, average_precision, summary_quantiles  # noqa: E402
from kafa_script.run_gsm8k_local_dev import PROMPT, load_jsonl  # noqa: E402
from kafa_script.run_gsm8k_mixture_dev import build_mixture, encode_supervised  # noqa: E402
from kafa_script.run_real_data_plumbing_smoke import collate, sha256_file  # noqa: E402


@torch.no_grad()
def example_loss(model, tokenizer, prompt: str, answer: str, device, max_length: int) -> float:
    batch = collate(tokenizer, [encode_supervised(tokenizer, prompt, answer, max_length)])
    batch = {key: value.to(device) for key, value in batch.items()}
    return float(model(**batch).loss)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model", type=Path, default=Path(".cache/models/Qwen2.5-1.5B"))
    parser.add_argument("--manifest", type=Path, default=Path("kafa_script/gsm8k_benchmark_manifest_v1.json"))
    parser.add_argument("--train-data", type=Path, default=Path(".cache/datasets/gsm8k/processed/train.jsonl"))
    parser.add_argument("--squad-data", type=Path, default=Path(".cache/datasets/squad/test.jsonl"))
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--client-index", type=int, default=0)
    parser.add_argument("--anchor-size", type=int, default=128)
    parser.add_argument("--teacher-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    manifest = json.loads(args.manifest.read_text())
    rows = load_jsonl(args.train_data); by_index = {row["source_index"]: row for row in rows}
    anchors = [by_index[i] for i in manifest["partitions"]["anchor"][:args.anchor_size]]
    base = [by_index[i] for i in manifest["partitions"]["client_pools"][args.client_index]]
    squad = [json.loads(line) for line in args.squad_data.read_text().split("\n") if line]
    mixture = build_mixture(base, squad, args.seed, args.client_index, .4, .2)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.teacher_model, local_files_only=True, dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")).to(device)
    encoded = [encode_supervised(tokenizer, PROMPT.format(question=row["question"]), row["final_answer"], args.max_length) for row in anchors]
    loader = DataLoader(encoded, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), collate_fn=lambda b: collate(tokenizer, b))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    iterator = iter(loader); model.train(); train_losses=[]
    for _ in range(args.teacher_steps):
        try: batch=next(iterator)
        except StopIteration: iterator=iter(loader); batch=next(iterator)
        batch={k:v.to(device) for k,v in batch.items()}; optimizer.zero_grad(set_to_none=True)
        loss=model(**batch).loss; loss.backward(); optimizer.step(); train_losses.append(float(loss.detach()))
    model.eval(); records=[]
    for item in mixture:
        records.append({"kind": item["kind"], "teacher_loss": example_loss(model, tokenizer, item["prompt"], item["answer"], device, args.max_length)})
    def report(positive: set[str]) -> dict:
        scores=[r["teacher_loss"] for r in records]; labels=[r["kind"] in positive for r in records]
        return {"positive_kinds":sorted(positive), "positives":sum(labels), "auroc":auroc(scores,labels), "average_precision":average_precision(scores,labels)}
    by_kind=defaultdict(list)
    for record in records: by_kind[record["kind"]].append(record["teacher_loss"])
    result={"evidence_label":"benchmark_dev", "paper_result":False, "method":"gsm8k_anchor_teacher_signal_diagnostic", "seed":args.seed, "teacher_model":str(args.teacher_model), "teacher_weight_sha256":sha256_file(args.teacher_model / "model.safetensors"), "manifest_sha256":sha256_file(args.manifest), "squad_sha256":sha256_file(args.squad_data), "anchor_examples":len(anchors), "teacher_steps":args.teacher_steps, "mean_teacher_anchor_train_loss":sum(train_losses)/len(train_losses), "mixture_counts":{kind:len(values) for kind,values in sorted(by_kind.items())}, "teacher_loss_audit":{"conflict":report({"conflict"}),"off_task":report({"off_task"}),"harmful":report({"conflict","off_task"}),"by_kind_quantiles":{kind:summary_quantiles(values) for kind,values in sorted(by_kind.items())}}, "limitations":["Diagnostic only; no teacher score selects examples in this run.","Teacher sees only public anchors and no official GSM8K test labels."]}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
