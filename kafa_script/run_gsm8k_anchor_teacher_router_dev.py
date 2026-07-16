#!/usr/bin/env python3
"""Development-only public-anchor teacher ranking for GSM8K local adaptation."""

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

from kafa_script.run_gsm8k_local_dev import PROMPT, evaluate, load_jsonl, normalize_numeric  # noqa: E402
from kafa_script.run_gsm8k_mixture_dev import build_mixture, encode_supervised  # noqa: E402
from kafa_script.run_real_data_plumbing_smoke import collate, sha256_file  # noqa: E402


def make_lora(path: Path, device):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16)
    return get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                                             target_modules=["q_proj", "v_proj"],
                                             task_type="CAUSAL_LM")).to(device), tokenizer


def train(model, tokenizer, items, device, steps, batch_size, max_length, lr, seed):
    encoded = [encode_supervised(tokenizer, item["prompt"], item["answer"], max_length) for item in items]
    loader = DataLoader(encoded, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), collate_fn=lambda b: collate(tokenizer, b))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    iterator, losses = iter(loader), []; model.train()
    for _ in range(steps):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        batch = {k:v.to(device) for k,v in batch.items()}; optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss; loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    return sum(losses) / len(losses)


@torch.no_grad()
def loss_for(model, tokenizer, item, device, max_length):
    batch = collate(tokenizer, [encode_supervised(tokenizer, item["prompt"], item["answer"], max_length)])
    return float(model(**{k:v.to(device) for k,v in batch.items()}).loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model", type=Path, default=Path(".cache/models/Qwen2.5-1.5B"))
    parser.add_argument("--client-model", type=Path, default=Path(".cache/models/Qwen2.5-0.5B"))
    parser.add_argument("--manifest", type=Path, default=Path("kafa_script/gsm8k_benchmark_manifest_v1.json"))
    parser.add_argument("--train-data", type=Path, default=Path(".cache/datasets/gsm8k/processed/train.jsonl"))
    parser.add_argument("--test-data", type=Path, default=Path(".cache/datasets/gsm8k/processed/test.jsonl"))
    parser.add_argument("--secondary-manifest", type=Path, default=Path("kafa_script/gsm8k_secondary_holdout_manifest_v1.json"))
    parser.add_argument("--squad-data", type=Path, default=Path(".cache/datasets/squad/test.jsonl"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--client-index", type=int, default=0)
    parser.add_argument("--conflict-ratio", type=float, default=.4)
    parser.add_argument("--off-task-ratio", type=float, default=.2)
    parser.add_argument("--anchor-size", type=int, default=128)
    parser.add_argument("--teacher-steps", type=int, default=40)
    parser.add_argument("--client-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--eval-size", type=int, default=64)
    parser.add_argument("--evaluation-split", choices=["development", "secondary_holdout", "official_test"], default="development")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--keep-count", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    total_started=time.perf_counter()
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    manifest=json.loads(args.manifest.read_text()); rows=load_jsonl(args.train_data); by={r["source_index"]:r for r in rows}
    base=[by[i] for i in manifest["partitions"]["client_pools"][args.client_index]]
    anchors=[{"prompt":PROMPT.format(question=by[i]["question"]),"answer":by[i]["final_answer"]} for i in manifest["partitions"]["anchor"][:args.anchor_size]]
    if args.evaluation_split == "development":
        dev=[by[i] for i in manifest["partitions"]["development"][:args.eval_size]]
    elif args.evaluation_split == "official_test":
        dev=load_jsonl(args.test_data)
    else:
        secondary=json.loads(args.secondary_manifest.read_text())
        dev=[by[i] for i in secondary["source_indices"]]
        if set(secondary["source_indices"]) & (set(manifest["partitions"]["anchor"]) | set(manifest["partitions"]["client_pools"][args.client_index])):
            raise AssertionError("secondary holdout overlaps anchor/client training data")
    squad=[json.loads(x) for x in args.squad_data.read_text().split("\n") if x]
    mixture=build_mixture(base,squad,args.seed,args.client_index,args.conflict_ratio,args.off_task_ratio)
    device=torch.device("cuda"); torch.cuda.reset_peak_memory_stats()
    teacher,ttok=make_lora(args.teacher_model,device)
    teacher_train_started=time.perf_counter()
    teacher_train_loss=train(teacher,ttok,anchors,device,args.teacher_steps,args.batch_size,args.max_length,args.learning_rate,args.seed)
    teacher_train_elapsed=time.perf_counter()-teacher_train_started
    scoring_started=time.perf_counter()
    scored=[]
    for index,item in enumerate(mixture):
        if normalize_numeric(item["answer"]) is not None:
            scored.append((loss_for(teacher,ttok,item,device,args.max_length),index))
    scoring_elapsed=time.perf_counter()-scoring_started
    if len(scored)<args.keep_count: raise ValueError("not enough numeric-interface examples")
    selected_indices=sorted(scored,key=lambda x:(x[0],mixture[x[1]]["source_id"]))[:args.keep_count]
    selected=[mixture[index] for _,index in selected_indices]
    teacher_scores={index:score for score,index in scored}
    del teacher; torch.cuda.empty_cache()
    client,ctok=make_lora(args.client_model,device)
    initial=None if args.skip_initial_eval else evaluate(client,ctok,dev,device,args.max_length,args.eval_batch_size,args.max_new_tokens)
    started=time.perf_counter(); client_train_loss=train(client,ctok,selected,device,args.client_steps,args.batch_size,args.max_length,args.learning_rate,args.seed); elapsed=time.perf_counter()-started
    final=evaluate(client,ctok,dev,device,args.max_length,args.eval_batch_size,args.max_new_tokens)
    total_elapsed=time.perf_counter()-total_started
    result={"evidence_label":"benchmark_dev" if args.evaluation_split == "development" else "benchmark_holdout","paper_result":False,"method":"gsm8k_public_anchor_teacher_numeric_rank","seed":args.seed,"teacher_model":str(args.teacher_model),"teacher_weight_sha256":sha256_file(args.teacher_model / "model.safetensors"),"client_model":str(args.client_model),"client_weight_sha256":sha256_file(args.client_model / "model.safetensors"),"manifest_sha256":sha256_file(args.manifest),"squad_sha256":sha256_file(args.squad_data),"evaluation_manifest_sha256":sha256_file(args.secondary_manifest) if args.evaluation_split == "secondary_holdout" else None,"anchor_examples":len(anchors),"teacher_steps":args.teacher_steps,"client_steps":args.client_steps,"teacher_mean_train_loss":teacher_train_loss,"teacher_train_wall_seconds":teacher_train_elapsed,"teacher_scoring_wall_seconds":scoring_elapsed,"client_mean_train_loss":client_train_loss,"mixture_counts":dict(sorted(Counter(x["kind"] for x in mixture).items())),"selection":{"keep_count":args.keep_count,"numeric_eligible":len(scored),"selected_kind_counts":dict(sorted(Counter(x["kind"] for x in selected).items())),"selected_source_ids_sha256":hashlib.sha256("\n".join(x["source_id"] for x in selected).encode()).hexdigest(),"selected_teacher_loss_range":[min(score for score,_ in selected_indices),max(score for score,_ in selected_indices)]},"evaluation_split":args.evaluation_split,"evaluation_examples":len(dev),"initial":initial,"final":final,"client_train_wall_seconds":elapsed,"total_wall_seconds":total_elapsed,"peak_allocated_gpu_bytes":torch.cuda.max_memory_allocated(),"limitations":["Official-test labels are loaded only when explicitly requested under a frozen final protocol.","Selection audit labels are recorded only after selection and never used to rank examples."]}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))


if __name__=="__main__": main()
