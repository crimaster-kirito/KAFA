"""Same-family federated LoRA development baselines on frozen AG News."""

import argparse
import hashlib
import io
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from peft import (LoraConfig, get_peft_model, get_peft_model_state_dict,
                  set_peft_model_state_dict)
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.run_agnews_local_dev import (  # noqa: E402
    evaluate, select_manifest_rows)
from kafa_script.run_real_data_plumbing_smoke import (  # noqa: E402
    collate, encode_training_example, load_jsonl, sha256_file)


def partition_clients(rows, manifest, seed, clients, per_class):
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
            f"fed-dev-v1|{seed}|{row['_id']}".encode()).hexdigest()
        grouped[int(row["label"])].append((rank, row))
    partitions = [[] for _ in range(clients)]
    for label in range(4):
        ranked = [row for _, row in sorted(grouped[label])]
        for client in range(clients):
            start = client * per_class
            partitions[client].extend(ranked[start:start + per_class])
    all_ids = [row["_id"] for partition in partitions for row in partition]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("client partitions overlap")
    return partitions


def clone_state(model):
    return {
        key: value.detach().float().cpu().clone()
        for key, value in get_peft_model_state_dict(model).items()
    }


def factor_fedavg(states):
    return {
        key: torch.stack([state[key] for state in states]).mean(0)
        for key in states[0]
    }


def product_svd(states, rank):
    """Average BA products and return a canonical rank-r factorization."""
    result = {}
    a_keys = [key for key in states[0] if "lora_A" in key]
    for a_key in a_keys:
        b_key = a_key.replace("lora_A", "lora_B")
        if b_key not in states[0]:
            raise KeyError(f"missing paired LoRA B for {a_key}")
        products = [state[b_key] @ state[a_key] for state in states]
        product = torch.stack(products).mean(0)
        u, singular, vh = torch.linalg.svd(product, full_matrices=False)
        kept = min(rank, len(singular))
        root = singular[:kept].clamp_min(0).sqrt()
        b = u[:, :kept] * root.unsqueeze(0)
        a = root.unsqueeze(1) * vh[:kept]
        if kept < rank:
            b = torch.cat([b, torch.zeros(b.shape[0], rank - kept)], dim=1)
            a = torch.cat([a, torch.zeros(rank - kept, a.shape[1])], dim=0)
        result[a_key] = a
        result[b_key] = b
    for key, value in states[0].items():
        if key not in result:
            result[key] = torch.stack([state[key] for state in states]).mean(0)
    return result


def state_bytes(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.tell()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--method", choices=["factor_fedavg", "product_svd"],
                        required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path("kafa_script/benchmark_manifest_v1.json"))
    parser.add_argument("--train-data", type=Path,
                        default=Path(".cache/datasets/agnews/train.jsonl"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--train-per-class", type=int, default=64)
    parser.add_argument("--eval-per-class", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    manifest = json.loads(args.manifest.read_text())
    rows = load_jsonl(args.train_data)
    partitions = partition_clients(rows, manifest, args.seed, args.clients,
                                   args.train_per_class)
    eval_rows = select_manifest_rows(rows, manifest, "development",
                                     args.eval_per_class)
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded_partitions = [[
        encode_training_example(tokenizer, row, args.max_length)
        for row in partition
    ] for partition in partitions]

    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")).to(device)
    global_state = clone_state(model)
    torch.cuda.reset_peak_memory_stats()
    initial = evaluate(model, tokenizer, eval_rows, device, args.max_length,
                       args.eval_batch_size)
    communication = 0
    round_records = []
    started = time.perf_counter()
    for round_index in range(args.rounds):
        local_states, local_losses = [], []
        for client_index, encoded in enumerate(encoded_partitions):
            set_peft_model_state_dict(model, global_state)
            generator = torch.Generator().manual_seed(
                args.seed * 10000 + round_index * 100 + client_index)
            loader = DataLoader(
                encoded, batch_size=args.batch_size, shuffle=True,
                generator=generator,
                collate_fn=lambda batch: collate(tokenizer, batch))
            iterator = iter(loader)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.learning_rate)
            model.train()
            losses = []
            for _ in range(args.local_steps):
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
            state = clone_state(model)
            local_states.append(state)
            local_losses.append(sum(losses) / len(losses))
            communication += state_bytes(global_state) + state_bytes(state)
        if args.method == "factor_fedavg":
            global_state = factor_fedavg(local_states)
        else:
            global_state = product_svd(local_states, args.rank)
        round_records.append({
            "round": round_index + 1,
            "client_mean_losses": local_losses,
        })
    train_seconds = time.perf_counter() - started
    set_peft_model_state_dict(model, global_state)
    final = evaluate(model, tokenizer, eval_rows, device, args.max_length,
                     args.eval_batch_size)
    result = {
        "evidence_label": "benchmark_dev",
        "paper_result": False,
        "method": args.method,
        "seed": args.seed,
        "model": str(args.model),
        "model_weight_sha256": sha256_file(args.model / "model.safetensors"),
        "manifest_sha256": sha256_file(args.manifest),
        "clients": args.clients,
        "examples_per_client": len(partitions[0]),
        "total_unique_train_examples": sum(map(len, partitions)),
        "eval_examples": len(eval_rows),
        "rounds": args.rounds,
        "local_steps": args.local_steps,
        "total_client_optimizer_steps": (args.clients * args.rounds *
                                         args.local_steps),
        "initial": initial,
        "final": final,
        "round_records": round_records,
        "train_wall_seconds": train_seconds,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "serialized_global_adapter_bytes": state_bytes(global_state),
        "measured_total_communication_bytes": communication,
        "client_partition_ids_sha256": [
            hashlib.sha256("\n".join(sorted(row["_id"] for row in partition))
                           .encode()).hexdigest()
            for partition in partitions
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
