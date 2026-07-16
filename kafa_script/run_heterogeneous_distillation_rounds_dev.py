"""Multi-round, tokenizer-agnostic heterogeneous distillation development runner.

This extends the one-round runner by retaining each client's own LoRA state
and the server LoRA state across rounds. It deliberately does not broadcast
server adapter tensors to incompatible clients: the only cross-model message is
the public-anchor canonical probability matrix.
"""

import argparse
import hashlib
import io
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import get_peft_model_state_dict, set_peft_model_state_dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.run_agnews_local_dev import evaluate, select_manifest_rows  # noqa: E402
from kafa_script.run_agnews_mixture_dev import build_mixture  # noqa: E402
from kafa_script.run_real_data_plumbing_smoke import load_jsonl, sha256_file  # noqa: E402
from kafa_script.run_heterogeneous_distillation_dev import (  # noqa: E402
    candidate_loss_matrix, make_lora, route_client_mixture,
    select_disjoint_client_bases, train_client)


def clone_adapter_state(model):
    return {key: value.detach().float().cpu().clone()
            for key, value in get_peft_model_state_dict(model).items()}


def state_bytes(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.tell()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-model", type=Path,
                        default=Path(".cache/models/Qwen2.5-1.5B"))
    parser.add_argument("--client-models", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path("kafa_script/benchmark_manifest_v1.json"))
    parser.add_argument("--agnews-data", type=Path,
                        default=Path(".cache/datasets/agnews/train.jsonl"))
    parser.add_argument("--test-data", type=Path,
                        default=Path(".cache/datasets/agnews/test.jsonl"))
    parser.add_argument("--imdb-data", type=Path,
                        default=Path(".cache/datasets/imdb/train.jsonl"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--client-steps", type=int, default=40)
    parser.add_argument("--server-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--server-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--anchor-score-batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--server-learning-rate", type=float, default=1e-4)
    parser.add_argument("--anchor-supervision-weight", type=float, default=1.0)
    parser.add_argument("--distillation-weight", type=float, default=1.0)
    parser.add_argument("--teacher-aggregation",
                        choices=["uniform", "anchor_accuracy"],
                        default="uniform")
    parser.add_argument("--teacher-temperature", type=float, default=0.1)
    parser.add_argument("--client-routing",
                        choices=["none", "relative_balanced_rank"],
                        default="none")
    parser.add_argument("--routing-keep-fraction", type=float, default=0.5)
    parser.add_argument("--routing-gate-threshold", type=float, default=0.8)
    parser.add_argument("--conflict-ratio", type=float, default=0.4)
    parser.add_argument("--off-task-ratio", type=float, default=0.2)
    parser.add_argument("--anchor-per-class", type=int, default=32)
    parser.add_argument("--eval-per-class", type=int, default=200)
    parser.add_argument("--evaluation-split",
                        choices=["development", "official_test"],
                        default="development")
    parser.add_argument("--evaluate-every-round", action="store_true",
                        help="Evaluate every server round; default evaluates only initial and final round.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds <= 0 or args.client_steps <= 0 or args.server_steps <= 0:
        parser.error("rounds and step counts must be positive")
    if not 0 < args.routing_keep_fraction <= 1:
        parser.error("routing keep fraction must be in (0, 1]")
    if args.teacher_temperature <= 0:
        parser.error("teacher temperature must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    manifest = json.loads(args.manifest.read_text())
    agnews, imdb = load_jsonl(args.agnews_data), load_jsonl(args.imdb_data)
    anchors = select_manifest_rows(agnews, manifest, "anchor",
                                   args.anchor_per_class)
    if args.evaluation_split == "development":
        evaluation_rows = select_manifest_rows(agnews, manifest, "development",
                                               args.eval_per_class)
    else:
        evaluation_rows = load_jsonl(args.test_data)
    bases = select_disjoint_client_bases(agnews, manifest, args.seed,
                                         len(args.client_models))
    mixtures = [build_mixture(base, imdb, args.seed + 1009 * index,
                              args.conflict_ratio, args.off_task_ratio)
                for index, base in enumerate(bases)]
    server_batch_size = args.batch_size if args.server_batch_size is None else args.server_batch_size
    anchor_score_batch_size = (args.eval_batch_size if args.anchor_score_batch_size
                               is None else args.anchor_score_batch_size)

    server, server_tokenizer = make_lora(args.server_model, device)
    server_optimizer = torch.optim.AdamW(
        [parameter for parameter in server.parameters() if parameter.requires_grad],
        lr=args.server_learning_rate)
    initial = evaluate(server, server_tokenizer, evaluation_rows, device,
                       args.max_length, args.eval_batch_size)
    client_states = [None] * len(args.client_models)
    round_records, total_communication = [], 0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for round_index in range(args.rounds):
        probability_matrices, client_records = [], []
        for client_index, (path, mixture) in enumerate(zip(args.client_models,
                                                            mixtures)):
            model, tokenizer = make_lora(path, device)
            if client_states[client_index] is not None:
                set_peft_model_state_dict(model, client_states[client_index])
            if args.client_routing == "relative_balanced_rank":
                selected, routing = route_client_mixture(
                    model, tokenizer, mixture, anchors, device, args.max_length,
                    args.routing_keep_fraction, args.routing_gate_threshold)
            else:
                selected, routing = mixture, {"gate_triggered": False,
                                               "selected": len(mixture)}
            mean_loss = train_client(
                model, tokenizer, selected, device, args.client_steps,
                args.batch_size, args.max_length, args.learning_rate,
                args.seed * 100000 + round_index * 100 + client_index)
            model.eval()
            losses = candidate_loss_matrix(model, tokenizer, anchors, device,
                                           args.max_length,
                                           anchor_score_batch_size)
            probabilities = torch.softmax(-losses.float(), dim=1).cpu()
            payload = io.BytesIO()
            torch.save(probabilities, payload)
            labels = torch.tensor([int(row["label"]) for row in anchors])
            client_states[client_index] = clone_adapter_state(model)
            probability_matrices.append(probabilities)
            client_records.append({
                "model": str(path),
                "mean_local_loss": mean_loss,
                "selected_examples": len(selected),
                "selected_kind_counts": dict(sorted(Counter(
                    item["kind"] for item in selected).items())),
                "routing": routing,
                "anchor_label_accuracy": float((probabilities.argmax(1) == labels).float().mean()),
                "payload_bytes": payload.tell(),
                "adapter_bytes": state_bytes(client_states[client_index]),
            })
            del model
            torch.cuda.empty_cache()

        if args.teacher_aggregation == "uniform":
            weights = torch.full((len(probability_matrices),),
                                 1 / len(probability_matrices))
        else:
            quality = torch.tensor([record["anchor_label_accuracy"]
                                    for record in client_records])
            weights = torch.softmax(quality / args.teacher_temperature, dim=0)
        teacher = sum(weight * probability for weight, probability in
                      zip(weights, probability_matrices))
        order = torch.randperm(len(anchors), generator=torch.Generator().manual_seed(
            args.seed + round_index))
        cursor, server_losses = 0, []
        server.train()
        for step_index in range(args.server_steps):
            indices = order[cursor:cursor + server_batch_size]
            if len(indices) < server_batch_size:
                order = torch.randperm(len(anchors), generator=torch.Generator().manual_seed(
                    args.seed + round_index * 1000 + step_index + 1))
                cursor = 0
                indices = order[:server_batch_size]
            cursor += server_batch_size
            rows = [anchors[int(index)] for index in indices]
            target = teacher[indices].to(device)
            hard = torch.tensor([int(row["label"]) for row in rows], device=device)
            server_optimizer.zero_grad(set_to_none=True)
            losses = candidate_loss_matrix(server, server_tokenizer, rows,
                                           device, args.max_length,
                                           server_batch_size, requires_grad=True)
            distill = -(target * F.log_softmax(-losses, dim=1)).sum(1).mean()
            anchored = F.cross_entropy(-losses, hard)
            objective = (args.distillation_weight * distill +
                         args.anchor_supervision_weight * anchored)
            objective.backward()
            server_optimizer.step()
            server_losses.append(float(objective.detach()))
        if args.evaluate_every_round or round_index == args.rounds - 1:
            server_eval = evaluate(server, server_tokenizer, evaluation_rows,
                                   device, args.max_length,
                                   args.eval_batch_size)
        else:
            server_eval = None
        sent = sum(record["payload_bytes"] for record in client_records)
        total_communication += sent
        round_records.append({
            "round": round_index + 1,
            "client_records": client_records,
            "teacher_weights": weights.tolist(),
            "mean_server_objective": sum(server_losses) / len(server_losses),
            "client_to_server_payload_bytes": sent,
            "server_development": server_eval,
        })
    elapsed = time.perf_counter() - started
    result = {
        "evidence_label": "benchmark_dev",
        "paper_result": False,
        "method": "multi_round_canonical_probability_distillation",
        "seed": args.seed,
        "rounds": args.rounds,
        "client_routing": args.client_routing,
        "client_data": "disjoint target-source partition",
        "manifest_sha256": sha256_file(args.manifest),
        "server_model": str(args.server_model),
        "server_weight_sha256": sha256_file(args.server_model / "model.safetensors"),
        "client_models": [str(path) for path in args.client_models],
        "anchor_examples": len(anchors),
        "evaluation_split": args.evaluation_split,
        "evaluation_examples": len(evaluation_rows),
        "client_steps_per_round": args.client_steps,
        "server_steps_per_round": args.server_steps,
        "client_batch_size": args.batch_size,
        "server_batch_size": server_batch_size,
        "anchor_score_batch_size": anchor_score_batch_size,
        "distillation_weight": args.distillation_weight,
        "anchor_supervision_weight": args.anchor_supervision_weight,
        "teacher_aggregation": args.teacher_aggregation,
        "initial": initial,
        "round_records": round_records,
        "final": round_records[-1]["server_development"],
        "total_client_to_server_payload_bytes": total_communication,
        "total_wall_seconds": elapsed,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "limitations": [
            "Clients retain local adapters but receive no server adapter broadcast because architectures are incompatible.",
            "No privacy claim is made for public-anchor probability exchange.",
            "Development runner only; final claims require frozen seeds and official test evaluation."
        ]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
