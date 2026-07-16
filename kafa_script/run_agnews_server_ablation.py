"""Post-primary AG News four-arm server-transfer ablation.

The runner evaluates a newly frozen train-derived holdout and excludes every
holdout record from all client pools.  It imports the established AG News
training primitives without modifying the frozen primary runner or artifacts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import set_peft_model_state_dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script import run_generic_classification_distillation as audit  # noqa: E402
from kafa_script.run_agnews_local_dev import evaluate, select_manifest_rows  # noqa: E402
from kafa_script.run_agnews_mixture_dev import build_mixture  # noqa: E402
from kafa_script.run_real_data_plumbing_smoke import load_jsonl  # noqa: E402
from kafa_script.run_heterogeneous_distillation_dev import (  # noqa: E402
    candidate_loss_matrix, make_lora, route_client_mixture,
    select_disjoint_client_bases, train_client)
from kafa_script.run_heterogeneous_distillation_rounds_dev import (  # noqa: E402
    clone_adapter_state, state_bytes)


METHODS = {
    "public_anchor_only": {
        "conflict_ratio": 0.4,
        "off_task_ratio": 0.2,
        "selection": "no_client_path",
    },
    "clean_unfiltered_distillation": {
        "conflict_ratio": 0.0,
        "off_task_ratio": 0.0,
        "selection": "all_examples",
    },
    "severe_unfiltered_distillation": {
        "conflict_ratio": 0.4,
        "off_task_ratio": 0.2,
        "selection": "all_examples",
    },
    "severe_routed_distillation": {
        "conflict_ratio": 0.4,
        "off_task_ratio": 0.2,
        "selection": "relative_balanced_routing",
    },
}


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ids_sha256(rows) -> str:
    return hashlib.sha256("\n".join(sorted(
        str(row.get("_id", row.get("source_id"))) for row in rows
    )).encode("ascii")).hexdigest()


def mixture_sha256(mixture) -> str:
    fields = []
    for item in mixture:
        fields.append({key: item[key] for key in sorted(item)
                       if key not in {"text"}})
    return canonical_sha256(fields)


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("name") != "agnews_server_post_primary_ablation_v1":
        raise ValueError("unexpected AG News ablation protocol")
    return protocol


def validate_contract(args, protocol: dict, parent: dict,
                      holdout: dict) -> dict:
    files = protocol["implementation"]["file_sha256"]
    for relative, expected in files.items():
        observed = audit.sha256_file(ROOT / relative)
        if observed != expected:
            raise ValueError(f"contract hash mismatch for {relative}: {observed}")
    if audit.sha256_file(args.parent_manifest) != protocol[
            "data"]["parent_manifest_sha256"]:
        raise ValueError("parent manifest hash mismatch")
    if audit.sha256_file(args.holdout_manifest) != protocol[
            "data"]["holdout_manifest_sha256"]:
        raise ValueError("holdout manifest hash mismatch")
    if audit.sha256_file(args.agnews_data) != protocol[
            "data"]["agnews_train_sha256"]:
        raise ValueError("AG News train hash mismatch")
    if audit.sha256_file(args.imdb_data) != protocol[
            "data"]["imdb_train_sha256"]:
        raise ValueError("IMDb train hash mismatch")
    if holdout["parent_manifest_sha256"] != protocol[
            "data"]["parent_manifest_sha256"]:
        raise ValueError("holdout parent link mismatch")
    if parent["datasets"]["agnews_train"]["sha256"] != protocol[
            "data"]["agnews_train_sha256"]:
        raise ValueError("parent dataset link mismatch")
    if holdout["holdout"]["ids_sha256"] != protocol[
            "data"]["secondary_holdout_ids_sha256"]:
        raise ValueError("holdout ID ledger mismatch")
    for record in protocol["parent_protocols"].values():
        if audit.sha256_file(ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"parent protocol hash mismatch: {record['path']}")
    for relative, expected in protocol["model_weight_sha256"].items():
        if audit.sha256_file(ROOT / relative) != expected:
            raise ValueError(f"model hash mismatch for {relative}")
    phase = ("development" if args.evaluation_split == "development" else
             "secondary_holdout")
    if args.seed not in protocol[phase]["seeds"]:
        raise ValueError(f"seed {args.seed} is not frozen for {phase}")
    if args.method not in protocol[phase]["comparators"]:
        raise ValueError(f"method {args.method} is not frozen for {phase}")
    if METHODS[args.method] != protocol["method_configuration"][args.method]:
        raise ValueError("method configuration differs from protocol")
    return protocol["fixed_configuration"]


def resolve_holdout(rows, manifest: dict) -> list[dict]:
    by_id = {row["_id"]: row for row in rows}
    requested = [record["id"] for record in manifest["holdout"]["records"]]
    missing = [record_id for record_id in requested if record_id not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} frozen holdout IDs are missing")
    selected = [by_id[record_id] for record_id in requested]
    if ids_sha256(selected) != manifest["holdout"]["ids_sha256"]:
        raise ValueError("resolved holdout ID hash mismatch")
    return selected


def train_server_round(server, tokenizer, optimizer, anchors, teacher,
                       fixed: dict, seed: int, round_index: int,
                       device) -> float:
    batch_size = fixed["server_batch_size"]
    order = torch.randperm(len(anchors), generator=torch.Generator().manual_seed(
        seed + round_index))
    cursor, objectives = 0, []
    server.train()
    for step_index in range(fixed["server_steps_per_round"]):
        indices = order[cursor:cursor + batch_size]
        if len(indices) < batch_size:
            order = torch.randperm(
                len(anchors), generator=torch.Generator().manual_seed(
                    seed + round_index * 1000 + step_index + 1))
            cursor = 0
            indices = order[:batch_size]
        cursor += batch_size
        rows = [anchors[int(index)] for index in indices]
        hard = torch.tensor([int(row["label"]) for row in rows], device=device)
        optimizer.zero_grad(set_to_none=True)
        losses = candidate_loss_matrix(
            server, tokenizer, rows, device, fixed["max_length"], batch_size,
            requires_grad=True)
        objective = fixed["anchor_supervision_weight"] * F.cross_entropy(
            -losses, hard)
        if teacher is not None:
            target = teacher[indices].to(device)
            distill = -(target * F.log_softmax(-losses, dim=1)).sum(1).mean()
            objective = objective + fixed["distillation_weight"] * distill
        objective.backward()
        optimizer.step()
        objectives.append(float(objective.detach()))
    return sum(objectives) / len(objectives)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path(
        "kafa_script/agnews_server_post_primary_ablation_v1.json"))
    parser.add_argument("--parent-manifest", type=Path, default=Path(
        "kafa_script/benchmark_manifest_v1.json"))
    parser.add_argument("--holdout-manifest", type=Path, default=Path(
        "kafa_script/agnews_secondary_holdout_manifest_v1.json"))
    parser.add_argument("--agnews-data", type=Path, default=Path(
        ".cache/datasets/agnews/train.jsonl"))
    parser.add_argument("--imdb-data", type=Path, default=Path(
        ".cache/datasets/imdb/train.jsonl"))
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--evaluation-split", required=True,
                        choices=["development", "secondary_holdout"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required for the frozen benchmark")

    protocol = load_protocol(args.protocol)
    parent = json.loads(args.parent_manifest.read_text(encoding="utf-8"))
    holdout = json.loads(args.holdout_manifest.read_text(encoding="utf-8"))
    fixed = validate_contract(args, protocol, parent, holdout)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    started_utc = datetime.now(timezone.utc).isoformat()

    agnews = load_jsonl(args.agnews_data)
    imdb = load_jsonl(args.imdb_data)
    anchors = select_manifest_rows(
        agnews, parent, "anchor", fixed["anchor_per_class"])
    frozen_holdout = resolve_holdout(agnews, holdout)
    if args.evaluation_split == "development":
        evaluation_rows = select_manifest_rows(
            agnews, parent, "development", fixed["development_per_class"])
    else:
        evaluation_rows = frozen_holdout

    allocation_manifest = copy.deepcopy(parent)
    allocation_manifest["splits"]["development"]["records"] = (
        parent["splits"]["development"]["records"] +
        holdout["holdout"]["records"])
    client_bases = select_disjoint_client_bases(
        agnews, allocation_manifest, args.seed,
        len(fixed["client_models"]), fixed["client_target_per_class"])
    holdout_ids = {row["_id"] for row in frozen_holdout}
    if any(holdout_ids & {row["_id"] for row in base}
           for base in client_bases):
        raise AssertionError("client allocation overlaps frozen holdout")

    method_configuration = METHODS[args.method]
    mixtures = [build_mixture(
        base, imdb, args.seed + 1009 * index,
        method_configuration["conflict_ratio"],
        method_configuration["off_task_ratio"])
        for index, base in enumerate(client_bases)]

    server_path = ROOT / fixed["server_model"]
    client_paths = [ROOT / path for path in fixed["client_models"]]
    server, server_tokenizer = make_lora(server_path, device)
    server_optimizer = torch.optim.AdamW(
        [parameter for parameter in server.parameters()
         if parameter.requires_grad], lr=fixed["server_learning_rate"])
    client_states = [None] * len(client_paths)
    round_records, total_payload = [], 0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    for round_index in range(fixed["rounds"]):
        probability_matrices, client_records = [], []
        execute_clients = method_configuration["selection"] != "no_client_path"
        iterator = enumerate(zip(client_paths, mixtures)) if execute_clients else []
        for client_index, (path, mixture) in iterator:
            model, tokenizer = make_lora(path, device)
            if client_states[client_index] is not None:
                set_peft_model_state_dict(model, client_states[client_index])
            if method_configuration["selection"] == "relative_balanced_routing":
                selected, selection = route_client_mixture(
                    model, tokenizer, mixture, anchors, device,
                    fixed["max_length"], fixed["routing_keep_fraction"],
                    fixed["routing_gate_threshold"])
                selection = {"policy": "relative_balanced_routing", **selection}
            else:
                selected = mixture
                selection = {
                    "policy": "all_examples",
                    "gate_triggered": False,
                    "selected": len(selected),
                }
            mean_loss = train_client(
                model, tokenizer, selected, device,
                fixed["client_steps_per_round"], fixed["client_batch_size"],
                fixed["max_length"], fixed["client_learning_rate"],
                args.seed * 100000 + round_index * 100 + client_index)
            model.eval()
            losses = candidate_loss_matrix(
                model, tokenizer, anchors, device, fixed["max_length"],
                fixed["anchor_score_batch_size"])
            probabilities = torch.softmax(-losses.float(), dim=1).cpu()
            payload = io.BytesIO()
            torch.save(probabilities, payload)
            client_states[client_index] = clone_adapter_state(model)
            probability_matrices.append(probabilities)
            client_records.append({
                "client_index": client_index,
                "model": str(path.relative_to(ROOT)),
                "base_target_ids_sha256": ids_sha256(client_bases[client_index]),
                "mixture_assignment_sha256": mixture_sha256(mixture),
                "selected_ids_sha256": ids_sha256(selected),
                "mixture_kind_counts": dict(sorted(Counter(
                    item["kind"] for item in mixture).items())),
                "selected_kind_counts_audit_only": dict(sorted(Counter(
                    item["kind"] for item in selected).items())),
                "mean_local_loss": mean_loss,
                "selection": selection,
                "probability_shape": list(probabilities.shape),
                "probability_dtype": str(probabilities.dtype),
                "probability_sha256": hashlib.sha256(
                    probabilities.contiguous().numpy().tobytes()).hexdigest(),
                "serialized_probability_payload_bytes": payload.tell(),
                "adapter_bytes": state_bytes(client_states[client_index]),
            })
            del model
            torch.cuda.empty_cache()
        teacher = (sum(probability_matrices) / len(probability_matrices)
                   if probability_matrices else None)
        total_payload += sum(record["serialized_probability_payload_bytes"]
                             for record in client_records)
        mean_objective = train_server_round(
            server, server_tokenizer, server_optimizer, anchors, teacher,
            fixed, args.seed, round_index, device)
        round_records.append({
            "round": round_index + 1,
            "client_records": client_records,
            "teacher_weights": ([] if not probability_matrices else
                                [1 / len(probability_matrices)] *
                                len(probability_matrices)),
            "mean_server_objective": mean_objective,
        })

    final = evaluate(server, server_tokenizer, evaluation_rows, device,
                     fixed["max_length"], fixed["evaluation_batch_size"])
    elapsed = time.perf_counter() - started
    full_args = {
        "protocol": str(args.protocol),
        "parent_manifest": str(args.parent_manifest),
        "holdout_manifest": str(args.holdout_manifest),
        "agnews_data": str(args.agnews_data),
        "imdb_data": str(args.imdb_data),
        "method": args.method,
        "evaluation_split": args.evaluation_split,
        "seed": args.seed,
        "output": str(args.output),
    }
    result = {
        "schema_version": "agnews_server_ablation_v1",
        "evidence_label": ("post_primary_development" if
                           args.evaluation_split == "development" else
                           "post_primary_run_disjoint_resampling_ablation"),
        "paper_result": args.evaluation_split == "secondary_holdout",
        "method": args.method,
        "method_configuration": method_configuration,
        "seed": args.seed,
        "started_utc": started_utc,
        "protocol_sha256": audit.sha256_file(args.protocol),
        "runner_file_sha256": audit.sha256_file(Path(__file__)),
        "parent_manifest_sha256": audit.sha256_file(args.parent_manifest),
        "holdout_manifest_sha256": audit.sha256_file(args.holdout_manifest),
        "full_args": full_args,
        "full_args_sha256": canonical_sha256(full_args),
        "fixed_configuration": fixed,
        "git": audit.git_state(),
        "datasets": {
            "agnews_train_sha256": audit.sha256_file(args.agnews_data),
            "imdb_train_sha256": audit.sha256_file(args.imdb_data),
            "official_test_accessed": False,
        },
        "model_artifacts": {
            "server": audit.artifact_hashes(server_path),
            "clients": [audit.artifact_hashes(path) for path in client_paths],
        },
        "split_source_ids_sha256": {
            "anchor": ids_sha256(anchors),
            "evaluation": ids_sha256(evaluation_rows),
            "frozen_holdout": ids_sha256(frozen_holdout),
            **{f"client_{index}": ids_sha256(base)
               for index, base in enumerate(client_bases)},
        },
        "mixture_assignment_sha256": [
            mixture_sha256(mixture) for mixture in mixtures],
        "evaluation_split": args.evaluation_split,
        "evaluation_examples": len(evaluation_rows),
        "anchor_examples": len(anchors),
        "round_records": round_records,
        "final": final,
        "total_client_to_server_payload_bytes": total_payload,
        "wall_seconds": elapsed,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "evidence_boundaries": protocol["evidence_boundaries"],
    }
    if not all(math.isfinite(float(final[key])) for key in
               ("accuracy", "macro_f1", "label_nll", "ece")):
        raise ValueError("non-finite final metric")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
