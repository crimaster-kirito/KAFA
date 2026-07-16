"""Post-primary TREC server ablations with a fail-closed frozen contract.

This runner intentionally supports only the two missing causal controls:
distillation from clean, unfiltered clients and distillation from severe,
unfiltered clients.  It imports the frozen v1 task/training primitives without
modifying the primary runner or its already reported artifacts.
"""

from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.classification_task import (  # noqa: E402
    audit_probability_payload, canonical_sha256, load_task_spec,
    mixture_assignment_sha256)
from kafa_script import run_generic_classification_distillation as core  # noqa: E402


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


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("name") != "trec_server_post_primary_ablation_v1":
        raise ValueError("unexpected ablation protocol name")
    return protocol


def validate_contract(args, protocol: dict, task_spec, off_task_spec,
                      manifest: dict) -> dict:
    implementation = protocol["implementation"]
    expected_files = {
        Path(__file__): implementation["ablation_runner_file_sha256"],
        ROOT / implementation["frozen_primary_runner"]:
            implementation["frozen_primary_runner_file_sha256"],
        ROOT / implementation["classification_task_core"]:
            implementation["classification_task_file_sha256"],
        args.manifest: protocol["task"]["manifest_file_sha256"],
        args.target_train: protocol["task"]["target_train_sha256"],
        args.off_task_train: protocol["task"]["off_task_train_sha256"],
    }
    for path, expected in expected_files.items():
        observed = core.sha256_file(path)
        if observed != expected:
            raise ValueError(f"contract hash mismatch for {path}: {observed}")
    if core.sha256_file(ROOT / protocol["parent_primary_protocol"]["path"]) != \
            protocol["parent_primary_protocol"]["sha256"]:
        raise ValueError("parent primary protocol hash mismatch")
    for relative_path, expected in protocol["model_weight_sha256"].items():
        if core.sha256_file(ROOT / relative_path) != expected:
            raise ValueError(f"model weight hash mismatch for {relative_path}")
    if task_spec.sha256 != protocol["task"]["task_spec_semantic_sha256"]:
        raise ValueError("target task semantic hash mismatch")
    if off_task_spec.sha256 != protocol["task"]["off_task_spec_semantic_sha256"]:
        raise ValueError("off-task semantic hash mismatch")
    if manifest["datasets"]["target_official_test"]["sha256"] != \
            protocol["task"]["target_official_test_sha256"]:
        raise ValueError("manifest official-test hash mismatch")
    phase = "development" if args.evaluation_split == "development" \
        else "official_evaluation"
    if args.seed not in protocol[phase]["seeds"]:
        raise ValueError(f"seed {args.seed} is not frozen for {phase}")
    if args.method not in protocol[phase]["comparators"]:
        raise ValueError(f"method {args.method} is not frozen for {phase}")
    observed_method = METHODS[args.method]
    if observed_method != protocol["method_configuration"][args.method]:
        raise ValueError("method configuration differs from runner contract")
    fixed = protocol["fixed_configuration"]
    if len(fixed["client_models"]) != 3:
        raise ValueError("protocol must define exactly three clients")
    return fixed


def train_server_round(server, tokenizer, optimizer, anchors, task_spec,
                       teacher: torch.Tensor | None, fixed: dict, seed: int,
                       round_index: int, device) -> float:
    generator = torch.Generator().manual_seed(seed + round_index * 1009)
    order = torch.randperm(len(anchors), generator=generator)
    cursor, objectives = 0, []
    server.train()
    for step_index in range(fixed["server_steps_per_round"]):
        batch_size = fixed["server_batch_size"]
        indices = order[cursor:cursor + batch_size]
        if len(indices) < batch_size:
            order = torch.randperm(
                len(anchors), generator=torch.Generator().manual_seed(
                    seed + round_index * 1009 + step_index + 1))
            cursor = 0
            indices = order[:batch_size]
        cursor += batch_size
        rows = [anchors[int(index)] for index in indices]
        hard = torch.tensor(
            [int(row[task_spec.label_field]) for row in rows], device=device)
        optimizer.zero_grad(set_to_none=True)
        losses = core.candidate_loss_matrix(
            server, tokenizer, rows, task_spec, device, fixed["max_length"],
            batch_size, requires_grad=True)
        anchored = F.cross_entropy(-losses, hard)
        objective = fixed["anchor_supervision_weight"] * anchored
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
        "kafa_script/trec_server_post_primary_ablation_v1.json"))
    parser.add_argument("--task-spec", type=Path, default=Path(
        "kafa_script/task_specs/trec_coarse_server_v1.json"))
    parser.add_argument("--off-task-spec", type=Path, default=Path(
        "kafa_script/task_specs/imdb_sentiment_server_v1.json"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "kafa_script/trec_server_manifest_v1.json"))
    parser.add_argument("--target-train", type=Path, default=Path(
        ".cache/datasets/trec/train_5500.label"))
    parser.add_argument("--official-test", type=Path, default=Path(
        ".cache/datasets/trec/TREC_10.label"))
    parser.add_argument("--off-task-train", type=Path, default=Path(
        ".cache/datasets/imdb/train.jsonl"))
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--evaluation-split", required=True,
                        choices=["development", "official_test"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is required for the frozen benchmark")
    protocol = load_protocol(args.protocol)
    task_spec = load_task_spec(args.task_spec)
    off_task_spec = load_task_spec(args.off_task_spec)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    fixed = validate_contract(args, protocol, task_spec, off_task_spec, manifest)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    started_utc = datetime.now(timezone.utc).isoformat()

    target_rows = core.load_trec_train(args.target_train)
    partitions = core.resolve_manifest_splits(
        target_rows, manifest,
        ["anchor", "development", "client_0", "client_1", "client_2"])
    anchors = partitions["anchor"]
    evaluation_rows = core.load_evaluation_rows(
        args.evaluation_split, partitions["development"], args.official_test,
        protocol["task"]["target_official_test_sha256"],
        fixed["official_test_examples"])
    client_bases = [partitions[f"client_{index}"] for index in range(3)]
    off_task_rows = core.load_jsonl(args.off_task_train)
    method_configuration = METHODS[args.method]
    mixtures = core.build_client_mixtures(
        client_bases, off_task_rows, task_spec, off_task_spec, args.seed,
        method_configuration["conflict_ratio"],
        method_configuration["off_task_ratio"])

    server_path = ROOT / fixed["server_model"]
    client_paths = [ROOT / path for path in fixed["client_models"]]
    server, server_tokenizer = core.make_lora(server_path, device)
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
        client_iterator = (enumerate(zip(client_paths, mixtures))
                           if execute_clients else [])
        for client_index, (path, mixture) in client_iterator:
            model, tokenizer = core.make_lora(path, device)
            if client_states[client_index] is not None:
                core.set_peft_model_state_dict(model, client_states[client_index])
            if method_configuration["selection"] == "relative_balanced_routing":
                selected, selection_record = core.route_client_mixture(
                    model, tokenizer, mixture, anchors, task_spec,
                    off_task_spec, device, fixed["max_length"],
                    fixed["routing_keep_fraction"],
                    fixed["routing_gate_threshold"])
                selection_record = {
                    "policy": "relative_balanced_routing",
                    "routing_executed": True,
                    **selection_record,
                }
            else:
                selected = mixture
                selection_record = {
                    "policy": "all_examples",
                    "selected": len(selected),
                    "routing_executed": False,
                }
            mean_loss = core.train_client(
                model, tokenizer, selected, task_spec, off_task_spec, device,
                fixed["client_steps_per_round"], fixed["client_batch_size"],
                fixed["max_length"], fixed["client_learning_rate"],
                args.seed * 100000 + round_index * 100 + client_index)
            model.eval()
            losses = core.candidate_loss_matrix(
                model, tokenizer, anchors, task_spec, device,
                fixed["max_length"], fixed["anchor_score_batch_size"])
            probabilities = torch.softmax(-losses.float(), dim=1).cpu()
            probability_payload = audit_probability_payload(
                probabilities, expected_rows=len(anchors),
                num_classes=task_spec.num_classes)
            wire_payload = core.serialized_probability_payload(probabilities)
            client_states[client_index] = core.clone_adapter_state(model)
            probability_matrices.append(probabilities)
            kind_counts = dict(sorted(Counter(
                item["kind"] for item in mixture).items()))
            selected_kind_counts = dict(sorted(Counter(
                item["kind"] for item in selected).items()))
            client_records.append({
                "client_index": client_index,
                "model": str(path.relative_to(ROOT)),
                "model_artifacts": core.artifact_hashes(path),
                "base_target_ids_sha256": core.adapter_record_ids(
                    client_bases[client_index]),
                "mixture_assignment_sha256": mixture_assignment_sha256(mixture),
                "selected_ids_sha256": core.adapter_record_ids(selected),
                "mixture_kind_counts": kind_counts,
                "selected_kind_counts_audit_only": selected_kind_counts,
                "mean_local_loss": mean_loss,
                "selection": selection_record,
                "probability_payload": probability_payload,
                "serialized_probability_payload": wire_payload,
                "adapter_bytes": core.state_bytes(client_states[client_index]),
            })
            del model
            torch.cuda.empty_cache()
        teacher = (sum(probability_matrices) / len(probability_matrices)
                   if probability_matrices else None)
        total_payload += sum(
            record["serialized_probability_payload"]["payload_bytes"]
            for record in client_records)
        mean_objective = train_server_round(
            server, server_tokenizer, server_optimizer, anchors, task_spec,
            teacher, fixed, args.seed, round_index, device)
        round_records.append({
            "round": round_index + 1,
            "client_records": client_records,
            "mean_server_objective": mean_objective,
        })

    final = core.evaluate(
        server, server_tokenizer, evaluation_rows, task_spec, device,
        fixed["max_length"], fixed["evaluation_batch_size"])
    elapsed = time.perf_counter() - started
    full_args = {
        "protocol": str(args.protocol),
        "task_spec": str(args.task_spec),
        "off_task_spec": str(args.off_task_spec),
        "manifest": str(args.manifest),
        "target_train": str(args.target_train),
        "official_test": str(args.official_test),
        "off_task_train": str(args.off_task_train),
        "method": args.method,
        "evaluation_split": args.evaluation_split,
        "seed": args.seed,
        "output": str(args.output),
    }
    result = {
        "schema_version": "trec_server_ablation_v1",
        "evidence_label": ("post_primary_development" if
                           args.evaluation_split == "development" else
                           "post_primary_official_test_ablation"),
        "paper_result": args.evaluation_split == "official_test",
        "task_id": task_spec.task_id,
        "method": args.method,
        "method_configuration": method_configuration,
        "seed": args.seed,
        "started_utc": started_utc,
        "protocol_sha256": core.sha256_file(args.protocol),
        "parent_primary_protocol_sha256": protocol[
            "parent_primary_protocol"]["sha256"],
        "runner_file_sha256": core.sha256_file(Path(__file__)),
        "frozen_primary_runner_file_sha256": core.sha256_file(
            ROOT / protocol["implementation"]["frozen_primary_runner"]),
        "classification_task_file_sha256": core.sha256_file(
            ROOT / protocol["implementation"]["classification_task_core"]),
        "task_spec_semantic_sha256": task_spec.sha256,
        "task_spec_file_sha256": core.sha256_file(args.task_spec),
        "off_task_spec_semantic_sha256": off_task_spec.sha256,
        "manifest_sha256": core.sha256_file(args.manifest),
        "full_args": full_args,
        "full_args_sha256": canonical_sha256(full_args),
        "fixed_configuration": fixed,
        "git": core.git_state(),
        "datasets": {
            "target_train_sha256": core.sha256_file(args.target_train),
            "target_official_test_sha256": protocol["task"][
                "target_official_test_sha256"],
            "target_official_test_hash_source": (
                "verified_raw_bytes" if args.evaluation_split == "official_test"
                else "frozen_protocol_not_reopened_in_development"),
            "off_task_train_sha256": core.sha256_file(args.off_task_train),
        },
        "model_artifacts": {
            "server": core.artifact_hashes(server_path),
            "clients": [core.artifact_hashes(path) for path in client_paths],
        },
        "split_source_ids_sha256": {
            "anchor": core.adapter_record_ids(anchors),
            "evaluation": core.adapter_record_ids(evaluation_rows),
            **{f"client_{index}": core.adapter_record_ids(base)
               for index, base in enumerate(client_bases)},
        },
        "mixture_assignment_sha256": [
            mixture_assignment_sha256(mixture) for mixture in mixtures],
        "mixture_source_ids_sha256": [
            core.adapter_record_ids(mixture) for mixture in mixtures],
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
    if not all(math.isfinite(value) for key, value in final.items()
               if key in {"accuracy", "macro_f1", "canonical_label_nll", "ece"}):
        raise ValueError("non-finite final metric")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
