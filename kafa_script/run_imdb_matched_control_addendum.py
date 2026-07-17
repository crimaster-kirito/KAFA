"""Prospective matched-K controls for the IMDb role-reversal experiment.

This addendum deliberately shares the frozen severe mixture and optimization
budget across all arms.  It never uses audit-only ``kind`` metadata except in
the explicitly labelled oracle arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.classification_task import (  # noqa: E402
    audit_probability_payload, canonical_sha256, load_task_spec,
    mixture_assignment_sha256)
from kafa_script import run_generic_classification_distillation as core  # noqa: E402
from kafa_script import run_imdb_server_ablation as base  # noqa: E402


METHODS = {
    "severe_random_k": {"selection": "claimed_label_balanced_random_k"},
    "severe_routed_k": {"selection": "forced_relative_balanced_routing_k"},
    "oracle_max_purity_k": {
        "selection": "audit_only_maximum_clean_then_conflict_balanced_k"},
}
SEVERE_CONFLICT_RATIO = 0.4
SEVERE_OFF_TASK_RATIO = 0.2
FORCED_GATE_THRESHOLD = 1e30


def stable_selection_rank(namespace: str, seed: int, client_index: int,
                          source_id: str) -> str:
    payload = (f"kafa-imdb-matched-control-v1|{namespace}|{seed}|"
               f"{client_index}|{source_id}").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _balanced_select(candidates: list[dict], label_field: str, k: int,
                     seed: int, client_index: int,
                     namespace: str) -> list[dict]:
    if k <= 0 or k % 2:
        raise ValueError("matched K must be a positive multiple of two")
    quota = k // 2
    selected_ids: set[str] = set()
    for label in (0, 1):
        pool = [item for item in candidates
                if int(item[label_field]) == label]
        if len(pool) < quota:
            raise ValueError(
                f"{namespace} fails closed: class {label} has {len(pool)} "
                f"eligible examples, fewer than quota {quota}")
        ranked = sorted(pool, key=lambda item: (
            stable_selection_rank(namespace, seed, client_index,
                                  item["source_id"]), item["source_id"]))
        selected_ids.update(item["source_id"] for item in ranked[:quota])
    selected = [item for item in candidates
                if item["source_id"] in selected_ids]
    if len(selected) != k or len(selected_ids) != k:
        raise AssertionError("balanced matched selection is not exactly K")
    return selected


def select_random_k(mixture: list[dict], k: int, seed: int,
                    client_index: int) -> tuple[list[dict], dict]:
    """Random matched control over the same target/claimed-label pool."""
    eligible = [item for item in mixture if item["kind"] != "off_task"]
    selected = _balanced_select(
        eligible, "training_label", k, seed, client_index, "random-k")
    return selected, {
        "policy": "claimed_label_balanced_random_k",
        "selected": len(selected),
        "eligible": len(eligible),
        "quota_per_claimed_label": k // 2,
        "off_task_selected": 0,
        "ranking_namespace": "kafa-imdb-matched-control-v1|random-k",
        "known_task_membership_used": True,
        "hidden_clean_conflict_metadata_used": False,
    }


def select_oracle_max_purity_k(mixture: list[dict], k: int, seed: int,
                               client_index: int) -> tuple[list[dict], dict]:
    """Maximize clean count, then fill claimed-label quotas with conflicts.

    This is intentionally *not* called oracle-clean: a severe IMDb mixture
    contains only 103 clean rows, so K=128 requires 25 conflict rows.
    """
    if k <= 0 or k % 2:
        raise ValueError("matched K must be a positive multiple of two")
    quota = k // 2
    clean = [item for item in mixture if item["kind"] == "clean"]
    conflict = [item for item in mixture if item["kind"] == "conflict"]
    if len(clean) > k:
        selected = _balanced_select(
            clean, "training_label", k, seed, client_index,
            "oracle-max-purity-clean-trim")
    else:
        selected = list(clean)
        for label in (0, 1):
            have = sum(int(item["training_label"]) == label
                       for item in selected)
            need = quota - have
            if need < 0:
                raise ValueError("clean rows exceed claimed-label quota")
            pool = [item for item in conflict
                    if int(item["training_label"]) == label]
            if len(pool) < need:
                raise ValueError(
                    f"oracle maximum-purity fails closed: claimed class "
                    f"{label} needs {need} conflicts but has {len(pool)}")
            ranked = sorted(pool, key=lambda item: (
                stable_selection_rank("oracle-max-purity-conflict-fill", seed,
                                      client_index, item["source_id"]),
                item["source_id"]))
            selected.extend(ranked[:need])
    if len(selected) != k or len({x["source_id"] for x in selected}) != k:
        raise AssertionError("maximum-purity selection is not exactly K")
    selected_ids = {item["source_id"] for item in selected}
    selected = [item for item in mixture if item["source_id"] in selected_ids]
    return selected, {
        "policy": "audit_only_maximum_clean_then_conflict_balanced_k",
        "selected": len(selected),
        "eligible_clean": len(clean),
        "selected_clean": sum(x["kind"] == "clean" for x in selected),
        "selected_conflict": sum(x["kind"] == "conflict" for x in selected),
        "quota_per_claimed_label": quota,
        "off_task_selected": 0,
        "ranking_namespace": (
            "kafa-imdb-matched-control-v1|oracle-max-purity-conflict-fill"),
        "known_task_membership_used": True,
        "hidden_clean_conflict_metadata_used": True,
    }


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("name") != "imdb_matched_control_addendum_v1":
        raise ValueError("unexpected IMDb matched-control protocol")
    return protocol


def validate_contract(args, protocol: dict, target_spec, off_task_spec,
                      manifest: dict) -> dict:
    for relative, expected in protocol["implementation"]["file_sha256"].items():
        if core.sha256_file(ROOT / relative) != expected:
            raise ValueError(f"contract hash mismatch for {relative}")
    for path, expected in ((args.manifest, protocol["data"]["manifest_sha256"]),
                           (args.target_train, protocol["data"]["target_train_sha256"]),
                           (args.off_task_train, protocol["data"]["off_task_train_sha256"])):
        if core.sha256_file(path) != expected:
            raise ValueError(f"data hash mismatch for {path}")
    if target_spec.sha256 != protocol["task"]["target_spec_semantic_sha256"]:
        raise ValueError("target task semantic hash mismatch")
    if off_task_spec.sha256 != protocol["task"]["off_task_spec_semantic_sha256"]:
        raise ValueError("off-task semantic hash mismatch")
    if manifest["datasets"]["target_official_test"]["sha256"] != \
            protocol["data"]["official_test_sha256"]:
        raise ValueError("manifest official-test hash link mismatch")
    phase = ("development" if args.evaluation_split == "development"
             else "official_evaluation")
    if args.seed not in protocol[phase]["seeds"] or \
            args.method not in protocol[phase]["comparators"]:
        raise ValueError("seed/method is not frozen for this phase")
    if METHODS[args.method] != protocol["method_configuration"][args.method]:
        raise ValueError("method configuration differs from protocol")
    fixed = protocol["fixed_configuration"]
    if fixed["matched_k"] != 128 or \
            fixed["matched_quota_per_claimed_label"] != 64 or \
            fixed["forced_gate_threshold"] != FORCED_GATE_THRESHOLD:
        raise ValueError("matched-selection constants differ from implementation")
    if args.evaluation_split == "development":
        if args.official_test.exists():
            raise ValueError("development requires a nonexistent test sentinel")
        if str(args.official_test) != protocol["development"]["nonexistent_test_sentinel"]:
            raise ValueError("development sentinel path differs from protocol")
    elif str(args.official_test) != protocol["data"]["official_test_path"]:
        raise ValueError("official-test path differs from protocol")
    for relative, expected in protocol["model_weight_sha256"].items():
        if core.sha256_file(ROOT / relative) != expected:
            raise ValueError(f"model weight mismatch for {relative}")
    return fixed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path(
        "kafa_script/imdb_matched_control_addendum_protocol_v1.json"))
    parser.add_argument("--task-spec", type=Path, default=Path(
        "kafa_script/task_specs/imdb_sentiment_server_v1.json"))
    parser.add_argument("--off-task-spec", type=Path, default=Path(
        "kafa_script/task_specs/agnews_offtask_v1.json"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "kafa_script/imdb_server_manifest_v1.json"))
    parser.add_argument("--target-train", type=Path, default=Path(
        ".cache/datasets/imdb/train.jsonl"))
    parser.add_argument("--official-test", type=Path, required=True)
    parser.add_argument("--off-task-train", type=Path, default=Path(
        ".cache/datasets/agnews/train.jsonl"))
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--evaluation-split",
                        choices=["development", "official_test"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required for the frozen benchmark")

    protocol = load_protocol(args.protocol)
    target_spec, off_task_spec = load_task_spec(args.task_spec), load_task_spec(args.off_task_spec)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    fixed = validate_contract(args, protocol, target_spec, off_task_spec, manifest)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    started_utc = datetime.now(timezone.utc).isoformat()

    target_rows = base.load_jsonl_unique_with_lines(args.target_train)
    partitions = core.resolve_manifest_splits(
        target_rows, manifest,
        ["anchor", "development", "client_0", "client_1", "client_2"])
    anchors = partitions["anchor"]
    if args.evaluation_split == "development":
        evaluation_rows = partitions["development"]
        test_hash_source = "frozen_manifest_not_opened"
    else:
        if core.sha256_file(args.official_test) != protocol["data"]["official_test_sha256"]:
            raise ValueError("official-test hash mismatch")
        evaluation_rows = base.load_jsonl_preserve_rows(args.official_test)
        if len(evaluation_rows) != fixed["official_test_examples"]:
            raise ValueError("official-test count mismatch")
        test_hash_source = "verified_raw_bytes"
    client_bases = [partitions[f"client_{i}"] for i in range(3)]
    off_task_rows = core.load_jsonl(args.off_task_train)
    mixtures = core.build_client_mixtures(
        client_bases, off_task_rows, target_spec, off_task_spec, args.seed,
        SEVERE_CONFLICT_RATIO, SEVERE_OFF_TASK_RATIO)

    server_path = ROOT / fixed["server_model"]
    client_paths = [ROOT / path for path in fixed["client_models"]]
    server, server_tokenizer = core.make_lora(server_path, device)
    server_optimizer = torch.optim.AdamW(
        [p for p in server.parameters() if p.requires_grad],
        lr=fixed["server_learning_rate"])
    client_states = [None] * 3
    round_records, total_payload = [], 0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for round_index in range(fixed["rounds"]):
        probability_matrices, client_records = [], []
        for client_index, (path, mixture) in enumerate(zip(client_paths, mixtures)):
            model, tokenizer = core.make_lora(path, device)
            if client_states[client_index] is not None:
                core.set_peft_model_state_dict(model, client_states[client_index])
            if args.method == "severe_random_k":
                selected, selection = select_random_k(
                    mixture, fixed["matched_k"], args.seed, client_index)
            elif args.method == "oracle_max_purity_k":
                selected, selection = select_oracle_max_purity_k(
                    mixture, fixed["matched_k"], args.seed, client_index)
            else:
                selected, routed = core.route_client_mixture(
                    model, tokenizer, mixture, anchors, target_spec,
                    off_task_spec, device, fixed["max_length"],
                    fixed["matched_k"] / fixed["client_examples_each"],
                    fixed["forced_gate_threshold"])
                selection = {
                    "policy": "forced_relative_balanced_routing_k",
                    **routed,
                    "forced_selection": True,
                    "known_task_membership_used": True,
                    "hidden_clean_conflict_metadata_used": False,
                }
            if len(selected) != fixed["matched_k"]:
                raise ValueError("matched arm did not select exactly K")
            mean_loss = core.train_client(
                model, tokenizer, selected, target_spec, off_task_spec, device,
                fixed["client_steps_per_round"], fixed["client_batch_size"],
                fixed["max_length"], fixed["client_learning_rate"],
                args.seed * 100000 + round_index * 100 + client_index)
            model.eval()
            losses = core.candidate_loss_matrix(
                model, tokenizer, anchors, target_spec, device,
                fixed["max_length"], fixed["anchor_score_batch_size"])
            probabilities = torch.softmax(-losses.float(), dim=1).cpu()
            probability_payload = audit_probability_payload(
                probabilities, expected_rows=len(anchors),
                num_classes=target_spec.num_classes)
            wire_payload = core.serialized_probability_payload(probabilities)
            client_states[client_index] = core.clone_adapter_state(model)
            probability_matrices.append(probabilities)
            client_records.append({
                "client_index": client_index,
                "model": str(path.relative_to(ROOT)),
                "model_artifacts": core.artifact_hashes(path),
                "base_target_ids_sha256": core.adapter_record_ids(client_bases[client_index]),
                "mixture_assignment_sha256": mixture_assignment_sha256(mixture),
                "selected_ids_sha256": core.adapter_record_ids(selected),
                "mixture_kind_counts": dict(sorted(Counter(x["kind"] for x in mixture).items())),
                "selected_kind_counts_audit_only": dict(sorted(Counter(x["kind"] for x in selected).items())),
                "selected_claimed_label_counts": {
                    str(label): sum(int(x["training_label"]) == label
                                    for x in selected)
                    for label in range(target_spec.num_classes)
                },
                "mean_local_loss": mean_loss,
                "selection": selection,
                "probability_payload": probability_payload,
                "serialized_probability_payload": wire_payload,
                "adapter_bytes": core.state_bytes(client_states[client_index]),
            })
            del model
            torch.cuda.empty_cache()
        teacher = sum(probability_matrices) / len(probability_matrices)
        total_payload += sum(x["serialized_probability_payload"]["payload_bytes"] for x in client_records)
        objective = base.train_server_round(
            server, server_tokenizer, server_optimizer, anchors, target_spec,
            teacher, fixed, args.seed, round_index, device)
        round_records.append({
            "round": round_index + 1,
            "client_records": client_records,
            "teacher_weights": [1 / 3] * 3,
            "mean_server_objective": objective,
        })

    final = core.evaluate(server, server_tokenizer, evaluation_rows,
                          target_spec, device, fixed["max_length"],
                          fixed["evaluation_batch_size"])
    full_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    result = {
        "schema_version": "imdb_matched_control_addendum_v1",
        "evidence_label": ("matched_control_development" if args.evaluation_split == "development"
                           else "prospective_untouched_test_matched_control"),
        "paper_result": args.evaluation_split == "official_test",
        "task_id": target_spec.task_id,
        "method": args.method,
        "method_configuration": METHODS[args.method],
        "seed": args.seed,
        "started_utc": started_utc,
        "protocol_sha256": core.sha256_file(args.protocol),
        "runner_file_sha256": core.sha256_file(Path(__file__)),
        "manifest_sha256": core.sha256_file(args.manifest),
        "full_args": full_args,
        "full_args_sha256": canonical_sha256(full_args),
        "fixed_configuration": fixed,
        "git": core.git_state(),
        "datasets": {
            "target_train_sha256": core.sha256_file(args.target_train),
            "target_official_test_sha256": protocol["data"]["official_test_sha256"],
            "target_official_test_hash_source": test_hash_source,
            "off_task_train_sha256": core.sha256_file(args.off_task_train),
        },
        "split_source_ids_sha256": {
            "anchor": core.adapter_record_ids(anchors),
            "evaluation": core.adapter_record_ids(evaluation_rows),
            **{f"client_{i}": core.adapter_record_ids(x) for i, x in enumerate(client_bases)},
        },
        "mixture_assignment_sha256": [mixture_assignment_sha256(x) for x in mixtures],
        "evaluation_split": args.evaluation_split,
        "evaluation_examples": len(evaluation_rows),
        "anchor_examples": len(anchors),
        "round_records": round_records,
        "final": final,
        "total_client_to_server_payload_bytes": total_payload,
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "evidence_boundaries": protocol["evidence_boundaries"],
    }
    if not all(math.isfinite(float(final[k])) for k in
               ("accuracy", "macro_f1", "canonical_label_nll", "ece")):
        raise ValueError("non-finite final metric")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
