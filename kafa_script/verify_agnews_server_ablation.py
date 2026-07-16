"""Fail-closed verification for the AG News post-primary four-arm matrix."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script import run_generic_classification_distillation as audit  # noqa: E402
from kafa_script.run_agnews_local_dev import select_manifest_rows  # noqa: E402
from kafa_script.run_agnews_mixture_dev import build_mixture  # noqa: E402
from kafa_script.run_real_data_plumbing_smoke import load_jsonl  # noqa: E402
from kafa_script.run_heterogeneous_distillation_dev import (  # noqa: E402
    select_disjoint_client_bases)
from kafa_script.run_agnews_server_ablation import (  # noqa: E402
    METHODS, canonical_sha256, ids_sha256, mixture_sha256)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_matrix(phase: str, output_dir: Path, protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(protocol["name"] == "agnews_server_post_primary_ablation_v1",
            "unexpected protocol")
    for relative, expected_sha in protocol["implementation"][
            "file_sha256"].items():
        require(audit.sha256_file(ROOT / relative) == expected_sha,
                f"implementation hash mismatch: {relative}")
    runner_path = ROOT / "kafa_script/run_agnews_server_ablation.py"
    runner_sha = audit.sha256_file(runner_path)
    require(runner_sha == protocol["implementation"]["file_sha256"][
        "kafa_script/run_agnews_server_ablation.py"], "stale runner hash")
    protocol_sha = audit.sha256_file(protocol_path)
    phase_config = protocol[phase]
    seeds = phase_config["seeds"]
    methods = phase_config["comparators"]
    require(methods == list(METHODS), "method order differs from contract")
    expected = {f"seed{seed}_{method}.json" for seed in seeds for method in methods}
    observed = {path.name for path in output_dir.glob("seed*.json")}
    require(observed == expected,
            f"matrix mismatch missing={sorted(expected-observed)} extra={sorted(observed-expected)}")
    require(not list(output_dir.glob("*.tmp*")), "temporary record remains")

    parent_path = ROOT / protocol["data"]["parent_manifest"]
    holdout_path = ROOT / protocol["data"]["holdout_manifest"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    require(audit.sha256_file(parent_path) == protocol["data"][
        "parent_manifest_sha256"], "parent manifest hash mismatch")
    require(audit.sha256_file(holdout_path) == protocol["data"][
        "holdout_manifest_sha256"], "holdout manifest hash mismatch")
    train_path = ROOT / ".cache/datasets/agnews/train.jsonl"
    imdb_path = ROOT / ".cache/datasets/imdb/train.jsonl"
    require(audit.sha256_file(train_path) == protocol["data"][
        "agnews_train_sha256"], "AG News train hash mismatch")
    require(audit.sha256_file(imdb_path) == protocol["data"][
        "imdb_train_sha256"], "IMDb train hash mismatch")
    agnews, imdb = load_jsonl(train_path), load_jsonl(imdb_path)
    by_id = {row["_id"]: row for row in agnews}
    holdout_rows = [by_id[item["id"]] for item in holdout["holdout"]["records"]]
    require(ids_sha256(holdout_rows) == protocol["data"][
        "secondary_holdout_ids_sha256"], "holdout ID hash mismatch")
    fixed = protocol["fixed_configuration"]
    anchors = select_manifest_rows(agnews, parent, "anchor",
                                   fixed["anchor_per_class"])
    development = select_manifest_rows(agnews, parent, "development",
                                       fixed["development_per_class"])
    expected_evaluation = (development if phase == "development" else holdout_rows)
    expected_evaluation_name = ("development" if phase == "development" else
                                "secondary_holdout")
    allocation_manifest = json.loads(json.dumps(parent))
    allocation_manifest["splits"]["development"]["records"] = (
        parent["splits"]["development"]["records"] +
        holdout["holdout"]["records"])

    records = {}
    for seed in seeds:
        bases = select_disjoint_client_bases(
            agnews, allocation_manifest, seed, len(fixed["client_models"]),
            fixed["client_target_per_class"])
        require(not any(set(row["_id"] for row in base) &
                        set(row["_id"] for row in holdout_rows)
                        for base in bases), f"seed {seed}: holdout overlap")
        reconstructed = {}
        for method in methods:
            path = output_dir / f"seed{seed}_{method}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            records[(seed, method)] = record
            config = METHODS[method]
            mixtures = [build_mixture(
                base, imdb, seed + 1009 * index,
                config["conflict_ratio"], config["off_task_ratio"])
                for index, base in enumerate(bases)]
            reconstructed[method] = mixtures
            require(record["schema_version"] == "agnews_server_ablation_v1",
                    f"{path.name}: schema")
            require(record["seed"] == seed and record["method"] == method,
                    f"{path.name}: identity")
            require(record["method_configuration"] == config,
                    f"{path.name}: method configuration")
            require(record["protocol_sha256"] == protocol_sha and
                    record["runner_file_sha256"] == runner_sha,
                    f"{path.name}: frozen hashes")
            require(record["parent_manifest_sha256"] == audit.sha256_file(parent_path)
                    and record["holdout_manifest_sha256"] == audit.sha256_file(holdout_path),
                    f"{path.name}: manifest hashes")
            require(record["fixed_configuration"] == fixed,
                    f"{path.name}: fixed configuration")
            require(record["evaluation_split"] == expected_evaluation_name,
                    f"{path.name}: evaluation split")
            require(record["evaluation_examples"] == len(expected_evaluation),
                    f"{path.name}: evaluation count")
            require(record["anchor_examples"] == len(anchors),
                    f"{path.name}: anchor count")
            require(record["datasets"]["official_test_accessed"] is False,
                    f"{path.name}: official-test boundary")
            require(record["split_source_ids_sha256"]["anchor"] == ids_sha256(anchors)
                    and record["split_source_ids_sha256"]["evaluation"] ==
                    ids_sha256(expected_evaluation)
                    and record["split_source_ids_sha256"]["frozen_holdout"] ==
                    ids_sha256(holdout_rows), f"{path.name}: split hashes")
            for index, base in enumerate(bases):
                require(record["split_source_ids_sha256"][f"client_{index}"] ==
                        ids_sha256(base), f"{path.name}: client split {index}")
            expected_mixture_hashes = [mixture_sha256(item) for item in mixtures]
            require(record["mixture_assignment_sha256"] == expected_mixture_hashes,
                    f"{path.name}: mixture hashes")
            args = record["full_args"]
            expected_output = (str(path.relative_to(ROOT)) if path.is_absolute()
                               else str(path))
            expected_args = {
                "protocol": "kafa_script/agnews_server_post_primary_ablation_v1.json",
                "parent_manifest": "kafa_script/benchmark_manifest_v1.json",
                "holdout_manifest": "kafa_script/agnews_secondary_holdout_manifest_v1.json",
                "agnews_data": ".cache/datasets/agnews/train.jsonl",
                "imdb_data": ".cache/datasets/imdb/train.jsonl",
                "method": method,
                "evaluation_split": expected_evaluation_name,
                "seed": seed,
                "output": expected_output,
            }
            require(args == expected_args, f"{path.name}: full args")
            require(record["full_args_sha256"] == canonical_sha256(args),
                    f"{path.name}: full args hash")
            metrics = record["final"]
            for metric in ("accuracy", "macro_f1", "label_nll", "ece"):
                require(math.isfinite(float(metrics[metric])),
                        f"{path.name}: non-finite {metric}")
            require(0 <= metrics["accuracy"] <= 1 and
                    0 <= metrics["macro_f1"] <= 1 and
                    0 <= metrics["ece"] <= 1,
                    f"{path.name}: metric range")
            require(len(metrics["per_class_recall"]) == 4,
                    f"{path.name}: recalls")
            require(len(record["round_records"]) == fixed["rounds"],
                    f"{path.name}: rounds")
            if config["selection"] == "no_client_path":
                require(record["total_client_to_server_payload_bytes"] == 0,
                        f"{path.name}: anchor payload")
                for round_record in record["round_records"]:
                    require(round_record["client_records"] == [] and
                            round_record["teacher_weights"] == [],
                            f"{path.name}: anchor client path")
            else:
                payload_total = 0
                for round_record in record["round_records"]:
                    clients = round_record["client_records"]
                    require(len(clients) == 3 and
                            round_record["teacher_weights"] == [1/3] * 3,
                            f"{path.name}: teacher ensemble")
                    for index, client in enumerate(clients):
                        mixture = mixtures[index]
                        require(client["client_index"] == index and
                                client["base_target_ids_sha256"] == ids_sha256(bases[index])
                                and client["mixture_assignment_sha256"] ==
                                mixture_sha256(mixture),
                                f"{path.name}: client {index} identity")
                        require(client["probability_shape"] == [128, 4] and
                                client["probability_dtype"] == "torch.float32",
                                f"{path.name}: probability interface")
                        expected_counts = dict(sorted(Counter(
                            item["kind"] for item in mixture).items()))
                        require(client["mixture_kind_counts"] == expected_counts,
                                f"{path.name}: mixture counts")
                        selected_count = sum(
                            client["selected_kind_counts_audit_only"].values())
                        selection = client["selection"]
                        require(selected_count == selection["selected"],
                                f"{path.name}: selected counts")
                        if config["selection"] == "all_examples":
                            require(selected_count == 256 and
                                    client["selected_kind_counts_audit_only"] == expected_counts,
                                    f"{path.name}: unfiltered selection")
                        elif selection["gate_triggered"]:
                            require(selected_count == 128 and
                                    client["selected_kind_counts_audit_only"].get(
                                        "off_task", 0) == 0,
                                    f"{path.name}: routed selection")
                        else:
                            require(selected_count == 256,
                                    f"{path.name}: untriggered selection")
                        payload_total += client[
                            "serialized_probability_payload_bytes"]
                require(record["total_client_to_server_payload_bytes"] == payload_total,
                        f"{path.name}: payload total")
        require([mixture_sha256(item) for item in reconstructed[
            "severe_unfiltered_distillation"]] ==
            [mixture_sha256(item) for item in reconstructed[
                "severe_routed_distillation"]],
                f"seed {seed}: severe arms differ")
    return {
        "phase": phase,
        "records": len(records),
        "protocol_sha256": protocol_sha,
        "runner_sha256": runner_sha,
        "seeds": seeds,
        "methods": methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
                        choices=["development", "secondary_holdout"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=Path(
        "kafa_script/agnews_server_post_primary_ablation_v1.json"))
    args = parser.parse_args()
    print(json.dumps(verify_matrix(args.phase, args.output_dir, args.protocol),
                     indent=2))


if __name__ == "__main__":
    main()
