"""Fail-closed verifier for the prospective IMDb role-reversal matrix."""

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

from kafa_script.classification_task import (  # noqa: E402
    canonical_sha256, load_task_spec, mixture_assignment_sha256)
from kafa_script import run_generic_classification_distillation as core  # noqa: E402
from kafa_script.run_imdb_server_ablation import (  # noqa: E402
    METHODS, load_jsonl_preserve_rows, load_jsonl_unique_with_lines)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_matrix(phase: str, output_dir: Path, protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(protocol["name"] == "imdb_server_role_reversal_protocol_v1",
            "unexpected protocol")
    for relative, expected in protocol["implementation"][
            "file_sha256"].items():
        require(core.sha256_file(ROOT / relative) == expected,
                f"implementation hash mismatch: {relative}")
    runner_path = ROOT / "kafa_script/run_imdb_server_ablation.py"
    runner_sha = core.sha256_file(runner_path)
    protocol_sha = core.sha256_file(protocol_path)
    section = protocol[phase]
    seeds, methods = section["seeds"], section["comparators"]
    require(methods == list(METHODS), "method order mismatch")
    expected_names = {f"seed{seed}_{method}.json"
                      for seed in seeds for method in methods}
    observed_names = {path.name for path in output_dir.glob("seed*.json")}
    require(expected_names == observed_names,
            f"matrix mismatch missing={sorted(expected_names-observed_names)} "
            f"extra={sorted(observed_names-expected_names)}")
    require(not list(output_dir.glob("*.tmp*")), "temporary record remains")

    manifest_path = ROOT / protocol["data"]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(core.sha256_file(manifest_path) == protocol["data"][
        "manifest_sha256"], "manifest hash mismatch")
    target_spec = load_task_spec(ROOT / protocol["task"]["target_spec"])
    off_task_spec = load_task_spec(ROOT / protocol["task"]["off_task_spec"])
    require(target_spec.sha256 == protocol["task"][
        "target_spec_semantic_sha256"], "target semantic mismatch")
    require(off_task_spec.sha256 == protocol["task"][
        "off_task_spec_semantic_sha256"], "off-task semantic mismatch")
    target_path = ROOT / protocol["data"]["target_train_path"]
    off_task_path = ROOT / protocol["data"]["off_task_train_path"]
    require(core.sha256_file(target_path) == protocol["data"][
        "target_train_sha256"], "target train hash mismatch")
    require(core.sha256_file(off_task_path) == protocol["data"][
        "off_task_train_sha256"], "off-task hash mismatch")
    # The manifest keys canonical-deduplicated training examples by their
    # original one-indexed source lines.  Reuse the runner's audited loader;
    # the generic JSONL loader intentionally does not retain that metadata.
    target_rows = load_jsonl_unique_with_lines(target_path)
    partitions = core.resolve_manifest_splits(
        target_rows, manifest,
        ["anchor", "development", "client_0", "client_1", "client_2"])
    anchors = partitions["anchor"]
    bases = [partitions[f"client_{index}"] for index in range(3)]
    off_task_rows = core.load_jsonl(off_task_path)
    if phase == "development":
        evaluation_rows = partitions["development"]
        evaluation_split = "development"
        official_path = protocol["development"]["nonexistent_test_sentinel"]
        require(not (ROOT / official_path).exists(),
                "development sentinel unexpectedly exists")
        expected_hash_source = "frozen_manifest_not_opened"
    else:
        official_path = protocol["data"]["official_test_path"]
        official_full_path = ROOT / official_path
        require(core.sha256_file(official_full_path) == protocol["data"][
            "official_test_sha256"], "official test hash mismatch")
        evaluation_rows = load_jsonl_preserve_rows(official_full_path)
        require(len(evaluation_rows) == protocol["data"][
            "official_test_examples"], "official test count mismatch")
        evaluation_split = "official_test"
        expected_hash_source = "verified_raw_bytes"
    fixed = protocol["fixed_configuration"]
    records = {}
    for seed in seeds:
        mixtures_by_method = {}
        for method in methods:
            path = output_dir / f"seed{seed}_{method}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            records[(seed, method)] = record
            config = METHODS[method]
            mixtures = core.build_client_mixtures(
                bases, off_task_rows, target_spec, off_task_spec, seed,
                config["conflict_ratio"], config["off_task_ratio"])
            mixtures_by_method[method] = mixtures
            require(record["schema_version"] == "imdb_server_role_reversal_v1",
                    f"{path.name}: schema")
            require(record["seed"] == seed and record["method"] == method,
                    f"{path.name}: identity")
            require(record["method_configuration"] == config,
                    f"{path.name}: method configuration")
            require(record["protocol_sha256"] == protocol_sha and
                    record["runner_file_sha256"] == runner_sha,
                    f"{path.name}: frozen hashes")
            require(record["manifest_sha256"] == core.sha256_file(manifest_path),
                    f"{path.name}: manifest hash")
            require(record["fixed_configuration"] == fixed,
                    f"{path.name}: fixed configuration")
            require(record["evaluation_split"] == evaluation_split and
                    record["evaluation_examples"] == len(evaluation_rows),
                    f"{path.name}: evaluation scope")
            require(record["datasets"]["target_official_test_hash_source"] ==
                    expected_hash_source, f"{path.name}: test hash source")
            require(record["split_source_ids_sha256"]["anchor"] ==
                    core.adapter_record_ids(anchors) and
                    record["split_source_ids_sha256"]["evaluation"] ==
                    core.adapter_record_ids(evaluation_rows),
                    f"{path.name}: split hashes")
            for index, base in enumerate(bases):
                require(record["split_source_ids_sha256"][f"client_{index}"] ==
                        core.adapter_record_ids(base),
                        f"{path.name}: client split {index}")
            expected_mixture_hashes = [
                mixture_assignment_sha256(mixture) for mixture in mixtures]
            require(record["mixture_assignment_sha256"] ==
                    expected_mixture_hashes, f"{path.name}: mixture hashes")
            expected_output = (str(path.relative_to(ROOT)) if path.is_absolute()
                               else str(path))
            expected_args = {
                "protocol": "kafa_script/imdb_server_role_reversal_protocol_v1.json",
                "task_spec": "kafa_script/task_specs/imdb_sentiment_server_v1.json",
                "off_task_spec": "kafa_script/task_specs/agnews_offtask_v1.json",
                "manifest": "kafa_script/imdb_server_manifest_v1.json",
                "target_train": ".cache/datasets/imdb/train.jsonl",
                "official_test": official_path,
                "off_task_train": ".cache/datasets/agnews/train.jsonl",
                "method": method,
                "evaluation_split": evaluation_split,
                "seed": seed,
                "output": expected_output,
            }
            require(record["full_args"] == expected_args,
                    f"{path.name}: full args")
            require(record["full_args_sha256"] == canonical_sha256(
                expected_args), f"{path.name}: full args hash")
            metrics = record["final"]
            for metric in ("accuracy", "macro_f1", "canonical_label_nll",
                           "target_answer_loss", "ece"):
                require(math.isfinite(float(metrics[metric])),
                        f"{path.name}: non-finite {metric}")
            require(len(metrics["per_class_recall"]) == 2,
                    f"{path.name}: recalls")
            require(len(record["round_records"]) == fixed["rounds"],
                    f"{path.name}: rounds")
            if config["selection"] == "no_client_path":
                require(record["total_client_to_server_payload_bytes"] == 0,
                        f"{path.name}: anchor payload")
                for round_record in record["round_records"]:
                    require(round_record["client_records"] == [] and
                            round_record["teacher_weights"] == [],
                            f"{path.name}: anchor path")
            else:
                payload_total = 0
                for round_record in record["round_records"]:
                    clients = round_record["client_records"]
                    require(len(clients) == 3 and
                            round_record["teacher_weights"] == [1/3] * 3,
                            f"{path.name}: teacher ensemble")
                    for index, client in enumerate(clients):
                        mixture = mixtures[index]
                        expected_counts = dict(sorted(Counter(
                            item["kind"] for item in mixture).items()))
                        require(client["base_target_ids_sha256"] ==
                                core.adapter_record_ids(bases[index]) and
                                client["mixture_assignment_sha256"] ==
                                mixture_assignment_sha256(mixture),
                                f"{path.name}: client identity")
                        require(client["mixture_kind_counts"] == expected_counts,
                                f"{path.name}: mixture counts")
                        selection = client["selection"]
                        selected_counts = client[
                            "selected_kind_counts_audit_only"]
                        selected_n = sum(selected_counts.values())
                        require(selection["selected"] == selected_n,
                                f"{path.name}: selected count")
                        if config["selection"] == "all_examples":
                            require(selected_n == 256 and
                                    selected_counts == expected_counts,
                                    f"{path.name}: unfiltered selection")
                        elif selection["gate_triggered"]:
                            require(selected_n == 128 and
                                    selected_counts.get("off_task", 0) == 0,
                                    f"{path.name}: routed selection")
                        else:
                            require(selected_n == 256,
                                    f"{path.name}: untriggered selection")
                        probability = client["probability_payload"]
                        require(probability["shape"] == [128, 2] and
                                probability["max_row_sum_error"] <= 1e-6,
                                f"{path.name}: payload interface")
                        payload = client["serialized_probability_payload"]
                        payload_total += payload["payload_bytes"]
                require(record["total_client_to_server_payload_bytes"] ==
                        payload_total, f"{path.name}: payload total")
        require([mixture_assignment_sha256(item) for item in mixtures_by_method[
            "severe_unfiltered_distillation"]] ==
            [mixture_assignment_sha256(item) for item in mixtures_by_method[
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
                        choices=["development", "official_evaluation"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(
        "kafa_script/imdb_server_role_reversal_protocol_v1.json"))
    args = parser.parse_args()
    print(json.dumps(verify_matrix(args.phase, args.output_dir, args.protocol),
                     indent=2))


if __name__ == "__main__":
    main()
