"""Fail-closed verifier for the IMDb matched-K control addendum."""

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
from kafa_script import run_imdb_server_ablation as base  # noqa: E402
from kafa_script import run_imdb_matched_control_addendum as runner  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_matrix(phase: str, output_dir: Path, protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(protocol["name"] == "imdb_matched_control_addendum_v1",
            "unexpected protocol")
    for relative, expected in protocol["implementation"]["file_sha256"].items():
        require(core.sha256_file(ROOT / relative) == expected,
                f"implementation hash mismatch: {relative}")
    section = protocol[phase]
    seeds, methods = section["seeds"], section["comparators"]
    require(methods == list(runner.METHODS), "method order mismatch")
    expected_names = {f"seed{s}_{m}.json" for s in seeds for m in methods}
    observed_names = {p.name for p in output_dir.glob("seed*.json")}
    require(expected_names == observed_names,
            f"matrix mismatch missing={sorted(expected_names-observed_names)} "
            f"extra={sorted(observed_names-expected_names)}")

    manifest_path = ROOT / protocol["data"]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(core.sha256_file(manifest_path) == protocol["data"]["manifest_sha256"],
            "manifest hash mismatch")
    target_spec = load_task_spec(ROOT / protocol["task"]["target_spec"])
    off_task_spec = load_task_spec(ROOT / protocol["task"]["off_task_spec"])
    target_path = ROOT / protocol["data"]["target_train_path"]
    off_task_path = ROOT / protocol["data"]["off_task_train_path"]
    require(core.sha256_file(target_path) == protocol["data"]["target_train_sha256"],
            "target train hash mismatch")
    require(core.sha256_file(off_task_path) == protocol["data"]["off_task_train_sha256"],
            "off-task train hash mismatch")
    target_rows = base.load_jsonl_unique_with_lines(target_path)
    partitions = core.resolve_manifest_splits(
        target_rows, manifest,
        ["anchor", "development", "client_0", "client_1", "client_2"])
    anchors = partitions["anchor"]
    bases = [partitions[f"client_{i}"] for i in range(3)]
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
        require(core.sha256_file(official_full_path) == protocol["data"]["official_test_sha256"],
                "official test hash mismatch")
        evaluation_rows = base.load_jsonl_preserve_rows(official_full_path)
        require(len(evaluation_rows) == protocol["data"]["official_test_examples"],
                "official test count mismatch")
        evaluation_split = "official_test"
        expected_hash_source = "verified_raw_bytes"

    fixed = protocol["fixed_configuration"]
    protocol_sha = core.sha256_file(protocol_path)
    runner_sha = core.sha256_file(ROOT / "kafa_script/run_imdb_matched_control_addendum.py")
    for seed in seeds:
        mixtures = core.build_client_mixtures(
            bases, off_task_rows, target_spec, off_task_spec, seed,
            runner.SEVERE_CONFLICT_RATIO, runner.SEVERE_OFF_TASK_RATIO)
        mixture_hashes = [mixture_assignment_sha256(x) for x in mixtures]
        for method in methods:
            path = output_dir / f"seed{seed}_{method}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            require(record["schema_version"] == "imdb_matched_control_addendum_v1",
                    f"{path.name}: schema")
            require(record["seed"] == seed and record["method"] == method,
                    f"{path.name}: identity")
            require(record["method_configuration"] == runner.METHODS[method],
                    f"{path.name}: method configuration")
            require(record["protocol_sha256"] == protocol_sha and
                    record["runner_file_sha256"] == runner_sha,
                    f"{path.name}: frozen hashes")
            require(record["fixed_configuration"] == fixed and
                    record["manifest_sha256"] == core.sha256_file(manifest_path),
                    f"{path.name}: frozen configuration")
            require(record["mixture_assignment_sha256"] == mixture_hashes,
                    f"{path.name}: severe mixture identity")
            require(record["evaluation_split"] == evaluation_split and
                    record["evaluation_examples"] == len(evaluation_rows),
                    f"{path.name}: evaluation scope")
            require(record["datasets"]["target_official_test_hash_source"] == expected_hash_source,
                    f"{path.name}: test boundary")
            expected_output = str(path.relative_to(ROOT)) if path.is_absolute() else str(path)
            expected_args = {
                "protocol": "kafa_script/imdb_matched_control_addendum_protocol_v1.json",
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
            require(record["full_args"] == expected_args and
                    record["full_args_sha256"] == canonical_sha256(expected_args),
                    f"{path.name}: full args")
            for metric in ("accuracy", "macro_f1", "canonical_label_nll",
                           "target_answer_loss", "ece"):
                require(math.isfinite(float(record["final"][metric])),
                        f"{path.name}: non-finite {metric}")
            require(len(record["round_records"]) == fixed["rounds"],
                    f"{path.name}: round count")
            payload_total = 0
            for rr in record["round_records"]:
                require(rr["teacher_weights"] == [1/3] * 3 and
                        len(rr["client_records"]) == 3,
                        f"{path.name}: teacher ensemble")
                for i, client in enumerate(rr["client_records"]):
                    mixture = mixtures[i]
                    require(client["mixture_assignment_sha256"] == mixture_hashes[i],
                            f"{path.name}: client mixture")
                    require(client["mixture_kind_counts"] == dict(sorted(Counter(
                        x["kind"] for x in mixture).items())),
                        f"{path.name}: mixture composition")
                    counts = client["selected_kind_counts_audit_only"]
                    require(sum(counts.values()) == fixed["matched_k"] ==
                            client["selection"]["selected"],
                            f"{path.name}: exact matched K")
                    require(counts.get("off_task", 0) == 0,
                            f"{path.name}: eligible pool")
                    require(client["selected_claimed_label_counts"] ==
                            {"0": fixed["matched_quota_per_claimed_label"],
                             "1": fixed["matched_quota_per_claimed_label"]},
                            f"{path.name}: claimed-label balance")
                    if method == "severe_random_k":
                        expected, _ = runner.select_random_k(
                            mixture, fixed["matched_k"], seed, i)
                        require(client["selected_ids_sha256"] == core.adapter_record_ids(expected),
                                f"{path.name}: deterministic random selection")
                    elif method == "oracle_max_purity_k":
                        expected, _ = runner.select_oracle_max_purity_k(
                            mixture, fixed["matched_k"], seed, i)
                        require(counts == {"clean": 103, "conflict": 25} and
                                client["selected_ids_sha256"] == core.adapter_record_ids(expected),
                                f"{path.name}: fail-closed maximum-purity oracle")
                    else:
                        require(client["selection"]["forced_selection"] is True and
                                client["selection"]["quota_per_claimed_label"] == fixed["matched_k"] // 2,
                                f"{path.name}: forced routed K")
                    require(client["selection"]["known_task_membership_used"] is True,
                            f"{path.name}: task-membership disclosure")
                    require(client["selection"]["hidden_clean_conflict_metadata_used"] is
                            (method == "oracle_max_purity_k"),
                            f"{path.name}: hidden-metadata disclosure")
                    prob = client["probability_payload"]
                    require(prob["shape"] == [128, 2] and prob["max_row_sum_error"] <= 1e-6,
                            f"{path.name}: probability payload")
                    payload_total += client["serialized_probability_payload"]["payload_bytes"]
            require(record["total_client_to_server_payload_bytes"] == payload_total,
                    f"{path.name}: payload total")
    return {"phase": phase, "records": len(expected_names),
            "protocol_sha256": protocol_sha, "runner_sha256": runner_sha,
            "seeds": seeds, "methods": methods}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["development", "official_evaluation"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(
        "kafa_script/imdb_matched_control_addendum_protocol_v1.json"))
    args = parser.parse_args()
    print(json.dumps(verify_matrix(args.phase, args.output_dir, args.protocol), indent=2))


if __name__ == "__main__":
    main()
