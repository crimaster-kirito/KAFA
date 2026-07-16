"""Fail-closed verifier for the post-primary TREC four-arm ablation."""

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

from kafa_script.classification_task import canonical_sha256
from kafa_script.run_generic_classification_distillation import sha256_file

PROTOCOL_PATH = ROOT / "kafa_script/trec_server_post_primary_ablation_v1.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expected_path(output_dir: Path, seed: int, method: str) -> Path:
    return output_dir / f"seed{seed}_{method}.json"


def verify_probability_record(record: dict) -> None:
    canonical = record["probability_payload"]
    wire = record["serialized_probability_payload"]
    require(canonical["encoding"] == "shape-prefixed little-endian-float64-v1",
            "unexpected canonical probability encoding")
    require(canonical["shape"] == [72, 6], "probability shape must be 72x6")
    require(0 <= canonical["minimum"] <= canonical["maximum"] <= 1,
            "probability bounds invalid")
    require(canonical["max_row_sum_error"] <= 1e-5,
            "probability rows do not sum to one")
    require(canonical["payload_bytes"] == 3485,
            "canonical audit payload size changed")
    require(len(canonical["payload_sha256"]) == 64,
            "canonical probability hash malformed")
    require(wire["encoding"] == "torch-save-tensor",
            "unexpected serialized tensor encoding")
    require(wire["payload_bytes"] == 3305,
            "serialized tensor payload size changed")
    require(len(wire["payload_sha256"]) == 64,
            "serialized tensor hash malformed")


def verify_matrix(phase: str, output_dir: Path,
                  protocol_path: Path = PROTOCOL_PATH) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(protocol["status"] ==
            "refrozen_after_selected_composition_instrumentation_repair_before_any_valid_official_ablation_record",
            "protocol chronology marker changed")
    require(protocol["official_test_access_history"]["previously_accessed"] is True,
            "official-test reuse is not disclosed")
    require(protocol["confirmatory"] is False,
            "post-primary ablation cannot be confirmatory")
    require(protocol["role"] ==
            "post_primary_reused_official_test_mechanistic_ablation",
            "post-primary role marker changed")
    official = protocol["official_evaluation"]
    require(official["primary_metric"] == "accuracy",
            "primary metric changed")
    expected_contrasts = [
        ("clean_client_utility", "clean_unfiltered_distillation",
         "public_anchor_only"),
        ("severe_contamination_effect", "severe_unfiltered_distillation",
         "clean_unfiltered_distillation"),
        ("routing_increment", "severe_routed_distillation",
         "severe_unfiltered_distillation"),
    ]
    require([(item["name"], item["left"], item["right"])
             for item in official["primary_contrasts"]] == expected_contrasts,
            "primary contrast family changed")
    require(all(token in official["analysis"] for token in
                ("two-sided", "Holm", "sign-flip")),
            "primary analysis contract changed")
    require(official["secondary_metrics"] == [
        "macro_f1", "canonical_label_nll", "ECE", "per_class_recall",
        "selection_composition", "serialized_probability_tensor_bytes",
        "peak_allocated_gpu_bytes", "scheduler_specific_wall_seconds"],
        "secondary metric ledger changed")
    require("No equivalence" in official["equivalence_policy"],
            "equivalence boundary changed")
    require(any("post-primary" in boundary.lower()
                for boundary in protocol["evidence_boundaries"]),
            "post-primary claim boundary missing")
    implementation = protocol["implementation"]
    require(sha256_file(ROOT / implementation["ablation_runner"]) ==
            implementation["ablation_runner_file_sha256"],
            "ablation runner hash mismatch")
    require(sha256_file(ROOT / implementation["frozen_primary_runner"]) ==
            implementation["frozen_primary_runner_file_sha256"],
            "frozen primary runner hash mismatch")
    require(sha256_file(ROOT / implementation["classification_task_core"]) ==
            implementation["classification_task_file_sha256"],
            "classification core hash mismatch")
    require(sha256_file(ROOT / protocol["parent_primary_protocol"]["path"]) ==
            protocol["parent_primary_protocol"]["sha256"],
            "parent protocol hash mismatch")
    require(sha256_file(ROOT / protocol["task"]["manifest"]) ==
            protocol["task"]["manifest_file_sha256"],
            "manifest hash mismatch")
    for relative_path, expected_hash in protocol["model_weight_sha256"].items():
        require(sha256_file(ROOT / relative_path) == expected_hash,
                f"model weight hash mismatch: {relative_path}")
    require(phase in {"development", "official_evaluation"}, "unknown phase")
    section = protocol[phase]
    seeds, methods = section["seeds"], section["comparators"]
    expected = {expected_path(output_dir, seed, method)
                for seed in seeds for method in methods}
    observed = set(output_dir.glob("*.json")) if output_dir.exists() else set()
    require(observed == expected,
            f"matrix files differ: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}")
    require(not list(output_dir.glob("*.tmp*")), "temporary outputs remain")

    protocol_sha = sha256_file(protocol_path)
    records: dict[tuple[int, str], dict] = {}
    base_ids_by_seed: dict[int, list[str]] = {}
    for path in sorted(expected):
        raw = json.loads(path.read_text(encoding="utf-8"))
        seed, method = int(raw["seed"]), raw["method"]
        require(path == expected_path(output_dir, seed, method),
                f"filename/content mismatch: {path}")
        require((seed, method) not in records, "duplicate seed/method")
        require(raw["schema_version"] == "trec_server_ablation_v1",
                "unexpected schema")
        require(raw["protocol_sha256"] == protocol_sha,
                "record protocol hash mismatch")
        require(raw["runner_file_sha256"] ==
                implementation["ablation_runner_file_sha256"],
                "record runner hash mismatch")
        require(raw["frozen_primary_runner_file_sha256"] ==
                implementation["frozen_primary_runner_file_sha256"],
                "record parent runner hash mismatch")
        require(raw["parent_primary_protocol_sha256"] ==
                protocol["parent_primary_protocol"]["sha256"],
                "record parent protocol hash mismatch")
        require(raw["classification_task_file_sha256"] ==
                implementation["classification_task_file_sha256"],
                "record core hash mismatch")
        require(raw["method_configuration"] ==
                protocol["method_configuration"][method],
                "record method configuration mismatch")
        require(raw["fixed_configuration"] == protocol["fixed_configuration"],
                "record fixed configuration mismatch")
        server_weight = protocol["model_weight_sha256"][
            protocol["fixed_configuration"]["server_model"] +
            "/model.safetensors"]
        require(raw["model_artifacts"]["server"]["model.safetensors"] ==
                server_weight, "record server weight hash mismatch")
        expected_client_weights = [
            protocol["model_weight_sha256"][path + "/model.safetensors"]
            for path in protocol["fixed_configuration"]["client_models"]]
        require([item["model.safetensors"]
                 for item in raw["model_artifacts"]["clients"]] ==
                expected_client_weights,
                "record client weight hash mismatch")
        require(raw["full_args_sha256"] == canonical_sha256(raw["full_args"]),
                "full-args hash mismatch")
        expected_split = "development" if phase == "development" else "official_test"
        expected_label = ("post_primary_development" if phase == "development"
                          else "post_primary_official_test_ablation")
        require(raw["evaluation_split"] == expected_split,
                "evaluation split mismatch")
        require(raw["evidence_label"] == expected_label,
                "evidence label mismatch")
        require(raw["paper_result"] == (phase == "official_evaluation"),
                "paper-result flag mismatch")
        expected_hash_source = ("frozen_protocol_not_reopened_in_development"
                                if phase == "development" else
                                "verified_raw_bytes")
        require(raw["datasets"]["target_official_test_hash_source"] ==
                expected_hash_source, "official-test access marker mismatch")
        require(raw["evaluation_examples"] ==
                (72 if phase == "development" else 500),
                "evaluation count mismatch")
        require(raw["anchor_examples"] == 72, "anchor count mismatch")
        require(len(raw["round_records"]) == 2, "round count mismatch")
        final = raw["final"]
        for metric in ("accuracy", "macro_f1", "canonical_label_nll",
                       "target_answer_loss", "ece"):
            require(math.isfinite(final[metric]), f"non-finite {metric}")
        require(len(final["per_class_recall"]) == 6,
                "per-class recall count mismatch")
        require(len(final["confusion_matrix"]) == 6 and
                all(len(row) == 6 for row in final["confusion_matrix"]),
                "confusion matrix shape mismatch")

        if method == "public_anchor_only":
            require(raw["total_client_to_server_payload_bytes"] == 0,
                    "anchor-only payload must be zero")
            require(all(not round_record["client_records"]
                        for round_record in raw["round_records"]),
                    "anchor-only must omit client path")
        else:
            round_base_ids = []
            for round_record in raw["round_records"]:
                clients = round_record["client_records"]
                require(len(clients) == 3, "distillation arm needs three clients")
                round_ids = []
                for client_index, client in enumerate(clients):
                    require(client["client_index"] == client_index,
                            "client order mismatch")
                    round_ids.append(client["base_target_ids_sha256"])
                    verify_probability_record(client)
                    if method == "clean_unfiltered_distillation":
                        expected_counts = {"clean": 120}
                    else:
                        expected_counts = {
                            "clean": 48, "conflict": 48, "off_task": 24}
                    require(client["mixture_kind_counts"] == expected_counts,
                            "mixture counts mismatch")
                    selection = client["selection"]
                    if method.endswith("unfiltered_distillation"):
                        require(selection == {
                            "policy": "all_examples", "selected": 120,
                            "routing_executed": False},
                            "unfiltered selection record mismatch")
                        require(client["selected_kind_counts_audit_only"] ==
                                expected_counts,
                                "unfiltered selection must retain mixture")
                        require(client["selected_ids_sha256"] ==
                                raw["mixture_source_ids_sha256"][client_index],
                                "selected ID hash must equal mixture hash")
                    else:
                        require(selection["policy"] ==
                                "relative_balanced_routing" and
                                selection["routing_executed"] is True,
                                "routed selection marker mismatch")
                        selected_total = sum(
                            client["selected_kind_counts_audit_only"].values())
                        require(selected_total == selection["selected"],
                                "routed selected-kind count mismatch")
                        if selection["gate_triggered"]:
                            require(selection["selected"] == 60 and
                                    selection["quota_per_claimed_label"] == 10,
                                    "triggered route must retain 10 per class")
                            require(client["selected_kind_counts_audit_only"].get(
                                    "off_task", 0) == 0,
                                    "triggered route must exclude known off-task interface")
                        else:
                            require(selection["selected"] == 120,
                                    "non-triggered route must retain all rows")
                            require(client["selected_ids_sha256"] ==
                                    raw["mixture_source_ids_sha256"][client_index],
                                    "non-triggered route must retain full mixture")
                round_base_ids.append(round_ids)
            require(round_base_ids[0] == round_base_ids[1],
                    "base IDs changed across rounds")
            base_ids_by_seed.setdefault(seed, round_base_ids[0])
            require(base_ids_by_seed[seed] == round_base_ids[0],
                    "base IDs differ across arms")
            require(raw["total_client_to_server_payload_bytes"] == 19830,
                    "six tensor payloads must total 19,830 bytes")
        records[(seed, method)] = raw

    for seed in seeds:
        severe_unfiltered = records[(seed, "severe_unfiltered_distillation")]
        severe_routed = records[(seed, "severe_routed_distillation")]
        require(severe_unfiltered["mixture_assignment_sha256"] ==
                severe_routed["mixture_assignment_sha256"],
                f"severe mixture mismatch at seed {seed}")
        for round_index in range(2):
            for client_index in range(3):
                left = severe_unfiltered["round_records"][round_index][
                    "client_records"][client_index]
                right = severe_routed["round_records"][round_index][
                    "client_records"][client_index]
                require(left["mixture_assignment_sha256"] ==
                        right["mixture_assignment_sha256"],
                        "per-client severe mixture mismatch")
    return {
        "phase": phase,
        "records": len(records),
        "protocol_sha256": protocol_sha,
        "seeds": seeds,
        "methods": methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
                        choices=["development", "official_evaluation"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    args = parser.parse_args()
    print(json.dumps(verify_matrix(args.phase, args.output_dir, args.protocol),
                     indent=2))


if __name__ == "__main__":
    main()
