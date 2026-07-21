"""Fail-closed structural verifier for IMDb selector-component ablations."""

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

from kafa_script import imdb_selector_component_ablation as selectors  # noqa: E402
from kafa_script import run_generic_classification_distillation as core  # noqa: E402


SEEDS = list(range(321, 331))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_record(path: Path, expected_seed: int, expected_method: str) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    require(record.get("schema_version") == "imdb_selector_component_ablation_v1",
            f"{path.name}: schema")
    require(record.get("evidence_label") ==
            "post_hoc_reused_test_selector_component_ablation",
            f"{path.name}: evidence label")
    require(record.get("seed") == expected_seed and
            record.get("method") == expected_method,
            f"{path.name}: identity")
    require(record.get("method_configuration") ==
            selectors.METHODS[expected_method],
            f"{path.name}: method configuration")
    require(record.get("evaluation_split") == "official_test",
            f"{path.name}: evaluation split")
    require(record.get("paper_result") is False,
            f"{path.name}: post-hoc record must not be primary paper result")
    require(record.get("evaluation_examples") == 25000,
            f"{path.name}: official evaluation count")
    require(record.get("anchor_examples") == 128,
            f"{path.name}: anchor count")
    for metric in ("accuracy", "macro_f1", "canonical_label_nll",
                   "target_answer_loss", "ece"):
        require(math.isfinite(float(record["final"][metric])),
                f"{path.name}: non-finite {metric}")
    selected_clean, selected_conflict = 0, 0
    payload_total = 0
    for rr in record["round_records"]:
        require(rr["teacher_weights"] == [1 / 3] * 3,
                f"{path.name}: teacher weights")
        require(len(rr["client_records"]) == 3,
                f"{path.name}: client count")
        for client in rr["client_records"]:
            selection = client["selection"]
            require(selection["policy"] ==
                    selectors.METHODS[expected_method]["selection"],
                    f"{path.name}: selection policy")
            require(selection["selected"] == 128 and
                    selection["quota_per_claimed_label"] == 64,
                    f"{path.name}: exact K/quota")
            require(selection["off_task_selected"] == 0,
                    f"{path.name}: off-task selected")
            require(selection["known_task_membership_used"] is True and
                    selection["hidden_clean_conflict_metadata_used"] is False,
                    f"{path.name}: metadata disclosure")
            counts = client["selected_kind_counts_audit_only"]
            require(sum(counts.values()) == 128 and counts.get("off_task", 0) == 0,
                    f"{path.name}: selected composition")
            require(client["selected_claimed_label_counts"] == {"0": 64, "1": 64},
                    f"{path.name}: claimed-label balance")
            prob = client["probability_payload"]
            require(prob["shape"] == [128, 2] and
                    prob["max_row_sum_error"] <= 1e-6,
                    f"{path.name}: probability payload")
            selected_clean += counts.get("clean", 0)
            selected_conflict += counts.get("conflict", 0)
            payload_total += client[
                "serialized_probability_payload"]["payload_bytes"]
    require(record["total_client_to_server_payload_bytes"] == payload_total,
            f"{path.name}: payload total")
    return {
        "accuracy": float(record["final"]["accuracy"]),
        "selected_kind_counts": {
            "clean": selected_clean,
            "conflict": selected_conflict,
        },
    }


def _load_protocol(protocol_path: Path | None) -> dict | None:
    if protocol_path is None:
        return None
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(protocol["name"] == "imdb_selector_component_ablation_v1",
            "unexpected protocol")
    require(protocol["analysis"]["classification"].startswith("post-hoc"),
            "selector-component protocol must remain post-hoc")
    for relative, expected in protocol["implementation"]["file_sha256"].items():
        require(core.sha256_file(ROOT / relative) == expected,
                f"implementation hash mismatch: {relative}")
    return protocol


def verify_matrix(output_dir: Path, protocol_path: Path | None = None) -> dict:
    protocol = _load_protocol(protocol_path)
    methods = (protocol["official_evaluation"]["comparators"]
               if protocol is not None else sorted(selectors.METHODS))
    seeds = (protocol["official_evaluation"]["seeds"]
             if protocol is not None else SEEDS)
    require(methods == sorted(selectors.METHODS), "method family mismatch")
    require(seeds == SEEDS, "seed family mismatch")
    expected = {f"seed{seed}_{method}.json" for seed in seeds for method in methods}
    observed = {path.name for path in output_dir.glob("seed*.json")}
    require(expected == observed,
            f"matrix mismatch missing={sorted(expected - observed)} "
            f"extra={sorted(observed - expected)}")
    records = {}
    for seed in seeds:
        for method in methods:
            records[(seed, method)] = verify_record(
                output_dir / f"seed{seed}_{method}.json", seed, method)
    return {
        "schema_version": "imdb_selector_component_ablation_verifier_v1",
        "phase": "official_evaluation",
        "classification": "post_hoc_secondary_reused_test",
        "records": len(records),
        "seeds": seeds,
        "methods": methods,
        "mean_accuracy": {
            method: sum(records[(seed, method)]["accuracy"] for seed in seeds) /
            len(seeds)
            for method in methods
        },
        "selected_kind_counts": {
            method: dict(sum((Counter(records[(seed, method)][
                "selected_kind_counts"]) for seed in seeds), Counter()))
            for method in methods
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(
        "kafa_script/imdb_selector_component_ablation_protocol_v1.json"))
    args = parser.parse_args()
    print(json.dumps(verify_matrix(args.output_dir, args.protocol), indent=2))


if __name__ == "__main__":
    main()
