import json
from pathlib import Path

import pytest

from kafa_script import imdb_selector_component_ablation as selectors
from kafa_script.verify_imdb_selector_component_ablation import (
    SEEDS, verify_matrix)


def _record(seed: int, method: str) -> dict:
    client = {
        "selection": {
            "policy": selectors.METHODS[method]["selection"],
            "selected": 128,
            "quota_per_claimed_label": 64,
            "off_task_selected": 0,
            "known_task_membership_used": True,
            "hidden_clean_conflict_metadata_used": False,
        },
        "selected_kind_counts_audit_only": {"clean": 96, "conflict": 32},
        "selected_claimed_label_counts": {"0": 64, "1": 64},
        "probability_payload": {"shape": [128, 2], "max_row_sum_error": 0.0},
        "serialized_probability_payload": {"payload_bytes": 100},
    }
    return {
        "schema_version": "imdb_selector_component_ablation_v1",
        "evidence_label": "post_hoc_reused_test_selector_component_ablation",
        "paper_result": False,
        "seed": seed,
        "method": method,
        "method_configuration": selectors.METHODS[method],
        "evaluation_split": "official_test",
        "evaluation_examples": 25000,
        "anchor_examples": 128,
        "round_records": [
            {"teacher_weights": [1 / 3] * 3,
             "client_records": [client, client, client]},
            {"teacher_weights": [1 / 3] * 3,
             "client_records": [client, client, client]},
        ],
        "final": {
            "accuracy": 0.9,
            "macro_f1": 0.9,
            "canonical_label_nll": 0.2,
            "target_answer_loss": 0.4,
            "ece": 0.01,
        },
        "total_client_to_server_payload_bytes": 600,
    }


def _write_matrix(root: Path) -> None:
    root.mkdir(exist_ok=True)
    for seed in SEEDS:
        for method in selectors.METHODS:
            (root / f"seed{seed}_{method}.json").write_text(
                json.dumps(_record(seed, method)))


def test_selector_component_verifier_accepts_complete_matrix(tmp_path: Path):
    _write_matrix(tmp_path)
    receipt = verify_matrix(tmp_path)
    assert receipt["records"] == 30
    assert receipt["classification"] == "post_hoc_secondary_reused_test"
    assert set(receipt["mean_accuracy"]) == set(selectors.METHODS)


def test_selector_component_verifier_rejects_missing_record(tmp_path: Path):
    _write_matrix(tmp_path)
    (tmp_path / "seed321_loss_only_k.json").unlink()
    with pytest.raises(AssertionError, match="matrix mismatch"):
        verify_matrix(tmp_path)


def test_selector_component_verifier_rejects_hidden_metadata_use(
        tmp_path: Path):
    _write_matrix(tmp_path)
    path = tmp_path / "seed321_loss_only_k.json"
    record = json.loads(path.read_text())
    record["round_records"][0]["client_records"][0]["selection"][
        "hidden_clean_conflict_metadata_used"] = True
    path.write_text(json.dumps(record))
    with pytest.raises(AssertionError, match="metadata disclosure"):
        verify_matrix(tmp_path)
