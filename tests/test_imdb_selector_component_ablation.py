from collections import Counter

import pytest

from kafa_script.imdb_selector_component_ablation import (
    METHODS, select_component_k_from_records)


def _toy_mixture() -> list[dict]:
    rows = []
    for label in (0, 1):
        for index in range(5):
            rows.append({
                "source_id": f"y{label}_{index}",
                "kind": "clean" if index < 3 else "conflict",
                "training_label": label,
            })
    rows.append({"source_id": "off_0", "kind": "off_task"})
    return rows


def _toy_records() -> list[dict]:
    records = []
    for label in (0, 1):
        for index in range(5):
            records.append({
                "source_id": f"y{label}_{index}",
                "claimed_label": label,
                "anchor_cosine": 1.0 - index * 0.1,
                "loss": 0.1 + index * 0.1,
                "semantic_margin": 0.2 + index * 0.1,
            })
    return records


@pytest.mark.parametrize("method", sorted(METHODS))
def test_component_selectors_are_balanced_target_eligible_and_deterministic(
        method: str) -> None:
    first, audit = select_component_k_from_records(
        _toy_mixture(), _toy_records(), method, 4, seed=321, client_index=0)
    second, _ = select_component_k_from_records(
        _toy_mixture(), _toy_records(), method, 4, seed=321, client_index=0)
    assert [row["source_id"] for row in first] == [
        row["source_id"] for row in second]
    assert len(first) == 4
    assert Counter(row["training_label"] for row in first) == {0: 2, 1: 2}
    assert Counter(row["kind"] for row in first).get("off_task", 0) == 0
    assert audit["known_task_membership_used"] is True
    assert audit["hidden_clean_conflict_metadata_used"] is False
    assert audit["quota_per_claimed_label"] == 2


def test_component_selectors_pick_expected_best_rows_without_ties() -> None:
    mixture, records = _toy_mixture(), _toy_records()
    for method in METHODS:
        selected, _ = select_component_k_from_records(
            mixture, records, method, 4, seed=321, client_index=0)
        assert [row["source_id"] for row in selected] == [
            "y0_0", "y0_1", "y1_0", "y1_1"]


def test_component_selector_rejects_score_identity_mismatch() -> None:
    records = _toy_records()
    records.pop()
    with pytest.raises(ValueError, match="identity mismatch"):
        select_component_k_from_records(
            _toy_mixture(), records, "loss_only_k", 4, 321, 0)


def test_margin_selector_rejects_offtask_score_records() -> None:
    mixture = _toy_mixture()
    records = _toy_records() + [{
        "source_id": "off_0",
        "claimed_label": None,
        "anchor_cosine": 0.0,
        "loss": 1.0,
        "semantic_margin": None,
    }]
    with pytest.raises(ValueError, match="identity mismatch"):
        select_component_k_from_records(
            mixture, records, "margin_only_k", 4, 321, 0)
