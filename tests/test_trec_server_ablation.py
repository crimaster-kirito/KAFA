import json
from pathlib import Path

import pytest

from kafa_script.run_generic_classification_distillation import sha256_file
from kafa_script.run_trec_server_ablation import METHODS
from kafa_script.summarize_trec_server_ablation import (
    exact_sign_flip_p, holm_adjust, mean_ci, paired_t_p)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "kafa_script/trec_server_post_primary_ablation_v1.json"


def test_ablation_protocol_is_post_primary_and_four_arm():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == (
        "refrozen_after_selected_composition_instrumentation_repair_before_any_valid_official_ablation_record")
    assert protocol["confirmatory"] is False
    assert protocol["official_test_access_history"]["previously_accessed"] is True
    assert protocol["method_configuration"] == METHODS
    assert protocol["official_evaluation"]["seeds"] == list(range(261, 271))
    assert len(protocol["official_evaluation"]["primary_contrasts"]) == 3


def test_ablation_implementation_hashes_close():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    implementation = protocol["implementation"]
    assert sha256_file(ROOT / implementation["ablation_runner"]) == (
        implementation["ablation_runner_file_sha256"])
    assert sha256_file(ROOT / implementation["frozen_primary_runner"]) == (
        implementation["frozen_primary_runner_file_sha256"])
    assert sha256_file(ROOT / implementation["classification_task_core"]) == (
        implementation["classification_task_file_sha256"])
    assert sha256_file(ROOT / protocol["parent_primary_protocol"]["path"]) == (
        protocol["parent_primary_protocol"]["sha256"])


def test_ablation_adjacent_contrasts_are_exactly_declared_family():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    contrasts = protocol["official_evaluation"]["primary_contrasts"]
    assert [(item["left"], item["right"]) for item in contrasts] == [
        ("clean_unfiltered_distillation", "public_anchor_only"),
        ("severe_unfiltered_distillation", "clean_unfiltered_distillation"),
        ("severe_routed_distillation", "severe_unfiltered_distillation"),
    ]
    assert "Holm" in protocol["official_evaluation"]["analysis"]


def test_severe_arms_share_mixture_configuration():
    assert METHODS["severe_unfiltered_distillation"]["conflict_ratio"] == 0.4
    assert METHODS["severe_routed_distillation"]["conflict_ratio"] == 0.4
    assert METHODS["severe_unfiltered_distillation"]["off_task_ratio"] == 0.2
    assert METHODS["severe_routed_distillation"]["off_task_ratio"] == 0.2
    assert METHODS["severe_unfiltered_distillation"]["selection"] == "all_examples"
    assert METHODS["severe_routed_distillation"]["selection"] == (
        "relative_balanced_routing")


def test_ablation_statistics_helpers():
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    with pytest.raises(ValueError):
        holm_adjust([float("nan"), 0.1, 0.2])
    assert exact_sign_flip_p([1.0, 1.0]) == 0.5
    assert paired_t_p([0.0] * 10) == 1.0
    center, low, high = mean_ci([1.0, 2.0, 3.0])
    assert center == 2.0
    assert low < center < high
