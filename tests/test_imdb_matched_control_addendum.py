import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from kafa_script.classification_task import load_task_spec
from kafa_script import run_generic_classification_distillation as core
from kafa_script import run_imdb_server_ablation as base
from kafa_script import run_imdb_matched_control_addendum as runner
from kafa_script.summarize_imdb_matched_control_addendum import holm_adjust
from kafa_script.verify_imdb_routed_equivalence import verify_pair


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "kafa_script/imdb_matched_control_addendum_protocol_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def mixtures_by_seed():
    manifest = json.loads((ROOT / "kafa_script/imdb_server_manifest_v1.json").read_text())
    rows = base.load_jsonl_unique_with_lines(ROOT / ".cache/datasets/imdb/train.jsonl")
    parts = core.resolve_manifest_splits(
        rows, manifest, ["client_0", "client_1", "client_2"])
    bases = [parts[f"client_{i}"] for i in range(3)]
    off_task = core.load_jsonl(ROOT / ".cache/datasets/agnews/train.jsonl")
    target = load_task_spec(ROOT / "kafa_script/task_specs/imdb_sentiment_server_v1.json")
    off_spec = load_task_spec(ROOT / "kafa_script/task_specs/agnews_offtask_v1.json")
    return {
        seed: core.build_client_mixtures(
            bases, off_task, target, off_spec, seed,
            runner.SEVERE_CONFLICT_RATIO, runner.SEVERE_OFF_TASK_RATIO)
        for seed in [311, 312, 313, *range(321, 331)]
    }


def test_protocol_is_secondary_matched_k_and_hash_locks_implementation():
    protocol = json.loads(PROTOCOL_PATH.read_text())
    assert protocol["status"].startswith(
        "frozen_after_imdb_development_observation_before_any")
    assert protocol["fixed_configuration"]["matched_k"] == 128
    assert protocol["analysis"]["classification"] == \
        "prospective secondary mechanistic addendum"
    assert [item["id"] for item in protocol["analysis"]["secondary_holm_family"]] == [
        "routed_k_minus_random_k", "oracle_max_purity_k_minus_routed_k"]
    assert "exactly the two" in protocol["analysis"]["multiplicity"]
    assert protocol["method_configuration"] == runner.METHODS
    for relative, expected in protocol["implementation"]["file_sha256"].items():
        assert sha256(ROOT / relative) == expected


def test_original_frozen_runner_and_protocol_were_not_modified_by_addendum():
    assert sha256(ROOT / "kafa_script/run_imdb_server_ablation.py") == \
        "95853ad21d850e9fe68464b01b3a179b0d487cb6a8c66e647bc6b08041792459"
    assert sha256(ROOT / "kafa_script/imdb_server_role_reversal_protocol_v1.json") == \
        "fbfc13064eca36163397017521474dde6aeda7c4304215b97491845c14ee9166"


def test_random_k_is_deterministic_balanced_and_uses_target_eligible_pool(
        mixtures_by_seed):
    for seed, mixtures in mixtures_by_seed.items():
        for client_index, mixture in enumerate(mixtures):
            first, audit = runner.select_random_k(mixture, 128, seed, client_index)
            second, _ = runner.select_random_k(mixture, 128, seed, client_index)
            assert [x["source_id"] for x in first] == [x["source_id"] for x in second]
            assert len(first) == 128
            assert Counter(x["training_label"] for x in first) == {0: 64, 1: 64}
            assert Counter(x["kind"] for x in first).get("off_task", 0) == 0
            assert audit["known_task_membership_used"] is True
            assert audit["hidden_clean_conflict_metadata_used"] is False
            changed, _ = runner.select_random_k(mixture, 128, seed + 10000, client_index)
            assert {x["source_id"] for x in first} != {x["source_id"] for x in changed}


def test_oracle_is_maximum_purity_not_falsely_clean_only(mixtures_by_seed):
    for seed, mixtures in mixtures_by_seed.items():
        for client_index, mixture in enumerate(mixtures):
            selected, audit = runner.select_oracle_max_purity_k(
                mixture, 128, seed, client_index)
            assert len(selected) == 128
            assert Counter(x["kind"] for x in selected) == {
                "clean": 103, "conflict": 25}
            assert Counter(x["training_label"] for x in selected) == {0: 64, 1: 64}
            assert audit["selected_clean"] == 103
            assert audit["selected_conflict"] == 25
            assert audit["known_task_membership_used"] is True
            assert audit["hidden_clean_conflict_metadata_used"] is True


def test_maximum_purity_fails_closed_when_quota_cannot_be_filled():
    mixture = [
        {"kind": "clean", "source_id": f"c{i}", "training_label": 0,
         "original_label": 0}
        for i in range(64)
    ] + [
        {"kind": "conflict", "source_id": f"x{i}", "training_label": 0,
         "original_label": 1}
        for i in range(64)
    ]
    with pytest.raises(ValueError, match="fails closed"):
        runner.select_oracle_max_purity_k(mixture, 128, 1, 0)


def test_holm_adjustment_is_monotone_in_sorted_order_and_bounded():
    assert holm_adjust([0.03, 0.01]) == pytest.approx([0.03, 0.02])
    assert holm_adjust([0.8, 0.9]) == pytest.approx([1.0, 1.0])


def test_development_sentinel_is_separate_and_official_boundary_is_explicit():
    protocol = json.loads(PROTOCOL_PATH.read_text())
    sentinel = ROOT / protocol["development"]["nonexistent_test_sentinel"]
    assert not sentinel.exists()
    boundary = protocol["chronology"]["official_test_access_during_addendum_design"]
    assert "did not open" in boundary and "decode" in boundary
    assert protocol["data"]["official_test_sha256"] == \
        "bceb588619d37a5b1c5e6ec7ec519b204eaaf8a4024735b279fe1eff33a46843"


def test_matrix_is_serial_and_does_not_launch_on_import():
    script = (ROOT / "kafa_script/run_imdb_matched_control_addendum_matrix.sh").read_text()
    assert "&)" not in script and ") &" not in script
    assert "run_one \"$seed\" \"$method\"" in script


def test_equivalence_verifier_rejects_a_triggered_functional_difference(tmp_path):
    client = {
        "client_index": 0, "model": "m", "base_target_ids_sha256": ["a"],
        "mixture_assignment_sha256": "mix", "selected_ids_sha256": ["s"],
        "mixture_kind_counts": {"clean": 128},
        "selected_kind_counts_audit_only": {"clean": 128},
        "selected_claimed_label_counts": {"0": 64, "1": 64},
        "mean_local_loss": 1.0, "probability_payload": {"shape": [1, 2]},
        "serialized_probability_payload": {"payload_sha256": "p"},
        "adapter_bytes": 1,
        "selection": {
            "policy": "relative_balanced_routing", "gate_ratio": 0.5,
            "client_q25": 0.25, "reference_q25": 0.5,
            "score_record_order_sha256": "order", "gate_triggered": True,
            "gate_threshold": 0.8, "selected": 128,
            "quota_per_claimed_label": 64, "off_task_selected": 0,
        },
    }
    record = {
        "seed": 1, "evaluation_split": "development",
        "evaluation_examples": 1, "anchor_examples": 1,
        "mixture_assignment_sha256": ["mix"], "final": {"accuracy": 1.0},
        "total_client_to_server_payload_bytes": 1,
        "round_records": [{"round": 1, "teacher_weights": [1.0],
                            "mean_server_objective": 1.0,
                            "client_records": [client]}],
    }
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text(json.dumps(record))
    changed = json.loads(json.dumps(record))
    changed_client = changed["round_records"][0]["client_records"][0]
    changed_client["selection"].update({
        "policy": "forced_relative_balanced_routing_k",
        "gate_threshold": 1e30, "forced_selection": True,
        "known_task_membership_used": True,
        "hidden_clean_conflict_metadata_used": False,
    })
    changed_client["selected_ids_sha256"] = ["different"]
    right.write_text(json.dumps(changed))
    with pytest.raises(AssertionError, match="triggered branch differs"):
        verify_pair(left, right)


def test_equivalence_verifier_accepts_only_gate_as_first_divergence(tmp_path):
    client = {
        "client_index": 0, "model": "m", "base_target_ids_sha256": ["a"],
        "mixture_assignment_sha256": "mix", "selected_ids_sha256": ["all"],
        "mixture_kind_counts": {"clean": 103, "conflict": 102, "off_task": 51},
        "selected_kind_counts_audit_only": {
            "clean": 103, "conflict": 102, "off_task": 51},
        "selected_claimed_label_counts": {"0": 128, "1": 128},
        "mean_local_loss": 1.0, "probability_payload": {"shape": [128, 2]},
        "serialized_probability_payload": {"payload_sha256": "original"},
        "adapter_bytes": 1,
        "selection": {
            "policy": "relative_balanced_routing", "gate_ratio": 0.9,
            "client_q25": 0.45, "reference_q25": 0.5,
            "score_record_order_sha256": "order", "gate_triggered": False,
            "gate_threshold": 0.8, "selected": 256,
        },
    }
    original = {
        "seed": 1, "evaluation_split": "development",
        "evaluation_examples": 1, "anchor_examples": 1,
        "mixture_assignment_sha256": ["mix"], "final": {"accuracy": 0.5},
        "total_client_to_server_payload_bytes": 1,
        "round_records": [{"round": 1, "teacher_weights": [1.0],
                            "mean_server_objective": 1.0,
                            "client_records": [client]}],
    }
    matched = json.loads(json.dumps(original))
    routed = matched["round_records"][0]["client_records"][0]
    routed["selected_ids_sha256"] = ["routed"]
    routed["selected_kind_counts_audit_only"] = {"clean": 103, "conflict": 25}
    routed["selected_claimed_label_counts"] = {"0": 64, "1": 64}
    routed["mean_local_loss"] = 0.8
    routed["serialized_probability_payload"] = {"payload_sha256": "matched"}
    routed["selection"].update({
        "policy": "forced_relative_balanced_routing_k", "gate_triggered": True,
        "gate_threshold": 1e30, "selected": 128,
        "quota_per_claimed_label": 64, "off_task_selected": 0,
        "forced_selection": True, "known_task_membership_used": True,
        "hidden_clean_conflict_metadata_used": False,
    })
    matched["final"] = {"accuracy": 0.6}
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text(json.dumps(original))
    right.write_text(json.dumps(matched))
    receipt = verify_pair(left, right)
    assert receipt["conditional_implementation_equivalence"] is True
    assert receipt["fully_identical_trajectory"] is False
    assert len(receipt["first_forced_gate_divergences"]) == 1
    assert receipt["exact_pre_fork_triggered_client_rounds"] == 0
    assert receipt["first_authorized_gate_divergence_client_rounds"] == 1
    assert receipt[
        "post_local_fork_not_equivalence_eligible_client_rounds"] == 0

    # A pre-branch router-score difference cannot be excused as forced-K.
    bad = json.loads(json.dumps(matched))
    bad["round_records"][0]["client_records"][0]["selection"]["gate_ratio"] = 0.91
    right.write_text(json.dumps(bad))
    with pytest.raises(AssertionError, match="pre-branch router differs"):
        verify_pair(left, right)
