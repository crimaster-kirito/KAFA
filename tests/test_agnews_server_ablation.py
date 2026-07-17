import hashlib
import json
from pathlib import Path

from kafa_script import run_agnews_server_ablation as runner


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "kafa_script/agnews_server_post_primary_ablation_v1.json"
HOLDOUT_PATH = ROOT / "kafa_script/agnews_secondary_holdout_manifest_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_protocol_matches_runner_and_method_contract():
    protocol = json.loads(PROTOCOL_PATH.read_text())
    assert protocol["status"] == (
        "refrozen_after_prior_training_exposure_audit_with_all_previous_records_invalidated")
    assert protocol["instrumentation_repair"]["evidence_observed"].startswith(
        "No JSON record or evaluation metric")
    assert protocol["chronology_repair"]["invalidated_records"] == 4
    invalidated = ROOT / protocol["chronology_repair"]["invalidated_location"]
    assert len(list(invalidated.glob("development/seed*.json"))) == 4
    exposure = protocol["prior_training_exposure_repair"]
    assert exposure["audit"]["unique_holdout_ids_previously_allocated"] == 303
    assert exposure["invalidated_records"] == 11
    assert len(list((ROOT / exposure["invalidated_location"]).glob(
        "development/seed*.json"))) == 11
    expected = protocol["implementation"]["file_sha256"][
        "kafa_script/run_agnews_server_ablation.py"]
    assert sha256(ROOT / "kafa_script/run_agnews_server_ablation.py") == expected
    assert protocol["method_configuration"] == runner.METHODS


def test_frozen_matrix_and_statistical_family_are_exact():
    protocol = json.loads(PROTOCOL_PATH.read_text())
    methods = list(runner.METHODS)
    assert protocol["development"]["seeds"] == [271, 272, 273]
    assert protocol["secondary_holdout"]["seeds"] == list(range(281, 291))
    assert protocol["development"]["comparators"] == methods
    assert protocol["secondary_holdout"]["comparators"] == methods
    contrasts = protocol["analysis"]["adjacent_contrasts"]
    assert [(item["left"], item["right"]) for item in contrasts] == [
        ("clean_unfiltered_distillation", "public_anchor_only"),
        ("severe_unfiltered_distillation", "clean_unfiltered_distillation"),
        ("severe_routed_distillation", "severe_unfiltered_distillation"),
    ]


def test_secondary_holdout_is_balanced_unique_and_parent_disjoint():
    parent = json.loads((ROOT / "kafa_script/benchmark_manifest_v1.json").read_text())
    holdout = json.loads(HOLDOUT_PATH.read_text())
    records = holdout["holdout"]["records"]
    ids = [record["id"] for record in records]
    reserved = {
        record["id"]
        for split in ("anchor", "development")
        for record in parent["splits"][split]["records"]
    }
    assert len(records) == len(set(ids)) == 4000
    assert holdout["holdout"]["class_counts"] == {
        "0": 1000, "1": 1000, "2": 1000, "3": 1000}
    assert not (set(ids) & reserved)
    assert hashlib.sha256("\n".join(sorted(ids)).encode("ascii")).hexdigest() == \
        holdout["holdout"]["ids_sha256"]


def test_protocol_links_manifests_and_parent_protocols():
    protocol = json.loads(PROTOCOL_PATH.read_text())
    assert sha256(ROOT / protocol["data"]["parent_manifest"]) == \
        protocol["data"]["parent_manifest_sha256"]
    assert sha256(ROOT / protocol["data"]["holdout_manifest"]) == \
        protocol["data"]["holdout_manifest_sha256"]
    for record in protocol["parent_protocols"].values():
        assert sha256(ROOT / record["path"]) == record["sha256"]


def test_mixture_hash_tracks_assignment_but_not_text_payload():
    first = [{"kind": "clean", "source_id": "a", "training_label": 0,
              "answer": "World", "text": "one"}]
    same_assignment = [{**first[0], "text": "different"}]
    changed_assignment = [{**first[0], "training_label": 1}]
    assert runner.mixture_sha256(first) == runner.mixture_sha256(same_assignment)
    assert runner.mixture_sha256(first) != runner.mixture_sha256(changed_assignment)


def test_matrix_scheduler_keeps_gradient_routing_exclusive():
    script = (ROOT /
        "kafa_script/run_agnews_server_ablation_matrix.sh").read_text()
    assert 'if [[ "$method" == "severe_routed_distillation" ]]' in script
    exclusive = script.split("run_exclusive()", 1)[1].split(
        "for seed in", 1)[0]
    assert exclusive.count("wait") >= 2
