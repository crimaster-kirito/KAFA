import hashlib
import json
from pathlib import Path

from kafa_script import run_imdb_server_ablation as runner


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "kafa_script/imdb_server_role_reversal_protocol_v1.json"
MANIFEST_PATH = ROOT / "kafa_script/imdb_server_manifest_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_was_frozen_with_exact_runner_and_four_arms():
    protocol = json.loads(PROTOCOL_PATH.read_text())
    assert protocol["status"] == (
        "refrozen_after_pre_record_training_line_audit_repair_before_any_imdb_metric_or_test_json_decoding")
    assert protocol["instrumentation_repair"]["evidence_observed"].startswith(
        "No JSON record or model evaluation metric")
    assert sha256(ROOT / "kafa_script/run_imdb_server_ablation.py") == \
        protocol["implementation"]["file_sha256"][
            "kafa_script/run_imdb_server_ablation.py"]
    assert protocol["method_configuration"] == runner.METHODS
    assert protocol["development"]["seeds"] == [311, 312, 313]
    assert protocol["official_evaluation"]["seeds"] == list(range(321, 331))


def test_manifest_is_balanced_disjoint_and_content_addressed():
    manifest = json.loads(MANIFEST_PATH.read_text())
    protocol = json.loads(PROTOCOL_PATH.read_text())
    assert sha256(MANIFEST_PATH) == protocol["data"]["manifest_sha256"]
    owners = {}
    expected = {
        "anchor": 128, "development": 1000,
        "client_0": 256, "client_1": 256, "client_2": 256,
    }
    for split, count in expected.items():
        record = manifest["splits"][split]
        assert record["count"] == len(record["records"]) == count
        assert record["class_counts"] == {"0": count // 2, "1": count // 2}
        ids = [item["source_id"] for item in record["records"]]
        assert hashlib.sha256("\n".join(ids).encode("ascii")).hexdigest() == \
            record["source_ids_sha256"]
        for source_id in ids:
            assert source_id not in owners
            owners[source_id] = split


def test_test_boundary_and_single_primary_are_explicit():
    protocol = json.loads(PROTOCOL_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())
    sentinel = ROOT / protocol["development"]["nonexistent_test_sentinel"]
    assert not sentinel.exists()
    test_record = manifest["datasets"]["target_official_test"]
    assert test_record["count"] == 25000
    assert "does not decode test JSON" in test_record["inspection_policy"]
    primary = protocol["analysis"]["single_primary_contrast"]
    assert (primary["left"], primary["right"], primary["metric"]) == (
        "severe_routed_distillation",
        "severe_unfiltered_distillation",
        "accuracy",
    )
    assert any(item.startswith("No additional p-values")
               for item in protocol["analysis"]["prohibited"])


def test_frozen_pre_run_artifact_audit_has_zero_historical_hits():
    audit = json.loads((ROOT /
        "kafa_script/imdb_test_pre_run_audit_v1.json").read_text())
    protocol_sha = sha256(PROTOCOL_PATH)
    assert audit["historical_artifact_files_scanned"] == 586
    assert audit["matching_historical_artifacts"] == []
    assert audit["final_protocol_sha256"] == protocol_sha
    assert "retained workspace history only" in audit["interpretation"]


def test_official_loader_preserves_duplicate_rows_with_unique_ids(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"text":"same","label":0}\n{"text":"same","label":0}\n',
        encoding="utf-8")
    rows = runner.load_jsonl_preserve_rows(path)
    assert len(rows) == 2
    assert rows[0]["text"] == rows[1]["text"] == "same"
    assert rows[0]["source_id"] != rows[1]["source_id"]


def test_training_loader_deduplicates_but_retains_original_lines(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        '{"text":"first","label":0}\n'
        '{"text":"first","label":0}\n'
        '{"text":"second","label":1}\n', encoding="utf-8")
    rows = runner.load_jsonl_unique_with_lines(path)
    assert [(row["text"], row["line"]) for row in rows] == [
        ("first", 1), ("second", 3)]


def test_matrix_scheduler_keeps_gradient_routing_exclusive():
    script = (ROOT /
        "kafa_script/run_imdb_server_role_reversal_matrix.sh").read_text()
    assert 'if [[ "$method" == "severe_routed_distillation" ]]' in script
    exclusive = script.split("run_exclusive()", 1)[1].split(
        "for seed in", 1)[0]
    assert exclusive.count("wait") >= 2


def test_verifier_reuses_line_preserving_training_loader():
    source = (ROOT /
        "kafa_script/verify_imdb_server_role_reversal.py").read_text()
    assert "target_rows = load_jsonl_unique_with_lines(target_path)" in source
    assert "target_rows = core.load_jsonl(target_path)" not in source
