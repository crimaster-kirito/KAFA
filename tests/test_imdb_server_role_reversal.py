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
        "frozen_before_any_imdb_target_model_evaluation_or_test_json_decoding")
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


def test_pre_run_analysis_outputs_do_not_reference_imdb_test():
    needles = {
        ".cache/datasets/imdb/test.jsonl",
        "bceb588619d37a5b1c5e6ec7ec519b204eaaf8a4024735b279fe1eff33a46843",
    }
    hits = []
    root = ROOT / "kafa_script/analysis_outputs"
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".log"}:
            text = path.read_text(errors="replace")
            if any(needle in text for needle in needles):
                hits.append(path)
    assert hits == []


def test_official_loader_preserves_duplicate_rows_with_unique_ids(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"text":"same","label":0}\n{"text":"same","label":0}\n',
        encoding="utf-8")
    rows = runner.load_jsonl_preserve_rows(path)
    assert len(rows) == 2
    assert rows[0]["text"] == rows[1]["text"] == "same"
    assert rows[0]["source_id"] != rows[1]["source_id"]
