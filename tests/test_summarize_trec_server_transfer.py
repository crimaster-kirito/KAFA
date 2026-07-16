import json
import math
from pathlib import Path

import pytest

from kafa_script.classification_task import canonical_sha256
from kafa_script import summarize_trec_server_transfer as summary


def frozen_context():
    protocol, hashes = summary.validate_frozen_artifacts(
        summary.ROOT, summary.PROTOCOL_PATH)
    return protocol, hashes


def raw_record(protocol, hashes, phase, seed, method, output):
    official = phase == "official"
    split = "official_test" if official else "development"
    evidence = "frozen_official_test" if official else "benchmark_development"
    count = 500 if official else 72
    full_args = {
        "task_spec": "kafa_script/task_specs/trec_coarse_server_v1.json",
        "off_task_spec": "kafa_script/task_specs/imdb_sentiment_server_v1.json",
        "manifest": "kafa_script/trec_server_manifest_v1.json",
        "protocol": "kafa_script/trec_server_transfer_protocol_v1.json",
        "target_train": ".cache/datasets/trec/train_5500.label",
        "official_test": ".cache/datasets/trec/TREC_10.label",
        "off_task_train": ".cache/datasets/imdb/train.jsonl",
        "server_model": ".cache/models/Qwen2.5-1.5B",
        "client_models": ["client-a", "client-b", "client-c"],
        "method": method,
        "evaluation_split": split,
        "seed": seed,
        "rounds": 2,
        "output": output,
    }
    routed = method == "routed_distillation"
    rounds = []
    for round_index in (1, 2):
        clients = []
        if routed:
            for client_index in range(3):
                clients.append({
                    "client_index": client_index,
                    "routing": {"gate_triggered": (client_index + round_index) % 2 == 0},
                    "probability_payload": {
                        "shape": [72, 6], "max_row_sum_error": 1e-8,
                    },
                    "serialized_probability_payload": {
                        "encoding": "torch-save-tensor",
                        "payload_bytes": 3305,
                        "payload_sha256": f"{client_index + round_index:064x}",
                    },
                })
        rounds.append({"round": round_index, "client_records": clients,
                       "mean_server_objective": 1.0})
    seed_offset = (seed % 10) * .001
    accuracy = .80 + seed_offset + (.01 if routed else 0.0)
    final = {
        "accuracy": accuracy,
        "macro_f1": accuracy - .01,
        "canonical_label_nll": .55 - (.02 if routed else 0.0) + seed_offset,
        "ece": .10 - (.01 if routed else 0.0) + seed_offset,
    }
    return {
        "schema_version": "2.0",
        "evidence_label": evidence,
        "paper_result": official,
        "task_id": protocol["task"]["task_id"],
        "method": method,
        "seed": seed,
        "started_utc": "2026-07-15T00:00:00+00:00",
        "protocol_sha256": hashes["protocol"],
        "runner_file_sha256": hashes["runner"],
        "classification_task_file_sha256": hashes["core"],
        "task_spec_semantic_sha256": protocol["task"]["task_spec_semantic_sha256"],
        "task_spec_file_sha256": hashes["task_spec_file"],
        "off_task_spec_semantic_sha256": protocol["task"]["off_task_spec_semantic_sha256"],
        "manifest_sha256": hashes["manifest"],
        "full_args": full_args,
        "full_args_sha256": canonical_sha256(full_args),
        "git": {"commit": "deadbeef", "dirty": True},
        "datasets": {
            "target_train_sha256": protocol["task"]["target_train_sha256"],
            "target_official_test_sha256": protocol["task"]["target_official_test_sha256"],
            "target_official_test_hash_source": (
                "verified_raw_bytes" if official else
                "frozen_protocol_not_reopened_in_development"),
            "off_task_train_sha256": protocol["task"]["off_task_train_sha256"],
        },
        "model_artifacts": {"server": {"model": "a"},
                            "clients": [{"model": "b"}] * 3},
        "split_source_ids_sha256": {
            "anchor": "a" * 64,
            "evaluation": ("o" if official else "d") * 64,
            "client_0": "0" * 64,
            "client_1": "1" * 64,
            "client_2": "2" * 64,
        },
        "evaluation_split": split,
        "evaluation_examples": count,
        "anchor_examples": 72,
        "round_records": rounds,
        "final": final,
        "total_client_to_server_payload_bytes": 3305 * 6 if routed else 0,
        "wall_seconds": 20.0 if not routed else 180.0,
    }


def write_matrix(directory: Path, phase: str):
    protocol, hashes = frozen_context()
    seeds = (protocol["development"]["seeds"] if phase == "development" else
             protocol["official_evaluation"]["seeds"])
    directory.mkdir()
    for seed in seeds:
        for method in summary.METHODS:
            path = directory / f"seed{seed}_{method}.json"
            record = raw_record(protocol, hashes, phase, seed, method, str(path))
            path.write_text(json.dumps(record), encoding="utf-8")
    return protocol, hashes


def rewrite(path: Path, mutate):
    record = json.loads(path.read_text())
    mutate(record)
    if "full_args" in record:
        record["full_args_sha256"] = canonical_sha256(record["full_args"])
    path.write_text(json.dumps(record), encoding="utf-8")


def test_paired_statistics_matches_known_t_interval_and_two_sided_p():
    result = summary.paired_statistics([1.0, 2.0, 3.0])
    assert result.mean == pytest.approx(2.0)
    assert result.lower < 0 < result.upper
    assert result.pvalue == pytest.approx(0.0741799, rel=1e-5)
    degenerate = summary.paired_statistics([.1, .1, .1])
    assert degenerate.pvalue is None


def test_development_matrix_is_complete_valid_and_marked_non_paper(tmp_path):
    input_dir = tmp_path / "development"
    protocol, _ = write_matrix(input_dir, "development")
    matrix, validated_protocol, _ = summary.load_validated_matrix(
        input_dir, "development")
    assert validated_protocol == protocol
    assert len(matrix) == 6
    report = summary.summarize_matrix(matrix, protocol, "development")
    assert "development sanity evidence; non-paper result" in report
    assert "variation over the frozen training seeds only" in report
    assert "Gates triggered" in report


def test_missing_matrix_member_is_rejected(tmp_path):
    input_dir = tmp_path / "development"
    write_matrix(input_dir, "development")
    next(input_dir.glob("*.json")).unlink()
    with pytest.raises(summary.SummaryValidationError, match="expected 6"):
        summary.load_validated_matrix(input_dir, "development")


def test_args_may_only_differ_by_seed_method_and_output(tmp_path):
    input_dir = tmp_path / "development"
    write_matrix(input_dir, "development")
    path = next(input_dir.glob("*routed*.json"))
    rewrite(path, lambda record: record["full_args"].update({"rounds": 99}))
    with pytest.raises(summary.SummaryValidationError, match="full_args differ"):
        summary.load_validated_matrix(input_dir, "development")


def test_official_requires_verified_raw_test_hash_source(tmp_path):
    input_dir = tmp_path / "official"
    write_matrix(input_dir, "official")
    path = next(input_dir.glob("*.json"))
    rewrite(path, lambda record: record["datasets"].update({
        "target_official_test_hash_source":
            "frozen_protocol_not_reopened_in_development"}))
    with pytest.raises(summary.SummaryValidationError, match="access/hash source"):
        summary.load_validated_matrix(input_dir, "official")


@pytest.mark.parametrize("metric,value", [
    ("accuracy", float("nan")),
    ("macro_f1", float("inf")),
    ("canonical_label_nll", -1.0),
    ("ece", 2.0),
])
def test_nonfinite_or_out_of_range_metrics_are_rejected(tmp_path, metric, value):
    input_dir = tmp_path / "development"
    write_matrix(input_dir, "development")
    path = next(input_dir.glob("*.json"))
    rewrite(path, lambda record: record["final"].update({metric: value}))
    with pytest.raises(summary.SummaryValidationError,
                       match=f"{metric.split('_')[0]}|outside valid range"):
        summary.load_validated_matrix(input_dir, "development")


def test_wire_byte_total_and_simplex_shape_are_strict(tmp_path):
    input_dir = tmp_path / "development"
    write_matrix(input_dir, "development")
    path = next(input_dir.glob("*routed*.json"))
    rewrite(path, lambda record: record.update({
        "total_client_to_server_payload_bytes": 1}))
    with pytest.raises(summary.SummaryValidationError, match="wire bytes"):
        summary.load_validated_matrix(input_dir, "development")
