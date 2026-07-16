import hashlib
import inspect
import math
from collections import Counter
from pathlib import Path

import pytest
import torch

from kafa_script.classification_task import load_task_spec
from kafa_script import run_generic_classification_distillation as runner


def task_spec(classes):
    return load_task_spec({
        "schema_version": "1.0",
        "task_id": f"test-{classes}",
        "prompt_template": "Text: {text}\nLabel:",
        "verbalizers": [f"label-{index}" for index in range(classes)],
        "conflict_permutation": list(range(1, classes)) + [0],
    })


def imdb_spec():
    return load_task_spec({
        "schema_version": "1.0",
        "task_id": "imdb-test",
        "prompt_template": "Review: {text}\nSentiment:",
        "verbalizers": ["Negative", "Positive"],
        "conflict_permutation": [1, 0],
    })


def test_development_evaluation_never_hashes_or_parses_official():
    development = [{"source_id": "dev", "text": "safe", "label": 0}]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("development touched official test")

    result = runner.load_evaluation_rows(
        "development", development, Path("must-not-be-opened"), "unused", 500,
        hash_file=forbidden, official_loader=forbidden)
    assert result is development


def test_official_hash_mismatch_fails_before_parser():
    parsed = False

    def parser(_path):
        nonlocal parsed
        parsed = True
        return []

    with pytest.raises(ValueError, match="hash"):
        runner.load_evaluation_rows(
            "official_test", [], Path("official"), "expected", 500,
            hash_file=lambda _path: "wrong", official_loader=parser)
    assert not parsed


def test_official_hash_and_count_are_checked_before_return():
    rows = [{"source_id": "official", "text": "q", "label": 0}]
    assert runner.load_evaluation_rows(
        "official_test", [], Path("official"), "expected", 1,
        hash_file=lambda _path: "expected",
        official_loader=lambda _path: rows) == rows
    with pytest.raises(ValueError, match="count"):
        runner.load_evaluation_rows(
            "official_test", [], Path("official"), "expected", 2,
            hash_file=lambda _path: "expected",
            official_loader=lambda _path: rows)


def test_fixed_configuration_is_exact_and_fails_closed():
    frozen = {"rounds": 2, "batch_size": 8}
    runner.validate_fixed_configuration(dict(frozen), frozen)
    with pytest.raises(ValueError, match="rounds"):
        runner.validate_fixed_configuration(
            {"rounds": 3, "batch_size": 8}, frozen)
    with pytest.raises(ValueError, match="extra"):
        runner.validate_fixed_configuration(
            {"rounds": 2, "batch_size": 8, "extra": True}, frozen)
    with pytest.raises(ValueError, match="batch_size"):
        runner.validate_fixed_configuration({"rounds": 2}, frozen)


def _manifest_split(records):
    source_ids = [record["source_id"] for record in records]
    return {
        "count": len(records),
        "source_ids_sha256": hashlib.sha256(
            "\n".join(source_ids).encode("ascii")).hexdigest(),
        "records": records,
    }


def test_trec_loader_reconnects_manifest_line_label_and_text(tmp_path):
    raw_lines = [
        b"ABBR:abb What does CPU stand for ?\n",
        b"NUM:count How many moons does Mars have ?\n",
    ]
    path = tmp_path / "train.label"
    path.write_bytes(b"".join(raw_lines))
    rows = runner.load_trec_train(path)
    records = [{
        "source_id": row["source_id"],
        "line": row["line"],
        "label": row["label"],
    } for row in rows]
    manifest = {"splits": {
        "anchor": _manifest_split([records[0]]),
        "development": _manifest_split([records[1]]),
    }}

    resolved = runner.resolve_manifest_splits(
        rows, manifest, ["anchor", "development"])
    assert resolved["anchor"][0]["text"] == "What does CPU stand for ?"
    assert resolved["development"][0]["label"] == 5

    manifest["splits"]["anchor"]["records"][0]["label"] = 1
    with pytest.raises(ValueError, match="label"):
        runner.select_manifest_rows(rows, manifest, "anchor")


def test_manifest_split_overlap_is_rejected(tmp_path):
    line = b"ABBR:abb What is an abbreviation ?\n"
    path = tmp_path / "train.label"
    path.write_bytes(line)
    row = runner.load_trec_train(path)[0]
    record = {"source_id": row["source_id"], "line": 1, "label": 0}
    manifest = {"splits": {
        "anchor": _manifest_split([record]),
        "development": _manifest_split([dict(record)]),
    }}
    with pytest.raises(ValueError, match="overlap"):
        runner.resolve_manifest_splits(
            [row], manifest, ["anchor", "development"])


@pytest.mark.parametrize("classes", [2, 6])
def test_mixture_uses_explicit_k_class_conflict_permutation(classes):
    target = task_spec(classes)
    off_task = imdb_spec()
    bases = [[{
        "source_id": f"target-{classes}-{index}",
        "text": f"target {index}",
        "label": index % classes,
    } for index in range(max(4, classes * 2))]]
    imdb = [{
        "source_id": f"imdb-{index}", "text": f"review {index}",
        "label": index % 2,
    } for index in range(20)]
    mixture = runner.build_client_mixtures(
        bases, imdb, target, off_task, seed=17,
        conflict_ratio=0.5, off_task_ratio=0.25)[0]

    conflicts = [item for item in mixture if item["kind"] == "conflict"]
    assert conflicts
    for item in conflicts:
        assert item["training_label"] == target.conflicting_label(
            item["original_label"])
        assert item["training_label"] != item["original_label"]
    off_items = [item for item in mixture if item["kind"] == "off_task"]
    assert off_items
    assert {item["answer"] for item in off_items} <= {"Negative", "Positive"}


def test_frozen_trec_three_client_mixture_is_balanced_disjoint_and_routable():
    """Lock the class-balanced contamination contract frozen for TREC."""
    target = task_spec(6)
    off_task = imdb_spec()
    client_bases = [[{
        "source_id": f"client-{client}-class-{label}-row-{index}",
        "text": f"client {client} class {label} row {index}",
        "label": label,
    } for label in range(6) for index in range(20)]
        for client in range(3)]
    off_task_rows = [{
        "source_id": f"imdb-{index}",
        "text": f"review {index}",
        "label": index % 2,
    } for index in range(100)]

    kwargs = {
        "target_spec": target,
        "off_task_spec": off_task,
        "seed": 241,
        "conflict_ratio": 0.4,
        "off_task_ratio": 0.2,
    }
    mixtures = runner.build_client_mixtures(
        client_bases, off_task_rows, **kwargs)
    repeated = runner.build_client_mixtures(
        client_bases, off_task_rows, **kwargs)

    assert mixtures == repeated
    assert len(mixtures) == 3
    all_off_task_ids = []
    routing_quota = round(120 * 0.5 / target.num_classes)
    assert routing_quota == 10
    for base, mixture in zip(client_bases, mixtures):
        assert len(mixture) == 120
        assert Counter(item["kind"] for item in mixture) == {
            "clean": 48, "conflict": 48, "off_task": 24,
        }

        by_original_class = {
            label: Counter(
                item["kind"] for row, item in zip(base, mixture)
                if row["label"] == label
            )
            for label in range(target.num_classes)
        }
        assert by_original_class == {
            label: Counter(clean=8, conflict=8, off_task=4)
            for label in range(target.num_classes)
        }

        claimed = Counter(
            item["training_label"] for item in mixture
            if item["kind"] != "off_task"
        )
        assert claimed == {label: 16 for label in range(target.num_classes)}
        assert all(claimed[label] >= routing_quota
                   for label in range(target.num_classes))
        for item in mixture:
            if item["kind"] == "conflict":
                assert item["training_label"] == target.conflicting_label(
                    item["original_label"])
        all_off_task_ids.extend(
            item["source_id"] for item in mixture
            if item["kind"] == "off_task"
        )

    assert len(all_off_task_ids) == 72
    assert len(set(all_off_task_ids)) == 72

    other_seed = runner.build_client_mixtures(
        client_bases, off_task_rows, target, off_task, seed=242,
        conflict_ratio=0.4, off_task_ratio=0.2)
    assert other_seed != mixtures


def test_known_loss_matrix_metrics_are_canonical_probability_metrics(monkeypatch):
    spec = task_spec(2)
    rows = [
        {"text": "first", "label": 0},
        {"text": "second", "label": 1},
    ]
    matrix = torch.tensor([[0.0, math.log(3)], [math.log(3), 0.0]])
    monkeypatch.setattr(runner, "candidate_loss_matrix",
                        lambda *_args, **_kwargs: matrix)

    class Model:
        def eval(self):
            return self

    metrics = runner.evaluate(Model(), None, rows, spec, "cpu", 32, 2)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["canonical_label_nll"] == pytest.approx(-math.log(0.75))
    assert metrics["target_answer_loss"] == pytest.approx(0.0)
    assert metrics["ece"] == pytest.approx(0.25)
    assert metrics["confusion_matrix"] == [[1, 0], [0, 1]]


def test_serialized_probability_payload_hashes_exact_wire_bytes():
    probabilities = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    first = runner.serialized_probability_payload(probabilities)
    second = runner.serialized_probability_payload(probabilities.clone())
    assert first == second
    assert first["encoding"] == "torch-save-tensor"
    assert first["payload_bytes"] > probabilities.numel() * 4
    assert len(first["payload_sha256"]) == 64


def test_schema_v2_source_contains_required_audit_fields():
    source = inspect.getsource(runner.main)
    required = {
        '"schema_version"', '"protocol_sha256"',
        '"task_spec_semantic_sha256"', '"task_spec_file_sha256"',
        '"manifest_sha256"', '"full_args"', '"full_args_sha256"',
        '"started_utc"', '"git"', '"datasets"', '"model_artifacts"',
        '"split_source_ids_sha256"', '"round_records"', '"final"',
    }
    assert not {field for field in required if field not in source}
