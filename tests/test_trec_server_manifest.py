import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from kafa_script import prepare_trec_server_manifest as trec_manifest
from kafa_script.classification_task import load_task_spec


ROOT = Path(__file__).resolve().parents[1]
FROZEN_MANIFEST = ROOT / "kafa_script/trec_server_manifest_v1.json"
FROZEN_TASK_SPEC = ROOT / "kafa_script/task_specs/trec_coarse_server_v1.json"


def _task_mapping():
    return {
        "schema_version": "1.0",
        "task_id": "trec_coarse_server_v1",
        "prompt_template": (
            "Classify what kind of answer the question asks for: "
            "abbreviation, entity, description, human, location, or number.\n"
            "Question: {text}\nAnswer type:"
        ),
        "verbalizers": [
            "abbreviation", "entity", "description",
            "human", "location", "number",
        ],
        "conflict_permutation": [1, 2, 3, 4, 5, 0],
        "text_field": "text",
        "label_field": "label",
    }


def _write_synthetic_train(path: Path, *, per_class: int = 84,
                           duplicate_first: bool = False) -> bytes:
    lines = []
    for coarse in trec_manifest.COARSE_LABELS:
        lines.extend(
            f"{coarse}:fixture What is {coarse} example {index}?\n".encode()
            for index in range(per_class)
        )
    if duplicate_first:
        lines.insert(1, lines[0])
    payload = b"".join(lines)
    path.write_bytes(payload)
    return payload


def _all_record_ids(manifest):
    return {
        name: {record["source_id"] for record in split["records"]}
        for name, split in manifest["splits"].items()
    }


def test_load_train_rows_removes_exact_duplicate_source_lines(tmp_path):
    train = tmp_path / "train.label"
    first = b"ABBR:def What does CPU stand for?\n"
    second = b"NUM:count How many moons does Mars have?\n"
    train.write_bytes(first + first + second)

    rows = trec_manifest.load_train_rows(train)

    assert len(rows) == 2
    assert [row["line"] for row in rows] == [1, 3]
    assert rows[0]["source_id"] == hashlib.sha256(first).hexdigest()
    assert rows[0]["text"] == "What does CPU stand for?"
    assert rows[0]["coarse_label"] == "ABBR"


def test_allocate_is_six_class_balanced_and_globally_disjoint(tmp_path):
    train = tmp_path / "train.label"
    _write_synthetic_train(train, duplicate_first=True)
    rows = trec_manifest.load_train_rows(train)
    quotas = [
        ("anchor", 12),
        ("development", 12),
        ("client_0", 20),
        ("client_1", 20),
        ("client_2", 20),
    ]

    splits = trec_manifest.allocate(rows, "fixture-namespace", quotas)

    expected_totals = {
        "anchor": 72,
        "development": 72,
        "client_0": 120,
        "client_1": 120,
        "client_2": 120,
    }
    all_ids = []
    for name, per_class in quotas:
        assert len(splits[name]) == expected_totals[name]
        assert Counter(row["coarse_label"] for row in splits[name]) == {
            label: per_class for label in trec_manifest.COARSE_LABELS
        }
        all_ids.extend(row["source_id"] for row in splits[name])
    assert len(all_ids) == 504
    assert len(set(all_ids)) == 504


def test_frozen_manifest_has_expected_hashes_balance_and_zero_overlap():
    manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    spec = load_task_spec(FROZEN_TASK_SPEC)

    assert manifest["task_id"] == spec.task_id
    assert manifest["task_spec_sha256"] == spec.sha256
    assert manifest["datasets"]["target_train"]["sha256"] == (
        trec_manifest.EXPECTED_TRAIN_SHA256
    )
    assert manifest["datasets"]["target_official_test"]["sha256"] == (
        trec_manifest.EXPECTED_TEST_SHA256
    )
    assert manifest["datasets"]["target_official_test"]["count"] == 500
    for dataset in manifest["datasets"].values():
        source_path = ROOT / dataset["path_hint"]
        assert source_path.is_file()
        assert trec_manifest.sha256_file(source_path) == dataset["sha256"]
        declared_count = dataset.get("raw_line_count", dataset.get("count"))
        assert trec_manifest.line_count_without_parsing(source_path) == declared_count

    expected_per_class = {
        "anchor": 12,
        "development": 12,
        "client_0": 20,
        "client_1": 20,
        "client_2": 20,
    }
    expected_totals = {
        "anchor": 72,
        "development": 72,
        "client_0": 120,
        "client_1": 120,
        "client_2": 120,
    }
    owner = {}
    for name, split in manifest["splits"].items():
        assert split["count"] == expected_totals[name]
        assert split["class_counts"] == {
            label: expected_per_class[name]
            for label in sorted(trec_manifest.COARSE_LABELS)
        }
        record_ids = [record["source_id"] for record in split["records"]]
        assert len(record_ids) == len(set(record_ids))
        assert split["source_ids_sha256"] == hashlib.sha256(
            "\n".join(record_ids).encode("ascii")
        ).hexdigest()
        for record_id in record_ids:
            assert record_id not in owner, (
                f"{record_id} occurs in both {owner.get(record_id)} and {name}"
            )
            owner[record_id] = name
    assert len(owner) == 504


def test_main_hashes_sources_and_never_parses_official_test(
        tmp_path, monkeypatch, capsys):
    task_spec = tmp_path / "task.json"
    # Reordered, compact JSON proves that the recorded task hash is semantic,
    # not a hash of the task-spec file's formatting.
    mapping = _task_mapping()
    task_spec.write_text(json.dumps(dict(reversed(list(mapping.items())))),
                         encoding="utf-8")
    train = tmp_path / "train.label"
    train_payload = _write_synthetic_train(train, duplicate_first=True)
    official_test = tmp_path / "official-test.label"
    # Deliberately malformed as TREC records. Manifest generation must still
    # succeed because final-test contents are only hashed and line-counted.
    test_payload = b"not-a-labelled-question\n\xff\x00\n"
    official_test.write_bytes(test_payload)
    off_task = tmp_path / "off-task.jsonl"
    off_task_payload = b'{"text":"a","label":0}\n{"text":"b","label":1}\n'
    off_task.write_bytes(off_task_payload)
    output = tmp_path / "manifest.json"

    monkeypatch.setattr(
        trec_manifest, "EXPECTED_TRAIN_SHA256",
        hashlib.sha256(train_payload).hexdigest())
    monkeypatch.setattr(
        trec_manifest, "EXPECTED_TEST_SHA256",
        hashlib.sha256(test_payload).hexdigest())
    parsed_paths = []
    original_load_train_rows = trec_manifest.load_train_rows

    def tracking_train_parser(path):
        parsed_paths.append(Path(path))
        assert Path(path) != official_test
        return original_load_train_rows(path)

    monkeypatch.setattr(trec_manifest, "load_train_rows",
                        tracking_train_parser)
    monkeypatch.setattr(sys, "argv", [
        "prepare_trec_server_manifest.py",
        "--task-spec", str(task_spec),
        "--train", str(train),
        "--official-test", str(official_test),
        "--off-task-train", str(off_task),
        "--output", str(output),
    ])

    trec_manifest.main()
    capsys.readouterr()
    manifest = json.loads(output.read_text(encoding="utf-8"))

    assert parsed_paths == [train]
    assert manifest["task_spec_sha256"] == load_task_spec(mapping).sha256
    assert manifest["task_spec_sha256"] != hashlib.sha256(
        task_spec.read_bytes()).hexdigest()
    assert manifest["datasets"]["target_train"] == {
        "path_hint": str(train),
        "sha256": hashlib.sha256(train_payload).hexdigest(),
        "raw_line_count": 505,
        "unique_record_count": 504,
        "exact_duplicate_lines_removed": 1,
    }
    assert manifest["datasets"]["target_official_test"]["sha256"] == (
        hashlib.sha256(test_payload).hexdigest()
    )
    assert manifest["datasets"]["target_official_test"]["count"] == 2
    assert manifest["datasets"]["off_task_train"]["sha256"] == (
        hashlib.sha256(off_task_payload).hexdigest()
    )
    assert manifest["datasets"]["off_task_train"]["count"] == 2

    ids_by_split = _all_record_ids(manifest)
    assert sum(map(len, ids_by_split.values())) == 504
    assert len(set().union(*ids_by_split.values())) == 504
