#!/usr/bin/env python3
"""Fail-fast verifier for the frozen TREC server-transfer evidence contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.classification_task import load_task_spec  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "kafa_script/trec_server_transfer_protocol_v1.json"
DEFAULT_DEVELOPMENT_OUTPUT = ROOT / "kafa_script/analysis_outputs/trec_server/development"
DEFAULT_OFFICIAL_OUTPUT = ROOT / "kafa_script/analysis_outputs/trec_server/official"
EXPECTED_COMPARATORS = ("public_anchor_only", "routed_distillation")
EXPECTED_SPLITS = {
    "anchor": (72, 12),
    "development": (72, 12),
    "client_0": (120, 20),
    "client_1": (120, 20),
    "client_2": (120, 20),
}


class VerificationError(ValueError):
    """Raised when frozen inputs or evidence artifacts violate the protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read JSON file {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    _require(path.is_file(), f"missing hashed file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, value: str, *, owner: str) -> Path:
    _require(isinstance(value, str) and value, f"{owner} path must be non-empty")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _validate_seeds(protocol: dict[str, Any]) -> None:
    development = protocol.get("development", {}).get("seeds")
    official = protocol.get("official_evaluation", {}).get("seeds")
    _require(isinstance(development, list) and isinstance(official, list),
             "development and official seeds must be lists")
    seeds = development + official
    _require(all(isinstance(seed, int) and not isinstance(seed, bool)
                 for seed in seeds), "all protocol seeds must be integers")
    _require(len(development) == 3, "development must declare exactly three seeds")
    _require(len(official) == 5, "official evaluation must declare exactly five seeds")
    _require(len(seeds) == len(set(seeds)),
             "development and official seeds must be globally unique")


def _validate_comparators(protocol: dict[str, Any]) -> None:
    for phase in ("development", "official_evaluation"):
        comparators = protocol.get(phase, {}).get("comparators")
        _require(comparators == list(EXPECTED_COMPARATORS),
                 f"{phase} comparators must be fixed as {list(EXPECTED_COMPARATORS)}")


def _validate_positive_configuration(configuration: Any) -> None:
    _require(isinstance(configuration, dict) and configuration,
             "fixed_configuration must be a non-empty object")
    numeric = {key: value for key, value in configuration.items()
               if isinstance(value, (int, float)) and not isinstance(value, bool)}
    _require(numeric, "fixed_configuration must contain numeric settings")
    for key, value in numeric.items():
        _require(value > 0, f"fixed_configuration.{key} must be positive")


def _validate_manifest(manifest: dict[str, Any], spec: Any) -> None:
    _require(manifest.get("task_id") == spec.task_id,
             "manifest task_id differs from task spec")
    _require(manifest.get("task_spec_sha256") == spec.sha256,
             "manifest task semantic hash differs from task spec")
    _require(manifest.get("selection", {}).get("all_partitions_disjoint") is True,
             "manifest must declare all partitions disjoint")

    splits = manifest.get("splits")
    _require(isinstance(splits, dict) and set(splits) == set(EXPECTED_SPLITS),
             "manifest split names differ from the frozen five-split design")
    owners: dict[str, str] = {}
    for split_name, (expected_total, expected_per_class) in EXPECTED_SPLITS.items():
        split = splits[split_name]
        records = split.get("records")
        _require(isinstance(records, list), f"{split_name}.records must be a list")
        _require(split.get("count") == expected_total == len(records),
                 f"{split_name} must contain exactly {expected_total} records")
        counts = Counter(record.get("label") for record in records)
        _require(counts == Counter({label: expected_per_class
                                   for label in range(spec.num_classes)}),
                 f"{split_name} is not balanced at {expected_per_class} per class")
        declared_counts = split.get("class_counts")
        observed_coarse = Counter(record.get("coarse_label") for record in records)
        _require(declared_counts == dict(sorted(observed_coarse.items())),
                 f"{split_name} class_counts do not match records")
        ids = [record.get("source_id") for record in records]
        _require(all(isinstance(record_id, str) and len(record_id) == 64
                     for record_id in ids), f"{split_name} has malformed source IDs")
        _require(split.get("source_ids_sha256") == hashlib.sha256(
            "\n".join(ids).encode("ascii")).hexdigest(),
            f"{split_name} source_ids_sha256 mismatch")
        for record_id in ids:
            _require(record_id not in owners,
                     f"source {record_id} overlaps {owners.get(record_id)} and {split_name}")
            owners[record_id] = split_name
    _require(len(owners) == 504,
             f"frozen partitions must contain 504 globally unique records, got {len(owners)}")


def _validate_datasets(root: Path, protocol: dict[str, Any],
                       manifest: dict[str, Any]) -> None:
    task = protocol["task"]
    key_map = {
        "target_train": "target_train_sha256",
        "target_official_test": "target_official_test_sha256",
        "off_task_train": "off_task_train_sha256",
    }
    datasets = manifest.get("datasets")
    _require(isinstance(datasets, dict) and set(key_map) <= set(datasets),
             "manifest is missing a frozen dataset entry")
    for dataset_name, protocol_key in key_map.items():
        dataset = datasets[dataset_name]
        declared = dataset.get("sha256")
        _require(task.get(protocol_key) == declared,
                 f"protocol {protocol_key} differs from manifest")
        source = _resolve(root, dataset.get("path_hint"),
                          owner=f"manifest datasets.{dataset_name}")
        _require(sha256_file(source) == declared,
                 f"raw hash mismatch for {dataset_name}: {source}")


def _json_records(directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    if not directory.exists():
        return []
    _require(directory.is_dir(), f"output path is not a directory: {directory}")
    records = []
    for path in sorted(directory.rglob("*.json")):
        records.append((path, _read_json(path)))
    return records


def _is_official(record: dict[str, Any]) -> bool:
    markers = (record.get("evidence_label"), record.get("evidence_phase"),
               record.get("evaluation_split"))
    return any(isinstance(marker, str) and
               ("official" in marker.lower() or "holdout" in marker.lower())
               for marker in markers)


def _validate_outputs(protocol: dict[str, Any], protocol_hash: str,
                      development_output: Path, official_output: Path) -> None:
    development_records = _json_records(development_output)
    for path, record in development_records:
        _require(not _is_official(record),
                 f"development output contains official evidence: {path}")

    official_records = _json_records(official_output)
    if not official_output.exists():
        return
    official_seeds = protocol["official_evaluation"]["seeds"]
    expected = {(seed, comparator) for seed in official_seeds
                for comparator in EXPECTED_COMPARATORS}
    observed = []
    for path, record in official_records:
        _require(_is_official(record), f"official output lacks official evidence marker: {path}")
        _require(record.get("protocol_sha256") == protocol_hash,
                 f"official output protocol hash mismatch: {path}")
        comparator = record.get("comparator", record.get("method"))
        pair = (record.get("seed"), comparator)
        _require(pair in expected, f"unexpected official seed/comparator in {path}: {pair}")
        observed.append(pair)
    _require(len(observed) == 10, f"official output must contain exactly 10 records, got {len(observed)}")
    _require(len(set(observed)) == len(observed),
             "official output contains duplicate seed/comparator records")
    _require(set(observed) == expected, "official output does not cover the frozen 5x2 matrix")


def verify(protocol_path: Path = DEFAULT_PROTOCOL, *, root: Path = ROOT,
           development_output: Path = DEFAULT_DEVELOPMENT_OUTPUT,
           official_output: Path = DEFAULT_OFFICIAL_OUTPUT) -> dict[str, Any]:
    """Verify frozen inputs and any output matrices; return a compact audit summary."""
    protocol_path = Path(protocol_path)
    root = Path(root)
    protocol = _read_json(protocol_path)
    protocol_hash = sha256_file(protocol_path)
    implementation = protocol.get("implementation")
    _require(isinstance(implementation, dict),
             "protocol.implementation must be an object")
    runner_path = _resolve(root, implementation.get("runner"),
                           owner="protocol implementation runner")
    core_path = _resolve(root, implementation.get("classification_task_core"),
                         owner="protocol classification task core")
    _require(sha256_file(runner_path) == implementation.get("runner_file_sha256"),
             "protocol runner file hash mismatch")
    _require(sha256_file(core_path) ==
             implementation.get("classification_task_file_sha256"),
             "protocol classification-task core hash mismatch")
    task = protocol.get("task")
    _require(isinstance(task, dict), "protocol.task must be an object")
    spec_path = _resolve(root, task.get("task_spec"), owner="protocol task_spec")
    off_task_spec_path = _resolve(
        root, task.get("off_task_spec"), owner="protocol off_task_spec")
    manifest_path = _resolve(root, task.get("manifest"), owner="protocol manifest")
    spec = load_task_spec(spec_path)
    off_task_spec = load_task_spec(off_task_spec_path)
    manifest = _read_json(manifest_path)

    _require(task.get("task_id") == spec.task_id,
             "protocol task_id differs from task spec")
    _require(task.get("task_spec_semantic_sha256") == spec.sha256,
             "protocol task semantic hash mismatch")
    _require(task.get("off_task_spec_semantic_sha256") == off_task_spec.sha256,
             "protocol off-task semantic hash mismatch")
    _require(task.get("manifest_file_sha256") == sha256_file(manifest_path),
             "protocol manifest raw-file hash mismatch")
    _validate_manifest(manifest, spec)
    _validate_datasets(root, protocol, manifest)
    _validate_seeds(protocol)
    _validate_comparators(protocol)
    _validate_positive_configuration(protocol.get("fixed_configuration"))
    _require(protocol["fixed_configuration"].get("conflict_permutation") ==
             list(spec.conflict_permutation),
             "protocol conflict permutation differs from task spec")
    _validate_outputs(protocol, protocol_hash, Path(development_output),
                      Path(official_output))
    return {"protocol_sha256": protocol_hash, "task_semantic_sha256": spec.sha256,
            "manifest_sha256": sha256_file(manifest_path), "unique_records": 504}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--development-output", type=Path,
                        default=DEFAULT_DEVELOPMENT_OUTPUT)
    parser.add_argument("--official-output", type=Path,
                        default=DEFAULT_OFFICIAL_OUTPUT)
    args = parser.parse_args()
    summary = verify(args.protocol, root=args.root,
                     development_output=args.development_output,
                     official_output=args.official_output)
    print("Verified frozen TREC server protocol: " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
