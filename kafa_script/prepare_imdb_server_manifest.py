"""Freeze IMDb server-task partitions without parsing the official test."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.classification_task import canonical_sha256, load_task_spec  # noqa: E402
from kafa_script.run_generic_classification_distillation import sha256_file  # noqa: E402


EXPECTED_TRAIN_SHA256 = (
    "81f3a4c4b142255135285655e4a19c1d973b8d236f809bfb15d2cdd82c6b34ef"
)
EXPECTED_TEST_SHA256 = (
    "bceb588619d37a5b1c5e6ec7ec519b204eaaf8a4024735b279fe1eff33a46843"
)
EXPECTED_AGNEWS_SHA256 = (
    "4fe547d3954adbccc000719839572ddd158c57b718e3b9bb8f3fbdffc01592ba"
)
NAMESPACE = "kafa-imdb-role-reversal-server-v1"


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def load_train_rows(path: Path) -> tuple[list[dict], int]:
    rows, seen = [], set()
    raw_count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_count += 1
            raw = json.loads(line)
            source_id = canonical_sha256(raw)
            if source_id in seen:
                continue
            seen.add(source_id)
            rows.append({
                "source_id": source_id,
                "line": line_number,
                "label": int(raw["label"]),
            })
    return rows, raw_count


def allocate(rows: list[dict], quotas: list[tuple[str, int]]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        rank = hashlib.sha256(
            f"{NAMESPACE}|{row['source_id']}".encode("ascii")).hexdigest()
        grouped[row["label"]].append((rank, row["source_id"], row))
    labels = sorted(grouped)
    if labels != [0, 1]:
        raise ValueError(f"IMDb train labels differ from [0, 1]: {labels}")
    required = sum(quota for _, quota in quotas)
    for label in labels:
        grouped[label].sort(key=lambda item: (item[0], item[1]))
        if len(grouped[label]) < required:
            raise ValueError(f"class {label} has insufficient records")
    splits, cursors = {name: [] for name, _ in quotas}, {0: 0, 1: 0}
    for name, quota in quotas:
        for label in labels:
            begin = cursors[label]
            splits[name].extend(item[2] for item in
                                grouped[label][begin:begin + quota])
            cursors[label] += quota
        splits[name].sort(key=lambda row: row["source_id"])
    all_ids = [row["source_id"] for values in splits.values() for row in values]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("IMDb manifest partitions overlap")
    return splits


def summarize(rows: list[dict]) -> dict:
    records = [{key: row[key] for key in ("source_id", "line", "label")}
               for row in rows]
    return {
        "count": len(records),
        "class_counts": dict(sorted(Counter(
            str(row["label"]) for row in rows).items())),
        "source_ids_sha256": hashlib.sha256(
            "\n".join(record["source_id"] for record in records).encode(
                "ascii")).hexdigest(),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-spec", type=Path, default=Path(
        "kafa_script/task_specs/imdb_sentiment_server_v1.json"))
    parser.add_argument("--train", type=Path, default=Path(
        ".cache/datasets/imdb/train.jsonl"))
    parser.add_argument("--official-test", type=Path, default=Path(
        ".cache/datasets/imdb/test.jsonl"))
    parser.add_argument("--off-task-train", type=Path, default=Path(
        ".cache/datasets/agnews/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path(
        "kafa_script/imdb_server_manifest_v1.json"))
    args = parser.parse_args()

    spec = load_task_spec(args.task_spec)
    if spec.num_classes != 2:
        raise ValueError("IMDb server task must define two classes")
    if sha256_file(args.train) != EXPECTED_TRAIN_SHA256:
        raise ValueError("IMDb train hash differs from frozen source")
    # Hash and count raw test bytes only. Do not decode JSON or inspect labels.
    if sha256_file(args.official_test) != EXPECTED_TEST_SHA256:
        raise ValueError("IMDb official-test hash differs from frozen source")
    if sha256_file(args.off_task_train) != EXPECTED_AGNEWS_SHA256:
        raise ValueError("AG News off-task source hash differs")
    rows, raw_count = load_train_rows(args.train)
    splits = allocate(rows, [
        ("anchor", 64),
        ("development", 500),
        ("client_0", 128),
        ("client_1", 128),
        ("client_2", 128),
    ])
    used = {row["source_id"] for values in splits.values() for row in values}
    manifest = {
        "schema_version": "1.0",
        "name": "imdb_server_manifest_v1",
        "created_utc": "2026-07-15",
        "task_id": spec.task_id,
        "task_spec_semantic_sha256": spec.sha256,
        "evidence_boundary": (
            "The complete IMDb test had no model-evaluation record in the "
            "workspace audit before this manifest. IMDb train and the sentiment "
            "interface had previously served as off-task inputs, so this is a "
            "prospective untouched-test role-reversal extension, not a dataset-"
            "naive or independently preregistered replication."
        ),
        "source": {
            "dataset": "IMDb sentiment JSONL cache",
            "license": "unknown in the local artifact",
            "record_id": "sha256(canonical JSON record)",
        },
        "datasets": {
            "target_train": {
                "path_hint": str(args.train),
                "sha256": EXPECTED_TRAIN_SHA256,
                "raw_count": raw_count,
                "unique_count": len(rows),
                "exact_duplicates_removed": raw_count - len(rows),
            },
            "target_official_test": {
                "path_hint": str(args.official_test),
                "sha256": EXPECTED_TEST_SHA256,
                "count": line_count(args.official_test),
                "inspection_policy": (
                    "Manifest generation hashes and counts raw lines only; it "
                    "does not decode test JSON or inspect labels/text."
                ),
            },
            "off_task_train": {
                "path_hint": str(args.off_task_train),
                "sha256": EXPECTED_AGNEWS_SHA256,
                "count": line_count(args.off_task_train),
                "dataset": "AG News train",
            },
        },
        "selection": {
            "namespace": NAMESPACE,
            "ranking": "sha256(namespace + '|' + source_id), ascending per class",
            "allocation_order": [
                "anchor", "development", "client_0", "client_1", "client_2"
            ],
            "per_class": {
                "anchor": 64,
                "development": 500,
                "client_0": 128,
                "client_1": 128,
                "client_2": 128,
            },
            "all_partitions_disjoint": True,
            "unused_unique_train_count": len(rows) - len(used),
        },
        "splits": {name: summarize(values) for name, values in splits.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "task_spec_semantic_sha256": spec.sha256,
        "train_unique": len(rows),
        "test_raw_count_without_parsing": manifest["datasets"][
            "target_official_test"]["count"],
        "split_counts": {name: len(values) for name, values in splits.items()},
        "unused_unique_train_count": len(rows) - len(used),
    }, indent=2))


if __name__ == "__main__":
    main()
