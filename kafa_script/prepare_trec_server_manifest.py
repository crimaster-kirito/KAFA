"""Freeze the TREC-Coarse development partitions without reading test labels."""

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

from kafa_script.classification_task import load_task_spec


COARSE_LABELS = ("ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM")
EXPECTED_TRAIN_SHA256 = (
    "9e4c8bdcaffb96ed61041bd64b564183d52793a8e91d84fc3a8646885f466ec3"
)
EXPECTED_TEST_SHA256 = (
    "033f22c028c2bbba9ca682f68ffe204dc1aa6e1cf35dd6207f2d4ca67f0d0e8e"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count_without_parsing(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def load_train_rows(path: Path) -> list[dict]:
    rows = []
    seen_source_ids = set()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            content = raw_line.rstrip(b"\r\n")
            try:
                label_token, question = content.split(b" ", 1)
                coarse = label_token.split(b":", 1)[0].decode("ascii")
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError(f"malformed TREC train line {line_number}") from error
            if coarse not in COARSE_LABELS:
                raise ValueError(f"unknown coarse label {coarse!r} on line {line_number}")
            source_id = hashlib.sha256(raw_line).hexdigest()
            # The source contains exact duplicate lines.  Treating identical
            # questions as separate records could leak them across partitions.
            if source_id in seen_source_ids:
                continue
            seen_source_ids.add(source_id)
            rows.append({
                "source_id": source_id,
                "line": line_number,
                "label": COARSE_LABELS.index(coarse),
                "coarse_label": coarse,
                "text": question.decode("latin-1"),
            })
    return rows


def allocate(rows: list[dict], namespace: str,
             split_quotas: list[tuple[str, int]]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        rank = hashlib.sha256(
            f"{namespace}|{row['source_id']}".encode("ascii")
        ).hexdigest()
        grouped[row["label"]].append((rank, row["source_id"], row))
    required = sum(quota for _, quota in split_quotas)
    for label in range(len(COARSE_LABELS)):
        grouped[label].sort(key=lambda item: (item[0], item[1]))
        if len(grouped[label]) < required:
            raise ValueError(
                f"class {COARSE_LABELS[label]} has {len(grouped[label])}; "
                f"{required} required"
            )
    splits = {name: [] for name, _ in split_quotas}
    cursors = [0] * len(COARSE_LABELS)
    for name, quota in split_quotas:
        for label in range(len(COARSE_LABELS)):
            begin = cursors[label]
            selected = grouped[label][begin:begin + quota]
            splits[name].extend(item[2] for item in selected)
            cursors[label] += quota
        splits[name].sort(key=lambda row: row["source_id"])
    all_ids = [row["source_id"] for rows_ in splits.values() for row in rows_]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("TREC development partitions overlap")
    return splits


def split_summary(rows: list[dict]) -> dict:
    records = [{key: row[key] for key in
                ("source_id", "line", "label", "coarse_label")}
               for row in rows]
    return {
        "count": len(records),
        "class_counts": dict(sorted(Counter(
            row["coarse_label"] for row in rows).items())),
        "source_ids_sha256": hashlib.sha256(
            "\n".join(row["source_id"] for row in rows).encode("ascii")
        ).hexdigest(),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-spec", type=Path, default=Path(
        "kafa_script/task_specs/trec_coarse_server_v1.json"))
    parser.add_argument("--train", type=Path, default=Path(
        ".cache/datasets/trec/train_5500.label"))
    parser.add_argument("--official-test", type=Path, default=Path(
        ".cache/datasets/trec/TREC_10.label"))
    parser.add_argument("--off-task-train", type=Path, default=Path(
        ".cache/datasets/imdb/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path(
        "kafa_script/trec_server_manifest_v1.json"))
    args = parser.parse_args()

    spec = load_task_spec(args.task_spec)
    if spec.num_classes != len(COARSE_LABELS):
        raise ValueError("TREC task spec must define exactly six coarse classes")
    train_sha256 = sha256_file(args.train)
    test_sha256 = sha256_file(args.official_test)
    if train_sha256 != EXPECTED_TRAIN_SHA256:
        raise ValueError("TREC train file hash differs from the frozen source")
    if test_sha256 != EXPECTED_TEST_SHA256:
        raise ValueError("TREC official-test hash differs from the frozen source")
    raw_train_count = line_count_without_parsing(args.train)
    rows = load_train_rows(args.train)
    splits = allocate(rows, "kafa-trec-coarse-v1", [
        ("anchor", 12),
        ("development", 12),
        ("client_0", 20),
        ("client_1", 20),
        ("client_2", 20),
    ])
    used_ids = {row["source_id"] for values in splits.values() for row in values}
    manifest = {
        "schema_version": "1.0",
        "created_utc": "2026-07-15",
        "task_id": spec.task_id,
        "task_spec_sha256": spec.sha256,
        "source": {
            "train_url": "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label",
            "official_test_url": "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label",
            "license": "unknown",
            "encoding": "latin-1",
            "record_id": "sha256(original line bytes including line terminator)",
        },
        "datasets": {
            "target_train": {
                "path_hint": str(args.train),
                "sha256": train_sha256,
                "raw_line_count": raw_train_count,
                "unique_record_count": len(rows),
                "exact_duplicate_lines_removed": raw_train_count - len(rows),
            },
            "target_official_test": {
                "path_hint": str(args.official_test),
                "sha256": test_sha256,
                "count": line_count_without_parsing(args.official_test),
                "inspection_policy": (
                    "manifest generation hashes and counts raw lines only; "
                    "it does not parse test labels or questions"
                ),
            },
            "off_task_train": {
                "path_hint": str(args.off_task_train),
                "sha256": sha256_file(args.off_task_train),
                "count": line_count_without_parsing(args.off_task_train),
                "dataset": "IMDb train",
            },
        },
        "selection": {
            "namespace": "kafa-trec-coarse-v1",
            "ranking": "sha256(namespace + '|' + source_id), ascending per class",
            "allocation_order": [
                "anchor", "development", "client_0", "client_1", "client_2"
            ],
            "per_class": {
                "anchor": 12,
                "development": 12,
                "client_0": 20,
                "client_1": 20,
                "client_2": 20,
            },
            "all_partitions_disjoint": True,
            "exact_duplicate_source_lines_removed_before_ranking": True,
            "unused_train_count": len(rows) - len(used_ids),
        },
        "splits": {name: split_summary(values)
                   for name, values in splits.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "task_spec_sha256": spec.sha256,
        "train_sha256": train_sha256,
        "unique_train_records": len(rows),
        "exact_duplicates_removed": raw_train_count - len(rows),
        "official_test_sha256": test_sha256,
        "split_counts": {name: len(values) for name, values in splits.items()},
        "unused_train_count": len(rows) - len(used_ids),
    }, indent=2))


if __name__ == "__main__":
    main()
