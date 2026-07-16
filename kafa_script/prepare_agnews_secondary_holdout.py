"""Create a frozen AG News train-derived evaluation holdout.

The split is selected without model outputs from records disjoint from the
existing anchor and development sets.  The post-primary ablation runner also
excludes these IDs from every client pool, preventing train/evaluation overlap.
"""

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

from kafa_script.prepare_benchmark_manifest import canonical_id, sha256_file


LABELS = (0, 1, 2, 3)
NAMESPACE = "kafa-agnews-post-primary-independent-holdout-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-manifest", type=Path, default=Path(
        "kafa_script/benchmark_manifest_v1.json"))
    parser.add_argument("--train-data", type=Path, default=Path(
        ".cache/datasets/agnews/train.jsonl"))
    parser.add_argument("--per-class", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path(
        "kafa_script/agnews_secondary_holdout_manifest_v1.json"))
    args = parser.parse_args()
    if args.per_class <= 0:
        parser.error("per-class must be positive")

    parent = json.loads(args.parent_manifest.read_text(encoding="utf-8"))
    if sha256_file(args.train_data) != parent["datasets"]["agnews_train"]["sha256"]:
        raise ValueError("AG News train hash differs from parent manifest")
    excluded = {
        record["id"]
        for split in ("anchor", "development")
        for record in parent["splits"][split]["records"]
    }
    candidates = defaultdict(list)
    with args.train_data.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            record_id = canonical_id(row)
            if record_id in excluded:
                continue
            label = int(row["label"])
            rank = hashlib.sha256(
                f"{NAMESPACE}|{record_id}".encode("ascii")).hexdigest()
            candidates[label].append((rank, {
                "id": record_id,
                "line": line_number,
                "label": label,
            }))

    selected = []
    for label in LABELS:
        ranked = sorted(candidates[label], key=lambda item: item[0])
        if len(ranked) < args.per_class:
            raise ValueError(f"class {label} has insufficient candidates")
        selected.extend(item[1] for item in ranked[:args.per_class])
    selected.sort(key=lambda record: record["id"])
    selected_ids = {record["id"] for record in selected}
    if selected_ids & excluded:
        raise AssertionError("secondary holdout overlaps parent reserved splits")

    manifest = {
        "name": "agnews_secondary_holdout_manifest_v1",
        "created_utc": "2026-07-15T13:30:00+00:00",
        "evidence_boundary": (
            "Selected after primary AG News results were observed. The split is "
            "excluded from every client allocation in the new runner, but some "
            "records may have appeared in earlier client training. It is a "
            "post-primary run-disjoint same-task resampling split, not globally "
            "untouched data, an independent replication, or a primary test."
        ),
        "selection_namespace": NAMESPACE,
        "namespace_label_note": (
            "The legacy word 'independent' in the deterministic namespace is "
            "not an evidence claim."
        ),
        "selection_rule": "sha256(namespace + '|' + canonical_record_id), ascending per class",
        "parent_manifest": str(args.parent_manifest),
        "parent_manifest_sha256": sha256_file(args.parent_manifest),
        "source": {
            "path": str(args.train_data),
            "sha256": sha256_file(args.train_data),
            "count": parent["datasets"]["agnews_train"]["count"],
        },
        "excluded_parent_splits": ["anchor", "development"],
        "excluded_parent_ids_sha256": hashlib.sha256(
            "\n".join(sorted(excluded)).encode("ascii")).hexdigest(),
        "holdout": {
            "count": len(selected),
            "class_counts": dict(sorted(Counter(
                record["label"] for record in selected).items())),
            "ids_sha256": hashlib.sha256(
                "\n".join(sorted(selected_ids)).encode("ascii")).hexdigest(),
            "records": selected,
        },
        "client_exclusion_requirement": (
            "Every ablation client pool must exclude all holdout IDs."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "records": len(selected),
        "class_counts": manifest["holdout"]["class_counts"],
        "ids_sha256": manifest["holdout"]["ids_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
