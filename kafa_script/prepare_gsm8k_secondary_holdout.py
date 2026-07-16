#!/usr/bin/env python3
"""Freeze an untouched GSM8K training-split holdout for secondary analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.run_gsm8k_local_dev import load_jsonl


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-manifest", type=Path,
                        default=Path("kafa_script/gsm8k_benchmark_manifest_v1.json"))
    parser.add_argument("--train-data", type=Path,
                        default=Path(".cache/datasets/gsm8k/processed/train.jsonl"))
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output", type=Path,
                        default=Path("kafa_script/gsm8k_secondary_holdout_manifest_v1.json"))
    args = parser.parse_args()

    parent = json.loads(args.parent_manifest.read_text())
    rows = load_jsonl(args.train_data)
    available = {row["source_index"] for row in rows}
    partitions = parent["partitions"]
    used = set(partitions["anchor"]) | set(partitions["development"])
    for pool in partitions["client_pools"]:
        used.update(pool)
    candidates = sorted(available - used)
    if len(candidates) < args.size:
        raise ValueError("not enough untouched source rows")
    random.Random(args.seed).shuffle(candidates)
    selected = candidates[:args.size]
    if set(selected) & used or len(set(selected)) != args.size:
        raise AssertionError("secondary holdout overlap/uniqueness invariant failed")

    manifest = {
        "name": "gsm8k_secondary_holdout_manifest_v1",
        "status": "frozen_before_intermediate_severity_runs",
        "source_split": "openai/gsm8k main train",
        "parent_manifest": str(args.parent_manifest),
        "parent_manifest_sha256": sha256(args.parent_manifest),
        "processed_train_sha256": sha256(args.train_data),
        "selection_seed": args.seed,
        "count": args.size,
        "excluded_counts": {
            "anchor": len(partitions["anchor"]),
            "development": len(partitions["development"]),
            "client_pools_total": sum(len(pool) for pool in partitions["client_pools"]),
        },
        "source_indices": selected,
        "source_indices_sha256": hashlib.sha256("\n".join(map(str, selected)).encode()).hexdigest(),
        "claim_boundary": "Secondary training-split holdout; never used for method selection or client/anchor training, but not the official GSM8K test split.",
        "paper_result": False,
    }
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output} with {len(selected)} disjoint examples")


if __name__ == "__main__":
    main()
