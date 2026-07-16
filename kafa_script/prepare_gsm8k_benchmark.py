#!/usr/bin/env python3
"""Create a deterministic GSM8K Stage-B data artifact and split manifest.

The official test questions are copied verbatim for reporting only.  All
anchors, development rows, and client pools are drawn without replacement from
the official `main` training split.  The canonical answer is the normalized
final answer after GSM8K's `####` delimiter; rationale text is retained for
provenance but not used to choose answers or routing thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd


FINAL = re.compile(r"####\s*(.+?)\s*$", re.DOTALL)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_answer(answer: str) -> str:
    match = FINAL.search(answer)
    if not match:
        raise ValueError(f"No GSM8K final-answer delimiter: {answer[-100:]!r}")
    value = match.group(1).strip().replace(",", "")
    # GSM8K answers are numeric; retain a minus sign and decimal point only.
    value = value.replace("$", "")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        raise ValueError(f"Non-numeric GSM8K final answer: {value!r}")
    return value


def rows_from_parquet(path: Path) -> list[dict[str, str]]:
    frame = pd.read_parquet(path)
    required = {"question", "answer"}
    if set(frame.columns) != required:
        raise ValueError(f"Unexpected GSM8K columns {frame.columns.tolist()}")
    rows = []
    for index, record in frame.iterrows():
        rationale = str(record["answer"])
        rows.append({
            "source_index": int(index),
            "question": str(record["question"]),
            "rationale": rationale,
            "final_answer": normalize_answer(rationale),
        })
    return rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=Path(".cache/datasets/gsm8k/main"))
    parser.add_argument("--output", type=Path,
                        default=Path(".cache/datasets/gsm8k/processed"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("kafa_script/gsm8k_benchmark_manifest_v1.json"))
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--anchor-size", type=int, default=256)
    parser.add_argument("--development-size", type=int, default=512)
    parser.add_argument("--client-size", type=int, default=256)
    parser.add_argument("--clients", type=int, default=3)
    args = parser.parse_args()
    train_path = args.source / "train-00000-of-00001.parquet"
    test_path = args.source / "test-00000-of-00001.parquet"
    train, test = rows_from_parquet(train_path), rows_from_parquet(test_path)
    required = args.anchor_size + args.development_size + args.client_size * args.clients
    if len(train) < required:
        raise ValueError(f"Need {required} training rows, have {len(train)}")
    indices = list(range(len(train)))
    random.Random(args.seed).shuffle(indices)
    cursor = 0
    anchor = indices[cursor:cursor + args.anchor_size]
    cursor += args.anchor_size
    development = indices[cursor:cursor + args.development_size]
    cursor += args.development_size
    client_pools = []
    for _ in range(args.clients):
        client_pools.append(indices[cursor:cursor + args.client_size])
        cursor += args.client_size
    selected = anchor + development + [item for pool in client_pools for item in pool]
    if len(set(selected)) != len(selected):
        raise AssertionError("Manifest partitions overlap")
    write_jsonl(args.output / "train.jsonl", train)
    write_jsonl(args.output / "test.jsonl", test)
    manifest = {
        "name": "gsm8k_stage_b_manifest_v1",
        "status": "frozen_before_stage_b_development",
        "dataset": "openai/gsm8k",
        "configuration": "main",
        "source_sha256": {"train_parquet": sha256(train_path), "test_parquet": sha256(test_path)},
        "processed_sha256": {"train_jsonl": sha256(args.output / "train.jsonl"), "test_jsonl": sha256(args.output / "test.jsonl")},
        "split_seed": args.seed,
        "counts": {"train": len(train), "official_test": len(test), "anchor": len(anchor), "development": len(development), "client_pool": args.client_size, "clients": args.clients},
        "partitions": {"anchor": anchor, "development": development, "client_pools": client_pools},
        "answer_protocol": {"source": "text after final `####` delimiter", "normalization": "strip whitespace, remove commas and dollar sign, retain signed decimal numeric string", "test_label_use": "reporting only"},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest), "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
