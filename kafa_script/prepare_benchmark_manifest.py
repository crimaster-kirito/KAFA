"""Freeze content-addressed KAFA benchmark splits without copying datasets."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


LABELS = (0, 1, 2, 3)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_id(row):
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_rows(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            rows.append({
                "id": canonical_id(row),
                "line": line_number,
                "label": int(row["label"]),
            })
    return rows


def take_balanced(rows, per_class, namespace, excluded=None):
    excluded = excluded or set()
    candidates = defaultdict(list)
    for row in rows:
        if row["id"] in excluded:
            continue
        key = hashlib.sha256(
            f"{namespace}|{row['id']}".encode("utf-8")).hexdigest()
        candidates[row["label"]].append((key, row))
    selected = []
    for label in LABELS:
        ranked = sorted(candidates[label], key=lambda item: item[0])
        if len(ranked) < per_class:
            raise ValueError(f"class {label} has fewer than {per_class} rows")
        selected.extend(item[1] for item in ranked[:per_class])
    return sorted(selected, key=lambda row: row["id"])


def summarize(rows):
    return {
        "count": len(rows),
        "class_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "records": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agnews-train", type=Path,
                        default=Path(".cache/datasets/agnews/train.jsonl"))
    parser.add_argument("--agnews-test", type=Path,
                        default=Path(".cache/datasets/agnews/test.jsonl"))
    parser.add_argument("--imdb-train", type=Path,
                        default=Path(".cache/datasets/imdb/train.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("kafa_script/benchmark_manifest_v1.json"))
    args = parser.parse_args()

    train_rows = load_rows(args.agnews_train)
    test_rows = load_rows(args.agnews_test)
    anchor = take_balanced(train_rows, 64, "kafa-v1-anchor")
    anchor_ids = {row["id"] for row in anchor}
    development = take_balanced(train_rows, 200, "kafa-v1-development",
                                excluded=anchor_ids)
    reserved = anchor_ids | {row["id"] for row in development}
    smoke_train = take_balanced(train_rows, 4, "kafa-v1-smoke-train",
                                excluded=reserved)
    smoke_test = take_balanced(test_rows, 2, "kafa-v1-smoke-test")
    client_pool_count = len(train_rows) - len(reserved)

    manifest = {
        "manifest_version": "1.0",
        "created_utc": "2026-07-14",
        "selection": {
            "id": "sha256(canonical JSON record)",
            "ranking": "sha256(namespace + '|' + id), ascending per class",
            "test_isolation": "official AG News test is never used for anchors, routing, hyperparameters, or early stopping",
        },
        "datasets": {
            "agnews_train": {
                "path_hint": str(args.agnews_train),
                "sha256": sha256_file(args.agnews_train),
                "count": len(train_rows),
            },
            "agnews_test": {
                "path_hint": str(args.agnews_test),
                "sha256": sha256_file(args.agnews_test),
                "count": len(test_rows),
            },
            "imdb_train": {
                "path_hint": str(args.imdb_train),
                "sha256": sha256_file(args.imdb_train),
                "count": sum(1 for _ in args.imdb_train.open(encoding="utf-8")),
            },
        },
        "splits": {
            "anchor": summarize(anchor),
            "development": summarize(development),
            "smoke_train": summarize(smoke_train),
            "smoke_test": summarize(smoke_test),
            "client_pool": {
                "count": client_pool_count,
                "definition": "AG News train excluding anchor and development IDs",
                "excluded_id_sha256": hashlib.sha256(
                    "\n".join(sorted(reserved)).encode("ascii")).hexdigest(),
            },
        },
        "models": {
            "server": {
                "repo": "Qwen/Qwen2.5-1.5B",
                "revision": "8faed761d45a263340a0528343f099c05c9a4323",
                "weight_sha256": "a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796",
            },
            "client_same_family": {
                "repo": "Qwen/Qwen2.5-0.5B",
                "revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
                "weight_sha256": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
            },
            "client_smol": {
                "repo": "HuggingFaceTB/SmolLM2-360M",
                "revision": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
                "weight_sha256": "7aaff6661428bed033abba9522bec81938678642cca3181fe752b6ca9e1e540f",
            },
            "client_llama": {
                "repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
                "weight_sha256": "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933",
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "anchor": len(anchor),
        "development": len(development),
        "client_pool": client_pool_count,
        "smoke_train": len(smoke_train),
        "smoke_test": len(smoke_test),
    }, indent=2))


if __name__ == "__main__":
    main()
