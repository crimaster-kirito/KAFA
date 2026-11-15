from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "gsm8k_train.jsonl"
DEFAULT_OUT = ROOT / "data" / "gsm8k_dirty_fair_seed42"
FINAL_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
CALC_RE = re.compile(r"<<([^<>]+)=(-?\d+(?:\.\d+)?)>>")


def read_gsm8k(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            item = json.loads(line)
            rows.append({
                "sample_id": idx,
                "instruction": item["question"],
                "input": None,
                "output": item["answer"].replace("####", "The answer is"),
                "category": "gsm8k",
            })
    return rows


def final_answer(output: str):
    match = re.search(r"(?:####|The answer is)\s*(-?\d+(?:\.\d+)?)", output)
    return match.group(1) if match else None


def replace_final(output: str, value: int):
    return re.sub(r"((?:####|The answer is)\s*)-?\d+(?:\.\d+)?",
                  lambda m: f"{m.group(1)}{value}",
                  output,
                  count=1)


def corrupt_answer(record, rng):
    clean = record["output"]
    old = final_answer(clean)
    old_value = int(float(old)) if old is not None else rng.randint(1, 100)
    bad_value = old_value + rng.choice([-41, -23, -13, -7, 11, 19, 37])
    dirty = dict(record)
    dirty["output"] = replace_final(clean, bad_value)
    dirty["_is_dirty"] = True
    dirty["_dirty_type"] = "answer_corruption"
    dirty["_clean_output"] = clean
    return dirty


def corrupt_reasoning(record, rng):
    clean = record["output"]
    matches = list(CALC_RE.finditer(clean))
    if not matches:
        return corrupt_answer(record, rng)
    target = rng.choice(matches)
    old_value = int(float(target.group(2)))
    bad_value = old_value + rng.choice([-10, -6, -3, 2, 5, 9])
    output = clean[:target.start(2)] + str(bad_value) + clean[target.end(2):]
    dirty = dict(record)
    dirty["output"] = replace_final(output, bad_value)
    dirty["_is_dirty"] = True
    dirty["_dirty_type"] = "reasoning_corruption"
    dirty["_clean_output"] = clean
    return dirty


def mismatch_answer(record, donor):
    dirty = dict(record)
    dirty["output"] = donor["output"]
    dirty["_is_dirty"] = True
    dirty["_dirty_type"] = "answer_mismatch"
    dirty["_clean_output"] = record["output"]
    dirty["_donor_sample_id"] = donor["sample_id"]
    return dirty


def corrupt(record, dirty_type, rng, donor_pool):
    if dirty_type == "mixed":
        dirty_type = rng.choice(
            ["answer_corruption", "reasoning_corruption", "answer_mismatch"])
    if dirty_type == "answer_corruption":
        return corrupt_answer(record, rng)
    if dirty_type == "reasoning_corruption":
        return corrupt_reasoning(record, rng)
    if dirty_type == "answer_mismatch":
        donors = [row for row in donor_pool if row["sample_id"] != record["sample_id"]]
        return mismatch_answer(record, rng.choice(donors))
    raise ValueError(f"Unknown dirty_type: {dirty_type}")


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--server-ratio", type=float, default=0.4)
    parser.add_argument("--client-ratio", type=float, default=0.4)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--dirty-ratio", type=float, default=0.10)
    parser.add_argument(
        "--dirty-type",
        choices=[
            "answer_corruption", "reasoning_corruption", "answer_mismatch",
            "mixed"
        ],
        default="answer_corruption")
    args = parser.parse_args()

    rows = read_gsm8k(args.source)
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    n = len(rows)
    server_n = int(n * args.server_ratio)
    client_n = int(n * args.client_ratio)
    server = rows[:server_n]
    client_pool = rows[server_n:server_n + client_n]
    test = rows[server_n + client_n:]

    clients = [[] for _ in range(args.clients)]
    for idx, row in enumerate(client_pool):
        clients[idx % args.clients].append(dict(row))

    materialized_clients = []
    for client_idx, client_rows in enumerate(clients, start=1):
        dirty_n = int(round(len(client_rows) * args.dirty_ratio))
        dirty_indices = set(rng.sample(range(len(client_rows)), dirty_n))
        out_rows = []
        for idx, row in enumerate(client_rows):
            row = dict(row)
            row["_client_id"] = client_idx
            row["split_role"] = "client"
            row["_is_dirty"] = False
            row["_dirty_type"] = "clean"
            if idx in dirty_indices:
                row = corrupt(row, args.dirty_type, rng, client_pool)
                row["_client_id"] = client_idx
                row["split_role"] = "client"
            out_rows.append(row)
        materialized_clients.append(out_rows)

    for row in server:
        row["split_role"] = "server"
        row["_is_dirty"] = False
        row["_dirty_type"] = "clean"
    for row in test:
        row["split_role"] = "test"
        row["_is_dirty"] = False
        row["_dirty_type"] = "clean"

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.outdir / "server_clean.jsonl", server)
    all_client_rows = []
    for client_idx, client_rows in enumerate(materialized_clients, start=1):
        write_jsonl(args.outdir / f"client_{client_idx}_train.jsonl",
                    client_rows)
        all_client_rows.extend(client_rows)
    write_jsonl(args.outdir / "clients_train.jsonl", all_client_rows)
    write_jsonl(args.outdir / "test.jsonl", test)

    manifest = {
        "source": str(args.source),
        "seed": args.seed,
        "dirty_ratio": args.dirty_ratio,
        "dirty_type": args.dirty_type,
        "server_clean": len(server),
        "client_train": len(all_client_rows),
        "test": len(test),
        "client_dirty": [
            sum(bool(row.get("_is_dirty")) for row in client_rows)
            for client_rows in materialized_clients
        ],
    }
    (args.outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
