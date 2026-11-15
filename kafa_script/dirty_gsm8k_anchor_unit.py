from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "gsm8k_train.jsonl"
DEFAULT_OUT = ROOT / "kafa_script" / "analysis_outputs" / "dirty_gsm8k_anchor_unit"

TOKEN_RE = re.compile(r"[a-zA-Z]+|\d+(?:\.\d+)?")
FINAL_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
CALC_RE = re.compile(r"<<([^<>]+)=(-?\d+(?:\.\d+)?)>>")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "in",
    "is",
    "it",
    "many",
    "much",
    "of",
    "on",
    "or",
    "she",
    "so",
    "the",
    "then",
    "to",
    "total",
    "was",
    "what",
    "will",
    "with",
}


@dataclass
class Sample:
    sample_id: int
    question: str
    output: str
    clean_output: str
    client_id: int | None
    is_dirty: bool
    dirty_type: str


def read_gsm8k(path: Path, limit: int) -> List[Sample]:
    rows: List[Sample] = []
    with path.open("r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            if limit and len(rows) >= limit:
                break
            item = json.loads(line)
            rows.append(
                Sample(
                    sample_id=idx,
                    question=item["question"],
                    output=item["answer"],
                    clean_output=item["answer"],
                    client_id=None,
                    is_dirty=False,
                    dirty_type="clean",
                )
            )
    return rows


def tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def vectorize(text: str) -> Counter:
    return Counter(tokenize(text))


def cosine(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def mean_vector(vectors: Sequence[Counter]) -> Counter:
    merged: Counter = Counter()
    for vec in vectors:
        merged.update(vec)
    if not vectors:
        return merged
    for key in list(merged):
        merged[key] /= len(vectors)
    return merged


def build_prototypes(server_samples: Sequence[Sample], num_prototypes: int) -> List[Counter]:
    buckets: Dict[int, List[Counter]] = {idx: [] for idx in range(num_prototypes)}
    for sample in server_samples:
        # Stable, cheap bucketing by question length. This is deliberately a unit
        # surrogate for a hidden-state prototype bank, not a training component.
        bucket = min(num_prototypes - 1, len(tokenize(sample.question)) * num_prototypes // 80)
        buckets[bucket].append(vectorize(sample.question))
    return [mean_vector(vectors) for vectors in buckets.values() if vectors]


def final_answer(answer: str) -> str | None:
    match = FINAL_RE.search(answer)
    return match.group(1) if match else None


def replace_final_answer(answer: str, new_value: int) -> str:
    if FINAL_RE.search(answer):
        return FINAL_RE.sub(f"#### {new_value}", answer, count=1)
    return answer.rstrip() + f"\n#### {new_value}"


def corrupt_answer(sample: Sample, rng: random.Random) -> Sample:
    clean_final = final_answer(sample.output)
    old_value = int(float(clean_final)) if clean_final is not None else rng.randint(1, 100)
    offset = rng.choice([-37, -19, -11, 7, 13, 23, 41])
    bad_value = old_value + offset
    if bad_value == old_value:
        bad_value += 1
    return Sample(
        sample.sample_id,
        sample.question,
        replace_final_answer(sample.output, bad_value),
        sample.clean_output,
        sample.client_id,
        True,
        "answer_corruption",
    )


def corrupt_reasoning(sample: Sample, rng: random.Random) -> Sample:
    matches = list(CALC_RE.finditer(sample.output))
    if not matches:
        return corrupt_answer(sample, rng)
    target = rng.choice(matches)
    old_value = int(float(target.group(2)))
    bad_value = old_value + rng.choice([-9, -5, -2, 3, 6, 10])
    output = sample.output[: target.start(2)] + str(bad_value) + sample.output[target.end(2) :]
    output = replace_final_answer(output, bad_value)
    return Sample(
        sample.sample_id,
        sample.question,
        output,
        sample.clean_output,
        sample.client_id,
        True,
        "reasoning_corruption",
    )


def mismatch_answer(sample: Sample, donor: Sample) -> Sample:
    return Sample(
        sample.sample_id,
        sample.question,
        donor.output,
        sample.clean_output,
        sample.client_id,
        True,
        "answer_mismatch",
    )


def split_samples(
    samples: Sequence[Sample],
    server_size: int,
    clients: int,
    client_size: int,
    rng: random.Random,
) -> Tuple[List[Sample], List[List[Sample]]]:
    shuffled = list(samples)
    rng.shuffle(shuffled)
    server_samples = shuffled[:server_size]
    offset = server_size
    client_sets: List[List[Sample]] = []
    for client_id in range(1, clients + 1):
        current = []
        for sample in shuffled[offset : offset + client_size]:
            current.append(
                Sample(
                    sample.sample_id,
                    sample.question,
                    sample.output,
                    sample.clean_output,
                    client_id,
                    False,
                    "clean",
                )
            )
        client_sets.append(current)
        offset += client_size
    return server_samples, client_sets


def inject_dirty(
    client_sets: List[List[Sample]],
    dirty_ratio: float,
    dirty_type: str,
    rng: random.Random,
) -> List[List[Sample]]:
    all_clean = [sample for client in client_sets for sample in client]
    result: List[List[Sample]] = []
    for client in client_sets:
        dirty_count = int(round(len(client) * dirty_ratio))
        dirty_ids = set(rng.sample(range(len(client)), dirty_count)) if dirty_count else set()
        new_client = []
        for idx, sample in enumerate(client):
            if idx not in dirty_ids:
                new_client.append(sample)
                continue
            if dirty_type == "answer_corruption":
                new_client.append(corrupt_answer(sample, rng))
            elif dirty_type == "reasoning_corruption":
                new_client.append(corrupt_reasoning(sample, rng))
            elif dirty_type == "answer_mismatch":
                donor = rng.choice([item for item in all_clean if item.sample_id != sample.sample_id])
                new_client.append(mismatch_answer(sample, donor))
            elif dirty_type == "mixed":
                chosen = rng.choice(["answer_corruption", "reasoning_corruption", "answer_mismatch"])
                if chosen == "answer_corruption":
                    new_client.append(corrupt_answer(sample, rng))
                elif chosen == "reasoning_corruption":
                    new_client.append(corrupt_reasoning(sample, rng))
                else:
                    donor_pool = [
                        item for item in all_clean if item.sample_id != sample.sample_id
                    ]
                    donor = rng.choice(donor_pool)
                    new_client.append(mismatch_answer(sample, donor))
            else:
                raise ValueError(f"Unknown dirty type: {dirty_type}")
        result.append(new_client)
    return result


def safe_eval_expression(expr: str) -> float | None:
    if not re.fullmatch(r"[\d\s\.\+\-\*\/\(\)]+", expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


def arithmetic_consistency(answer: str) -> float:
    matches = list(CALC_RE.finditer(answer))
    if not matches:
        return 0.85 if final_answer(answer) is not None else 0.0
    correct = 0
    declared_values = []
    for match in matches:
        expr_value = safe_eval_expression(match.group(1))
        declared = float(match.group(2))
        declared_values.append(declared)
        if expr_value is not None and abs(expr_value - declared) <= 1e-6:
            correct += 1
    calc_score = correct / len(matches)
    final = final_answer(answer)
    if final is None:
        return 0.0
    final_score = 1.0 if declared_values and abs(float(final) - declared_values[-1]) <= 1e-6 else 0.0
    if final_score == 0.0:
        return min(0.35, 0.25 * calc_score)
    return 0.65 * calc_score + 0.35 * final_score


def numeric_grounding(question: str, answer: str) -> float:
    q_nums = set(re.findall(r"-?\d+(?:\.\d+)?", question))
    a_nums = set(re.findall(r"-?\d+(?:\.\d+)?", answer))
    if not q_nums:
        return 0.7
    return len(q_nums & a_nums) / len(q_nums)


def lexical_grounding(question: str, answer: str) -> float:
    q_tokens = {
        tok
        for tok in tokenize(question)
        if len(tok) > 2 and tok not in STOPWORDS and not tok.replace(".", "", 1).isdigit()
    }
    a_tokens = {
        tok
        for tok in tokenize(answer)
        if len(tok) > 2 and tok not in STOPWORDS and not tok.replace(".", "", 1).isdigit()
    }
    if not q_tokens:
        return 0.7
    return len(q_tokens & a_tokens) / len(q_tokens)


def relevance_score(question: str, prototypes: Sequence[Counter]) -> float:
    q_vec = vectorize(question)
    return max((cosine(q_vec, proto) for proto in prototypes), default=0.0)


def score_sample(
    sample: Sample,
    prototypes: Sequence[Counter],
    relevance_threshold: float,
    dirty_threshold: float,
) -> Dict[str, float | bool]:
    relevance = relevance_score(sample.question, prototypes)
    arithmetic = arithmetic_consistency(sample.output)
    grounding = (
        0.65 * numeric_grounding(sample.question, sample.output)
        + 0.35 * lexical_grounding(sample.question, sample.output)
    )
    consistency = 0.55 * arithmetic + 0.45 * grounding
    dirty_score = relevance * (1.0 - consistency)
    return {
        "relevance_score": relevance,
        "consistency_score": consistency,
        "dirty_score": dirty_score,
        "filtered": relevance >= relevance_threshold and dirty_score >= dirty_threshold,
    }


def summarize(scored: Sequence[Dict]) -> Dict[str, float | int]:
    dirty = [row for row in scored if row["is_dirty"]]
    clean = [row for row in scored if not row["is_dirty"]]
    filtered = [row for row in scored if row["filtered"]]
    dirty_filtered = [row for row in filtered if row["is_dirty"]]
    clean_filtered = [row for row in filtered if not row["is_dirty"]]
    precision = len(dirty_filtered) / len(filtered) if filtered else 0.0
    recall = len(dirty_filtered) / len(dirty) if dirty else 0.0
    fpr = len(clean_filtered) / len(clean) if clean else 0.0
    clean_retention = 1.0 - fpr if clean else 0.0
    return {
        "samples": len(scored),
        "dirty_samples_seen": len(dirty),
        "dirty_samples_filtered": len(dirty_filtered),
        "clean_samples_mistakenly_filtered": len(clean_filtered),
        "dirty_filter_precision": precision,
        "dirty_filter_recall": recall,
        "clean_filter_false_positive_rate": fpr,
        "clean_retention_rate": clean_retention,
    }


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Dict:
    rng = random.Random(args.seed)
    samples = read_gsm8k(args.data, args.total_samples)
    required = args.server_size + args.clients * args.client_size
    if len(samples) < required:
        raise ValueError(f"Need at least {required} samples, found {len(samples)}")

    server_samples, client_sets = split_samples(
        samples,
        args.server_size,
        args.clients,
        args.client_size,
        rng,
    )
    client_sets = inject_dirty(client_sets, args.dirty_ratio, args.dirty_type, rng)
    prototypes = build_prototypes(server_samples, args.num_prototypes)

    scored_rows = []
    for client in client_sets:
        for sample in client:
            scores = score_sample(
                sample,
                prototypes,
                args.relevance_threshold,
                args.dirty_threshold,
            )
            scored_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "client_id": sample.client_id,
                    "dirty_type": sample.dirty_type,
                    "is_dirty": sample.is_dirty,
                    "relevance_score": round(float(scores["relevance_score"]), 6),
                    "consistency_score": round(float(scores["consistency_score"]), 6),
                    "dirty_score": round(float(scores["dirty_score"]), 6),
                    "filtered": bool(scores["filtered"]),
                    "question": sample.question,
                    "output": sample.output,
                    "_clean_output": sample.clean_output,
                }
            )

    summary = summarize(scored_rows)
    per_client = {}
    for client_id in range(1, args.clients + 1):
        per_client[str(client_id)] = summarize(
            [row for row in scored_rows if row["client_id"] == client_id]
        )
    result = {
        "config": {
            "data": str(args.data),
            "seed": args.seed,
            "server_size": args.server_size,
            "clients": args.clients,
            "client_size": args.client_size,
            "dirty_ratio": args.dirty_ratio,
            "dirty_type": args.dirty_type,
            "num_prototypes": args.num_prototypes,
            "relevance_threshold": args.relevance_threshold,
            "dirty_threshold": args.dirty_threshold,
        },
        "summary": summary,
        "per_client": per_client,
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.outdir / "scored_samples.jsonl", scored_rows)
    write_csv(args.outdir / "scored_samples.csv", scored_rows)
    (args.outdir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone GSM8K dirty-supervision anchor consistency unit experiment."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-samples", type=int, default=600)
    parser.add_argument("--server-size", type=int, default=120)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--client-size", type=int, default=80)
    parser.add_argument("--dirty-ratio", type=float, default=0.10)
    parser.add_argument(
        "--dirty-type",
        choices=["answer_corruption", "reasoning_corruption", "answer_mismatch", "mixed"],
        default="answer_corruption",
    )
    parser.add_argument("--num-prototypes", type=int, default=8)
    parser.add_argument("--relevance-threshold", type=float, default=0.15)
    parser.add_argument("--dirty-threshold", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
