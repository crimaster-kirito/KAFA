#!/usr/bin/env python3
"""Validate and summarize the frozen GSM8K cross-family holdout study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from scipy import stats

SEEDS = (164, 165, 166)
FAMILIES = ("smol", "tiny")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_interval(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    half = stats.t.ppf(.975, len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))
    return mean, half


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("kafa_script/analysis_outputs"))
    parser.add_argument("--protocol", type=Path, default=Path("kafa_script/gsm8k_cross_family_final_protocol_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records: dict[tuple[str, str, str, int], tuple[Path, dict]] = {}
    for family in FAMILIES:
        for method in ("naive", "teacher"):
            for condition in ("clean", "severe"):
                for seed in SEEDS:
                    path = args.directory / f"gsm8k_crossfamily_final_{family}_{method}_{condition}_seed{seed}.json"
                    record = json.loads(path.read_text())
                    if record.get("evidence_label") != "benchmark_holdout":
                        raise ValueError(f"{path}: not a holdout record")
                    if record.get("evaluation_split") != "official_test" or record.get("evaluation_examples") != 1319:
                        raise ValueError(f"{path}: invalid evaluation scope")
                    if record.get("seed") != seed:
                        raise ValueError(f"{path}: seed mismatch")
                    records[(family, method, condition, seed)] = (path, record)

    manifests = {record["manifest_sha256"] for _, record in records.values()}
    if len(manifests) != 1:
        raise ValueError("records use different manifests")

    values: dict[str, dict[str, list[float]]] = {}
    pvalues = {}
    for family in FAMILIES:
        values[family] = {}
        for condition in ("clean", "severe"):
            naive = [records[(family, "naive", condition, seed)][1]["final"]["exact_match"] for seed in SEEDS]
            teacher = [records[(family, "teacher", condition, seed)][1]["final"]["exact_match"] for seed in SEEDS]
            values[family][f"naive_{condition}"] = naive
            values[family][f"teacher_{condition}"] = teacher
            values[family][f"delta_{condition}"] = [b - a for a, b in zip(naive, teacher)]
        pvalues[family] = float(stats.ttest_rel(values[family]["teacher_severe"], values[family]["naive_severe"], alternative="greater").pvalue)
    adjusted = holm(pvalues)

    lines = [
        "# Final GSM8K cross-family teacher-ranking result", "",
        "**Status: completed frozen holdout evaluation.**", "",
        "## Verification", "",
        f"- Frozen protocol SHA-256: `{sha256(args.protocol)}`.",
        f"- Frozen data manifest SHA-256: `{next(iter(manifests))}`.",
        "- All 24 records are labelled `benchmark_holdout`, evaluate all 1,319 official test questions, and use holdout seeds 164--166.",
        "- Two primary severe-condition tests use Holm correction at alpha 0.05; clean safety requires every paired difference to be at least -2 pp.", "",
        "## Results", "",
        "| Client | Condition | Naive EM | Teacher EM | Paired gain [95% CI] | Raw one-sided p | Holm p | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    passes = []
    for family in FAMILIES:
        safe = min(values[family]["delta_clean"]) >= -.02
        severe_pass = adjusted[family] < .05
        passes.append(safe and severe_pass)
        for condition in ("clean", "severe"):
            naive = values[family][f"naive_{condition}"]
            teacher = values[family][f"teacher_{condition}"]
            delta = values[family][f"delta_{condition}"]
            mean, half = mean_interval(delta)
            if condition == "severe":
                raw_p, adj_p = f"{pvalues[family]:.4f}", f"{adjusted[family]:.4f}"
                gate = "pass" if safe and severe_pass else "fail"
            else:
                raw_p = adj_p = "--"
                gate = "safe" if safe else "fail"
            lines.append(f"| {family} | {condition} | {100*statistics.mean(naive):.2f}% | {100*statistics.mean(teacher):.2f}% | {100*mean:+.2f} pp [{100*(mean-half):+.2f}, {100*(mean+half):+.2f}] | {raw_p} | {adj_p} | {gate} |")
    lines += ["", "Per-seed paired gains:", ""]
    for family in FAMILIES:
        for condition in ("clean", "severe"):
            gains = ", ".join(f"{100*x:+.2f}" for x in values[family][f"delta_{condition}"])
            lines.append(f"- {family} {condition}: {gains} pp.")
    lines += ["", "## Claim decision", ""]
    if all(passes):
        lines.append("Both client families pass the frozen multiplicity-corrected severe and clean-safety gates. The evidence supports cross-family transfer of the public-anchor teacher ranking mechanism for this controlled GSM8K mixture.")
    else:
        lines.append("At least one client family fails the frozen gate, so the study does not support a general cross-family holdout claim. Family-specific effects remain reportable with this failure explicit.")
    lines += ["", "This result does not establish a complete federated aggregation method, privacy protection, or robustness to naturally occurring noise. The teacher-ranking endpoint is centralized and its per-example scoring adds substantial computation.", "", "## Source records", ""]
    for key in sorted(records):
        path, _ = records[key]
        lines.append(f"- `{path}` (SHA-256 `{sha256(path)}`)")
    lines.append("")
    args.output.write_text("\n".join(lines))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
