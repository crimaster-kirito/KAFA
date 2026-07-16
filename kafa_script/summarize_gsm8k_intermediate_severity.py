#!/usr/bin/env python3
"""Validate and summarize the frozen secondary-holdout severity curve."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from scipy import stats

FAMILIES = ("qwen", "smol", "tiny")
LEVELS = ("clean", "medium", "severe")
SEEDS = (72, 73, 74)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_interval(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    half = stats.t.ppf(.975, len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))
    return mean, half


def holm(pvalues: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("kafa_script/analysis_outputs"))
    parser.add_argument("--protocol", type=Path, default=Path("kafa_script/gsm8k_intermediate_severity_holdout_protocol_v1.json"))
    parser.add_argument("--evaluation-manifest", type=Path, default=Path("kafa_script/gsm8k_secondary_holdout_manifest_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evaluation_hash = sha256(args.evaluation_manifest)
    records = {}
    paths = []
    for family in FAMILIES:
        for method in ("naive", "teacher"):
            for level in LEVELS:
                for seed in SEEDS:
                    path = args.directory / f"gsm8k_severity_{family}_{method}_{level}_seed{seed}.json"
                    record = json.loads(path.read_text())
                    if record.get("evidence_label") != "benchmark_holdout":
                        raise ValueError(f"{path}: not a holdout record")
                    if record.get("evaluation_split") != "secondary_holdout" or record.get("evaluation_examples") != 1024:
                        raise ValueError(f"{path}: invalid evaluation scope")
                    if record.get("evaluation_manifest_sha256") != evaluation_hash:
                        raise ValueError(f"{path}: evaluation manifest mismatch")
                    if record.get("seed") != seed or record.get("initial") is not None:
                        raise ValueError(f"{path}: seed or skip-initial invariant failed")
                    records[(family, method, level, seed)] = record
                    paths.append(path)

    def metric(family: str, method: str, level: str) -> list[float]:
        return [records[(family, method, level, seed)]["final"]["exact_match"] for seed in SEEDS]

    pvalues = {}
    deltas = {}
    for family in FAMILIES:
        for level in LEVELS:
            naive = metric(family, "naive", level)
            teacher = metric(family, "teacher", level)
            deltas[(family, level)] = [b - a for a, b in zip(naive, teacher)]
            pvalues[(family, level)] = float(stats.ttest_rel(teacher, naive, alternative="greater").pvalue)
    adjusted = holm(pvalues)

    lines = [
        "# GSM8K intermediate-severity secondary holdout", "",
        "**Status: completed frozen secondary-holdout study; the official GSM8K test split was not accessed.**", "",
        "## Verification", "",
        f"- Protocol SHA-256: `{sha256(args.protocol)}`.",
        f"- Secondary holdout manifest SHA-256: `{evaluation_hash}`.",
        "- All 54 records evaluate the same 1,024 disjoint examples, use seeds 72--74, and skip repeated initial evaluation.",
        "- Nine one-sided paired contrasts are Holm-adjusted together; intervals are descriptive with three seeds.", "",
        "## Severity curves", "",
        "| Client | Severity | Naive EM | Teacher EM | Paired gain [95% CI] | Raw p | Holm p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        for level in LEVELS:
            naive = metric(family, "naive", level)
            teacher = metric(family, "teacher", level)
            mean, half = mean_interval(deltas[(family, level)])
            lines.append(f"| {family} | {level} | {100*statistics.mean(naive):.2f}% | {100*statistics.mean(teacher):.2f}% | {100*mean:+.2f} pp [{100*(mean-half):+.2f}, {100*(mean+half):+.2f}] | {pvalues[(family, level)]:.4f} | {adjusted[(family, level)]:.4f} |")
    lines += ["", "Per-seed paired gains:", ""]
    for family in FAMILIES:
        for level in LEVELS:
            gains = ", ".join(f"{100*x:+.2f}" for x in deltas[(family, level)])
            lines.append(f"- {family} {level}: {gains} pp.")

    safe = all(min(deltas[(family, "clean")]) >= -.02 for family in FAMILIES)
    nonnegative = all(statistics.mean(deltas[(family, level)]) >= 0
                      for family in FAMILIES for level in ("medium", "severe"))
    lines += ["", "## Frozen curve decision", ""]
    if safe and nonnegative:
        lines.append("The predeclared descriptive robustness-curve gate passes: clean safety holds and every family has nonnegative mean teacher gain at medium and severe contamination. This does not imply that all multiplicity-corrected individual contrasts are significant.")
    else:
        lines.append("The predeclared descriptive robustness-curve gate fails; a general monotone/safe robustness statement is unsupported.")
    lines += ["", "This secondary holdout is disjoint from anchors, development examples, and client pools, but comes from the GSM8K training split. It is not a replacement for the official test and does not repair previously failed primary claims.", "", "## Source artifacts", ""]
    for path in sorted(paths):
        lines.append(f"- `{path}` (SHA-256 `{sha256(path)}`)")
    lines.append("")
    args.output.write_text("\n".join(lines))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
