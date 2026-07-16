#!/usr/bin/env python3
"""Validate and summarize the frozen GSM8K cross-family development study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from scipy import stats

SEEDS = (61, 62, 63)
FAMILIES = ("smol", "tiny")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    half = stats.t.ppf(.975, len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))
    return mean, half


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("kafa_script/analysis_outputs"))
    parser.add_argument("--protocol", type=Path, default=Path("kafa_script/gsm8k_cross_family_development_protocol_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records: dict[tuple[str, str, str, int], tuple[Path, dict]] = {}
    for family in FAMILIES:
        for method in ("naive", "teacher"):
            for condition in ("clean", "severe"):
                for seed in SEEDS:
                    path = args.directory / f"gsm8k_crossfamily_{family}_{method}_{condition}_seed{seed}.json"
                    record = json.loads(path.read_text())
                    if record.get("evidence_label") != "benchmark_dev":
                        raise ValueError(f"{path}: not a development record")
                    if record.get("evaluation_examples") != 512 or record.get("seed") != seed:
                        raise ValueError(f"{path}: invalid scope or seed")
                    records[(family, method, condition, seed)] = (path, record)

    manifests = {record["manifest_sha256"] for _, record in records.values()}
    if len(manifests) != 1:
        raise ValueError("records use different manifests")

    lines = [
        "# GSM8K cross-family development results", "",
        "**Status: frozen development evaluation; not an official-test result.**", "",
        "## Verification", "",
        f"- Protocol SHA-256: `{sha256(args.protocol)}`.",
        f"- Data manifest SHA-256: `{next(iter(manifests))}`.",
        "- All 24 records use all 512 frozen development questions and seeds 61--63.", "",
        "## Results", "",
        "| Client family | Condition | Naive EM | Teacher EM | Paired gain | Descriptive 95% CI | Gate |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    advances = []
    details = []
    for family in FAMILIES:
        family_deltas = {}
        for condition in ("clean", "severe"):
            naive = [records[(family, "naive", condition, seed)][1]["final"]["exact_match"] for seed in SEEDS]
            teacher = [records[(family, "teacher", condition, seed)][1]["final"]["exact_match"] for seed in SEEDS]
            delta = [b - a for a, b in zip(naive, teacher)]
            family_deltas[condition] = delta
            mean, half = interval(delta)
            gate = "clean-safe" if condition == "clean" and min(delta) >= -.02 else "positive" if condition == "severe" and mean > 0 else "fail"
            lines.append(f"| {family} | {condition} | {100*statistics.mean(naive):.2f}% | {100*statistics.mean(teacher):.2f}% | {100*mean:+.2f} pp | [{100*(mean-half):+.2f}, {100*(mean+half):+.2f}] | {gate} |")
            details.append(f"- {family} {condition}: per-seed gains {', '.join(f'{100*x:+.2f}' for x in delta)} pp.")
        advances.append(min(family_deltas["clean"]) >= -.02 and statistics.mean(family_deltas["severe"]) > 0)
    lines += ["", "Per-seed audit:", "", *details, "", "## Decision", ""]
    if all(advances):
        lines.append("Both non-Qwen client families pass the predeclared development gate. This permits drafting a separate holdout protocol; it is not itself a cross-family final result.")
    else:
        lines.append("The general cross-family development gate fails. No cross-family holdout protocol should be drafted from this candidate without declaring a new method-selection cycle.")
    lines += ["", "This study transfers teacher scores across model families but remains centralized local adaptation, not a full federated server comparison. Confidence intervals are descriptive because each family has only three seeds.", "", "## Source records", ""]
    for key in sorted(records):
        path, _ = records[key]
        lines.append(f"- `{path}` (SHA-256 `{sha256(path)}`)")
    lines.append("")
    args.output.write_text("\n".join(lines))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
