#!/usr/bin/env python3
"""Summarize matched GSM8K selection controls against the teacher endpoint."""

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
SELECTORS = ("numeric_random", "client_small_loss")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ci(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    half = stats.t.ppf(.975, len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))
    return mean, half


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("kafa_script/analysis_outputs"))
    parser.add_argument("--protocol", type=Path, default=Path("kafa_script/gsm8k_matched_selection_baselines_protocol_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {}
    sources = []
    for family in FAMILIES:
        for selector in SELECTORS:
            for condition in ("clean", "severe"):
                for seed in SEEDS:
                    path = args.directory / f"gsm8k_matched_{family}_{selector}_{condition}_seed{seed}.json"
                    record = json.loads(path.read_text())
                    if record.get("evidence_label") != "benchmark_dev" or record.get("development_examples") != 512:
                        raise ValueError(f"{path}: invalid development scope")
                    records[(family, selector, condition, seed)] = record
                    sources.append(path)
        for method in ("naive", "teacher"):
            for condition in ("clean", "severe"):
                for seed in SEEDS:
                    path = args.directory / f"gsm8k_crossfamily_{family}_{method}_{condition}_seed{seed}.json"
                    record = json.loads(path.read_text())
                    if record.get("evidence_label") != "benchmark_dev" or record.get("evaluation_examples") != 512:
                        raise ValueError(f"{path}: invalid reference scope")
                    records[(family, method, condition, seed)] = record
                    sources.append(path)

    def metrics(family: str, method: str, condition: str) -> list[float]:
        return [records[(family, method, condition, seed)]["final"]["exact_match"] for seed in SEEDS]

    lines = [
        "# GSM8K matched selection baseline results", "",
        "**Status: post-hoc frozen development comparison; no official-test labels used.**", "",
        f"Protocol SHA-256: `{sha256(args.protocol)}`. All methods retain 128 examples and use 40 client optimizer steps.", "",
        "## Mean exact match", "",
        "| Client | Condition | Naive all | Numeric random | Client small loss | Public-anchor teacher |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        for condition in ("clean", "severe"):
            means = [statistics.mean(metrics(family, method, condition)) for method in ("naive", *SELECTORS, "teacher")]
            lines.append(f"| {family} | {condition} | " + " | ".join(f"{100*x:.2f}%" for x in means) + " |")
    lines += ["", "## Teacher-specific paired contrasts", "",
              "| Client | Condition | Against | Teacher gain [descriptive 95% CI] | Per-seed gains |",
              "|---|---|---|---:|---|" ]
    severe_wins = []
    for family in FAMILIES:
        for condition in ("clean", "severe"):
            teacher = metrics(family, "teacher", condition)
            for selector in SELECTORS:
                baseline = metrics(family, selector, condition)
                delta = [a - b for a, b in zip(teacher, baseline)]
                mean, half = ci(delta)
                lines.append(f"| {family} | {condition} | {selector} | {100*mean:+.2f} pp [{100*(mean-half):+.2f}, {100*(mean+half):+.2f}] | {', '.join(f'{100*x:+.2f}' for x in delta)} |")
                if condition == "severe": severe_wins.append(mean > 0)
    lines += ["", "## Selection audit", ""]
    for family in FAMILIES:
        for selector in SELECTORS:
            counts = []
            times = []
            for seed in SEEDS:
                selection = records[(family, selector, "severe", seed)]["selection"]
                counts.append(selection["selected_kind_counts"])
                times.append(selection["wall_seconds"])
            lines.append(f"- {family} {selector}: severe selected kinds by seed {counts}; mean selection time {statistics.mean(times):.2f} s.")
    lines += ["", "## Interpretation", ""]
    if all(severe_wins):
        lines.append("The teacher has higher mean severe EM than both matched selectors for both families. This supports a teacher-signal interpretation on development data, while uncertainty and the failed multiplicity-corrected holdout gate still preclude a general claim.")
    else:
        lines.append("The teacher does not beat every matched selector in severe mean EM. Its gain cannot be specifically attributed to the public-anchor trust signal under this comparison.")
    lines += ["", "These controls are post-hoc development analyses. They do not repair the failed cross-family holdout claim and must not be reported as confirmatory tests.", "", "## Source artifacts", ""]
    for path in sorted(set(sources)):
        lines.append(f"- `{path}` (SHA-256 `{sha256(path)}`)")
    lines.append("")
    args.output.write_text("\n".join(lines))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
