#!/usr/bin/env python3
"""Validate and summarize the frozen AG News Co-Teaching development matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from scipy import stats

FAMILIES = ("qwen", "smol", "tiny")
CONDITIONS = ("clean", "conflict40", "joint")
SEEDS = (75, 76, 77)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    half = stats.t.ppf(.975, len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))
    return mean, half


def holm(pvalues):
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def paired_greater_p(candidate: list[float], baseline: list[float]) -> float:
    """Return a conservative finite p-value for degenerate paired samples."""
    differences = [candidate_value - baseline_value
                   for candidate_value, baseline_value in zip(candidate, baseline)]
    if math.isclose(max(differences), min(differences),
                    rel_tol=0.0, abs_tol=1e-12):
        # Three identical differences have zero estimated seed variance; a
        # t statistic is undefined, not infinitely certain.  Returning one is
        # conservative for the declared improvement test and avoids a false
        # p=0 claim.
        return 1.0
    return float(stats.ttest_rel(candidate, baseline, alternative="greater").pvalue)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("kafa_script/analysis_outputs"))
    parser.add_argument("--protocol", type=Path, default=Path("kafa_script/agnews_coteaching_development_protocol_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, paths = {}, []
    for family in FAMILIES:
        for method in ("naive", "coteach"):
            for condition in CONDITIONS:
                for seed in SEEDS:
                    path = args.directory / f"agnews_coteach_{family}_{method}_{condition}_seed{seed}.json"
                    record = json.loads(path.read_text())
                    if record.get("evidence_label") != "benchmark_dev" or record.get("seed") != seed:
                        raise ValueError(f"{path}: invalid evidence or seed")
                    if record.get("eval_examples") != 800:
                        raise ValueError(f"{path}: invalid development scope")
                    records[(family, method, condition, seed)] = record
                    paths.append(path)
    for family in FAMILIES:
        for condition in CONDITIONS:
            for seed in SEEDS:
                naive = records[(family, "naive", condition, seed)]
                coteach = records[(family, "coteach", condition, seed)]
                if naive["mixture_assignment_sha256"] != coteach["mixture_assignment_sha256"]:
                    raise ValueError(f"mixture mismatch: {family}/{condition}/{seed}")

    def metric(family, method, condition):
        key = "final" if method == "naive" else "ensemble_final"
        return [records[(family, method, condition, seed)][key]["accuracy"] for seed in SEEDS]

    deltas, pvalues = {}, {}
    for family in FAMILIES:
        for condition in CONDITIONS:
            naive = metric(family, "naive", condition)
            coteach = metric(family, "coteach", condition)
            deltas[(family, condition)] = [b - a for a, b in zip(naive, coteach)]
            if condition != "clean":
                pvalues[(family, condition)] = paired_greater_p(coteach, naive)
    adjusted = holm(pvalues)

    lines = [
        "# AG News Co-Teaching strong-baseline development result", "",
        "**Status: frozen development-only comparison; no official test labels used.**", "",
        "## Verification", "",
        f"- Protocol SHA-256: `{sha256(args.protocol)}`.",
        "- All 54 records evaluate the same frozen 800-example development split at seeds 75--77.",
        "- Every naive/Co-Teaching pair has an identical mixture-assignment hash.",
        "- Six contaminated family-by-condition tests are Holm-adjusted together; Co-Teaching is advantaged by the true contamination fraction as its forget rate.", "",
        "## Accuracy", "",
        "| Client | Condition | Naive | Co-Teaching ensemble | Paired gain [95% CI] | Raw p | Holm p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        for condition in CONDITIONS:
            naive = metric(family, "naive", condition)
            coteach = metric(family, "coteach", condition)
            mean, half = interval(deltas[(family, condition)])
            raw = adj = "--"
            if condition != "clean":
                raw = f"{pvalues[(family, condition)]:.4f}"
                adj = f"{adjusted[(family, condition)]:.4f}"
            lines.append(f"| {family} | {condition} | {100*statistics.mean(naive):.2f}% | {100*statistics.mean(coteach):.2f}% | {100*mean:+.2f} pp [{100*(mean-half):+.2f}, {100*(mean+half):+.2f}] | {raw} | {adj} |")
    lines += ["", "Per-seed gains:", ""]
    for family in FAMILIES:
        for condition in CONDITIONS:
            lines.append(f"- {family} {condition}: " + ", ".join(f"{100*x:+.2f}" for x in deltas[(family, condition)]) + " pp.")

    lines += ["", "## Selection and cost audit", ""]
    for family in FAMILIES:
        for condition in ("conflict40", "joint"):
            clean_rates, wall_ratios, byte_ratios = [], [], []
            for seed in SEEDS:
                naive = records[(family, "naive", condition, seed)]
                coteach = records[(family, "coteach", condition, seed)]
                counts = coteach["exchanged_selected_kind_counts"]
                for count in counts:
                    clean_rates.append(count.get("clean", 0) / max(1, sum(count.values())))
                wall_ratios.append(coteach["train_wall_seconds"] / naive["train_wall_seconds"])
                byte_ratios.append(coteach["serialized_two_adapter_bytes"] / naive["serialized_adapter_bytes"])
            lines.append(f"- {family} {condition}: mean exchanged clean fraction {100*statistics.mean(clean_rates):.1f}%; train-wall ratio {statistics.mean(wall_ratios):.2f}x; serialized-adapter ratio {statistics.mean(byte_ratios):.2f}x.")

    safe = all(min(deltas[(family, "clean")]) >= -.02 for family in FAMILIES)
    contaminated_positive = all(statistics.mean(deltas[(family, condition)]) > 0
                                for family in FAMILIES for condition in ("conflict40", "joint"))
    lines += ["", "## Interpretation", ""]
    if safe and contaminated_positive:
        lines.append("Co-Teaching is clean-safe and has positive mean gains in every contaminated setting. Corrected significance is reported above; this development result can justify a separately frozen holdout only if the predeclared multiplicity gate is met.")
    else:
        lines.append("Co-Teaching fails either clean safety or a contaminated mean comparison, so it does not provide a uniformly stronger compatible baseline under this protocol.")
    lines += ["", "This baseline uses two complete backbones/adapters and oracle contamination rates. The joint off-task condition is an intentional stress test beyond Co-Teaching's original finite-label-noise setting. Delora is not represented: its published pipeline additionally requires GPT-4o pseudo-label construction and relabeling.", "", "## Source artifacts", ""]
    for path in sorted(paths):
        lines.append(f"- `{path}` (SHA-256 `{sha256(path)}`)")
    lines.append("")
    args.output.write_text("\n".join(lines))
    print(f"Wrote {args.output}")
if __name__ == "__main__":
    main()
