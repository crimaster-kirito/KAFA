#!/usr/bin/env python3
"""Validate and summarize the frozen GSM8K holdout comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from scipy import stats

SEEDS = (158, 159, 160)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def interval(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = stats.t.ppf(.975, len(values) - 1) * sd / math.sqrt(len(values))
    return mean, sd, half


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("kafa_script/analysis_outputs"))
    parser.add_argument("--protocol", type=Path, default=Path("kafa_script/gsm8k_anchor_teacher_final_protocol_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records: dict[tuple[str, str, int], tuple[Path, dict]] = {}
    for method in ("naive", "teacher"):
        for condition in ("clean", "severe"):
            for seed in SEEDS:
                path = args.directory / f"gsm8k_final_{method}_{condition}_seed{seed}.json"
                if not path.is_file():
                    raise FileNotFoundError(path)
                record = load(path)
                if record.get("evidence_label") != "benchmark_holdout":
                    raise ValueError(f"{path}: not a holdout record")
                if record.get("evaluation_split") != "official_test" or record.get("evaluation_examples") != 1319:
                    raise ValueError(f"{path}: invalid evaluation scope")
                if record.get("seed") != seed:
                    raise ValueError(f"{path}: seed mismatch")
                records[(method, condition, seed)] = (path, record)
    manifests = {r["manifest_sha256"] for _, r in records.values()}
    if len(manifests) != 1:
        raise ValueError("Records use different manifests")
    def metric(method: str, condition: str) -> list[float]:
        return [records[(method, condition, s)][1]["final"]["exact_match"] for s in SEEDS]
    ns, ts, nc, tc = metric("naive", "severe"), metric("teacher", "severe"), metric("naive", "clean"), metric("teacher", "clean")
    severe_delta = [b - a for a, b in zip(ns, ts)]
    clean_delta = [b - a for a, b in zip(nc, tc)]
    sm, ss, sh = interval(severe_delta)
    test = stats.ttest_rel(ts, ns, alternative="greater")
    lines = [
        "# Final GSM8K public-anchor teacher result", "",
        "**Status: completed frozen holdout evaluation.** This report is derived from immutable per-seed records; it reports the controlled result and its narrow claim boundary.", "",
        "## Verification", "",
        f"- Frozen protocol SHA-256: `{sha256(args.protocol)}`.",
        f"- Frozen data manifest SHA-256: `{next(iter(manifests))}`.",
        "- All 12 records are explicitly labelled `benchmark_holdout`, evaluate all 1,319 official GSM8K test questions, and use seeds 158--160.",
        "- The teacher uses public training anchors only; client pools and mixtures are built from the GSM8K training split. Test labels are used only for reported exact match.", "",
        "## Severe mixture primary comparison", "",
        "| Seed | Naive EM | Teacher-routed EM | Difference | Retained clean / conflict / off-task |",
        "|---:|---:|---:|---:|---:|",
    ]
    for s, a, b in zip(SEEDS, ns, ts):
        selected = records[("teacher", "severe", s)][1]["selection"]["selected_kind_counts"]
        counts = f"{selected.get('clean', 0)} / {selected.get('conflict', 0)} / {selected.get('off_task', 0)}"
        lines.append(f"| {s} | {100*a:.2f} | {100*b:.2f} | {100*(b-a):+.2f} pp | {counts} |")
    nmean, nsd, _ = interval(ns); tmean, tsd, _ = interval(ts)
    lines += ["", f"Naive severe mean ± SD: **{100*nmean:.2f} ± {100*nsd:.2f}%**. Teacher-routed severe mean ± SD: **{100*tmean:.2f} ± {100*tsd:.2f}%**.", f"The paired improvement is **{100*sm:+.2f} pp** (descriptive 95% t interval [{100*(sm-sh):+.2f}, {100*(sm+sh):+.2f}] pp); one-sided t(2)={test.statistic:.3f}, p={test.pvalue:.4f}.", "", "## Clean safety", "", "| Seed | Naive clean EM | Teacher-routed clean EM | Difference |", "|---:|---:|---:|---:|"]
    for s, a, b in zip(SEEDS, nc, tc):
        lines.append(f"| {s} | {100*a:.2f} | {100*b:.2f} | {100*(b-a):+.2f} pp |")
    lines += ["", f"Clean paired differences: {', '.join(f'{100*x:+.2f} pp' for x in clean_delta)}. The protocol safety bound is -2.00 pp per seed.", "", "## Interpretation and boundary", "", "If the severe interval is positive and clean safety holds, this supports a **controlled** conclusion: an independently anchor-trained teacher can make supplied numerical labels more trustworthy than a local-only selector under the tested synthetic conflict/off-task mixture. It does not establish privacy, naturally occurring label-noise robustness, general cross-family federation, or a universal teacher-selection method.", "", "The heterogeneous server-distillation baseline on AG News remains a negative result. A CCF-A submission still requires matched heterogeneous server experiments, broader baselines, naturally occurring heterogeneity, and correction-aware multi-setting testing.", "", "## Source records", ""]
    for key in sorted(records):
        path, _ = records[key]
        lines.append(f"- `{path}` (SHA-256 `{sha256(path)}`)")
    lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
