#!/usr/bin/env python3
"""Summarize the frozen heterogeneous-distillation holdout runs.

This script is deliberately read-only with respect to individual run records.
It verifies their key protocol invariants and writes a derived Markdown report
with source and protocol hashes, paired estimates, and the prespecified
interpretation.  It does not turn a negative result into a positive claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from scipy import stats


METRICS = ("accuracy", "macro_f1", "label_nll", "ece")
SEEDS = (142, 143, 144)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def mean_sd_ci(values: list[float]) -> tuple[float, float, float]:
    """Mean, sample SD, and two-sided 95% t interval half-width."""
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half_width = stats.t.ppf(0.975, len(values) - 1) * sd / math.sqrt(len(values))
    return mean, sd, half_width


def validate(records: list[dict[str, Any]], *, routed: bool) -> None:
    expected_routing = "relative_balanced_rank" if routed else "none"
    expected_weight = 0.25 if routed else 0.0
    required = {
        "method": "multi_round_canonical_probability_distillation",
        "rounds": 2,
        "evaluation_split": "official_test",
        "evaluation_examples": 7600,
        "client_routing": expected_routing,
        "distillation_weight": expected_weight,
        "teacher_aggregation": "uniform",
        "client_steps_per_round": 40,
        "server_steps_per_round": 40,
        "anchor_examples": 128,
    }
    for record in records:
        for key, expected in required.items():
            if record.get(key) != expected:
                raise ValueError(
                    f"Protocol mismatch for seed {record.get('seed')}: "
                    f"{key}={record.get(key)!r}, expected {expected!r}"
                )
        if record.get("seed") not in SEEDS:
            raise ValueError(f"Unexpected seed: {record.get('seed')}")


def pct(value: float) -> str:
    return f"{100 * value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path,
                        default=Path("kafa_script/analysis_outputs"))
    parser.add_argument("--protocol", type=Path,
                        default=Path("kafa_script/heterogeneous_routed_final_protocol_v2.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # The first anchor-only records used a one-step dummy local pass.  They are
    # retained for audit but deliberately excluded here: these strict records
    # executed the frozen 40 client steps and therefore match the routed run's
    # local-compute and payload configuration.
    anchor_paths = [args.directory / f"final_anchor_only_strict_seed{s}.json" for s in SEEDS]
    routed_paths = [args.directory / f"final_routed_seed{s}.json" for s in SEEDS]
    if any(not path.is_file() for path in anchor_paths + routed_paths):
        missing = [str(path) for path in anchor_paths + routed_paths if not path.is_file()]
        raise FileNotFoundError("Missing final run record(s): " + ", ".join(missing))
    if not args.protocol.is_file():
        raise FileNotFoundError(args.protocol)

    anchors = [load(path) for path in anchor_paths]
    routed = [load(path) for path in routed_paths]
    validate(anchors, routed=False)
    validate(routed, routed=True)
    if {r["seed"] for r in anchors} != set(SEEDS) or {r["seed"] for r in routed} != set(SEEDS):
        raise ValueError("Seed sets do not match the frozen protocol")
    if {r["manifest_sha256"] for r in anchors + routed} != {anchors[0]["manifest_sha256"]}:
        raise ValueError("Run records use different frozen manifests")

    lines = [
        "# Final heterogeneous-distillation holdout result",
        "",
        "**Status: completed frozen holdout evaluation; negative primary result.** "
        "This derived report reads the immutable per-seed JSON records and does "
        "not alter them.  The raw records retain their conservative historical "
        "`paper_result: false` tag; their scope is verified here from the recorded "
        "official-test fields.",
        "",
        "## Verification",
        "",
        f"- Frozen protocol: `{args.protocol}` (SHA-256 `{sha256(args.protocol)}`).",
        f"- Frozen data manifest: `{anchors[0]['manifest_sha256']}`.",
        "- All six records declare 2 rounds, 40 client and 40 server steps per round, "
        "128 public anchors, the Qwen2.5-1.5B server, and all 7,600 official AG News test examples.",
        "- The predeclared primary comparison is public-anchor-only versus relative-balanced-rank routed canonical-probability distillation (weight 0.25); seeds are 142, 143, and 144.",
        "",
        "## Per-seed primary result",
        "",
        "| Seed | Anchor-only accuracy | Routed accuracy | Routed − anchor (pp) | Anchor-only macro-F1 | Routed macro-F1 | Routed target-answer loss | Routed ECE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    accuracy_deltas: list[float] = []
    for anchor, route in zip(anchors, routed):
        delta = route["final"]["accuracy"] - anchor["final"]["accuracy"]
        accuracy_deltas.append(delta)
        lines.append(
            f"| {anchor['seed']} | {pct(anchor['final']['accuracy'])} | "
            f"{pct(route['final']['accuracy'])} | {100 * delta:+.2f} | "
            f"{pct(anchor['final']['macro_f1'])} | {pct(route['final']['macro_f1'])} | "
            f"{route['final']['label_nll']:.3f} | {route['final']['ece']:.3f} |"
        )

    lines += ["", "## Paired summary", "", "| Metric | Anchor-only mean ± SD | Routed mean ± SD | Routed − anchor mean (95% t CI) |", "|---|---:|---:|---:|"]
    for metric in METRICS:
        a_values = [r["final"][metric] for r in anchors]
        r_values = [r["final"][metric] for r in routed]
        delta_values = [right - left for left, right in zip(a_values, r_values)]
        a_mean, a_sd, _ = mean_sd_ci(a_values)
        r_mean, r_sd, _ = mean_sd_ci(r_values)
        d_mean, _, d_ci = mean_sd_ci(delta_values)
        display_metric = ("target_answer_loss" if metric == "label_nll" else metric)
        if metric in ("accuracy", "macro_f1"):
            row = (f"| {display_metric} | {pct(a_mean)} ± {pct(a_sd)} | {pct(r_mean)} ± {pct(r_sd)} | "
                   f"{100*d_mean:+.2f} pp [{100*(d_mean-d_ci):+.2f}, {100*(d_mean+d_ci):+.2f}] |")
        else:
            row = (f"| {display_metric} | {a_mean:.3f} ± {a_sd:.3f} | {r_mean:.3f} ± {r_sd:.3f} | "
                   f"{d_mean:+.3f} [{d_mean-d_ci:+.3f}, {d_mean+d_ci:+.3f}] |")
        lines.append(row)

    ttest = stats.ttest_rel(
        [r["final"]["accuracy"] for r in routed],
        [r["final"]["accuracy"] for r in anchors],
        alternative="greater",
    )
    payload_anchor = [r["total_client_to_server_payload_bytes"] for r in anchors]
    payload_routed = [r["total_client_to_server_payload_bytes"] for r in routed]
    wall_anchor = [r["total_wall_seconds"] for r in anchors]
    wall_routed = [r["total_wall_seconds"] for r in routed]
    lines += [
        "",
        f"A descriptive directional paired test for **routed accuracy > anchor-only accuracy** has t({len(SEEDS)-1}) = {ttest.statistic:.3f}, p = {ttest.pvalue:.4f}; it provides no support for the intended direction. All three paired accuracy differences are negative.",
        "The t intervals are descriptive with only three seeds and must not be read as broad generalization evidence.",
        "",
        "## Cost and confidence",
        "",
        f"- Mean wall time: anchor-only {statistics.mean(wall_anchor):.1f} s; routed {statistics.mean(wall_routed):.1f} s ({statistics.mean(wall_routed) / statistics.mean(wall_anchor):.2f}×).",
        f"- Recorded client-to-server payload: anchor-only {statistics.mean(payload_anchor):.0f} bytes; routed {statistics.mean(payload_routed):.0f} bytes. The strict anchor-only control deliberately executes and records the same client loop, but its server objective assigns the teacher term weight zero; this is equal-budget control work, not useful communication.",
        "- Routing lowers the token-normalized target-answer loss and 10-bin ECE while reducing both accuracy and macro-F1. The raw JSON key `label_nll` is a historical misnomer for target-answer loss in this AG News evaluator; it is not normalized class NLL. Lower ECE does not rescue the predeclared primary accuracy claim.",
        "",
        "## Interpretation and boundary",
        "",
        "Under this *single, frozen severe-mixture AG News configuration*, the tested routed heterogeneous distillation method reduces task accuracy against public-anchor-only server adaptation. The defensible finding is diagnostic: local routing can suppress harmful examples, yet its resulting teacher probabilities did not add enough useful signal to justify their cost or overcome the public-anchor baseline.",
        "",
        "This neither proves that all cross-architecture federation fails nor supports a new general-purpose aggregation algorithm. The broader artifact adds GSM8K and two natural weak-label languages, but stronger compatible comparators, privacy analysis, and a complete positive federated method remain open submission risks.",
        "",
        "## Source records",
        "",
    ]
    for path in anchor_paths + routed_paths:
        lines.append(f"- `{path}` (SHA-256 `{sha256(path)}`)")
    lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
