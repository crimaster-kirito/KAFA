#!/usr/bin/env python3
"""Validate and summarize the post-observation AG News seed replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from scipy import stats

ORIGINAL = (142, 143, 144)
REPLICATION = (145, 146, 147, 148, 149)
EXPECTED_PROTOCOL_SHA256 = (
    "e952b40e07d830e0ea8ced6dafda31fd24e6549b661b87dc3941e278aedbd5a8"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    half = (stats.t.ppf(.975, len(values) - 1) * statistics.stdev(values) /
            math.sqrt(len(values)))
    return mean, half


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path,
                        default=Path("kafa_script/analysis_outputs"))
    parser.add_argument("--protocol", type=Path, default=Path(
        "kafa_script/agnews_post_observation_seed_replication_protocol_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_sha256 = sha256(args.protocol)
    if protocol_sha256 != EXPECTED_PROTOCOL_SHA256:
        raise ValueError(
            "replication protocol differs from the pre-run frozen artifact: "
            f"expected {EXPECTED_PROTOCOL_SHA256}, got {protocol_sha256}"
        )
    pairs: dict[int, tuple[dict, dict]] = {}
    paths: list[Path] = []
    for seed in ORIGINAL + REPLICATION:
        if seed in ORIGINAL:
            anchor_path = args.directory / f"final_anchor_only_strict_seed{seed}.json"
            routed_path = args.directory / f"final_routed_seed{seed}.json"
        else:
            anchor_path = args.directory / f"replication_anchor_only_seed{seed}.json"
            routed_path = args.directory / f"replication_routed_seed{seed}.json"
        anchor, routed = (json.loads(anchor_path.read_text()),
                          json.loads(routed_path.read_text()))
        for record in (anchor, routed):
            expected = {"seed": seed, "rounds": 2,
                        "client_steps_per_round": 40,
                        "server_steps_per_round": 40,
                        "client_batch_size": 8, "server_batch_size": 8,
                        "anchor_score_batch_size": 8,
                        "anchor_supervision_weight": 1.0,
                        "teacher_aggregation": "uniform",
                        "evaluation_split": "official_test",
                        "evaluation_examples": 7600}
            for key, value in expected.items():
                if record.get(key) != value:
                    raise ValueError(f"{seed}: {key} is not {value!r}")
        if anchor["distillation_weight"] != 0 or anchor["client_routing"] != "none":
            raise ValueError(f"{seed}: invalid anchor control")
        if (routed["distillation_weight"] != .25 or
                routed["client_routing"] != "relative_balanced_rank"):
            raise ValueError(f"{seed}: invalid routed treatment")
        for key in ("manifest_sha256", "server_weight_sha256", "client_models"):
            if anchor[key] != routed[key]:
                raise ValueError(f"{seed}: paired {key} mismatch")
        pairs[seed] = (anchor, routed)
        paths.extend((anchor_path, routed_path))

    def deltas(seeds):
        return [pairs[seed][1]["final"]["accuracy"] -
                pairs[seed][0]["final"]["accuracy"] for seed in seeds]

    original = deltas(ORIGINAL)
    replication = deltas(REPLICATION)
    combined = deltas(ORIGINAL + REPLICATION)
    nonpositive = sum(value <= 0 for value in replication)
    gate = nonpositive >= 4 and statistics.mean(replication) <= 0

    lines = [
        "# AG News post-observation independent-seed replication", "",
        "**Status: official test was already observed before this protocol; this is seed replication, not a new untouched holdout.**", "",
        "## Verification", "",
        f"- Replication protocol SHA-256: `{protocol_sha256}` (matches the pre-run frozen digest).",
        "- The frozen protocol and scheduler specify no method, hyperparameter, data, or evaluation changes from seeds 142--144; all fields present in both generations of raw records match. The legacy schema omits some CLI fields, so this is internal consistency evidence rather than complete independent provenance.",
        "- New seeds 145--149 were frozen together before any new run.", "",
        "## Per-seed accuracy", "",
        "| Evidence phase | Seed | Anchor-only | Routed | Routed - anchor |",
        "|---|---:|---:|---:|---:|",
    ]
    for phase, seeds in (("original frozen holdout", ORIGINAL),
                         ("post-observation replication", REPLICATION)):
        for seed in seeds:
            anchor, routed = pairs[seed]
            delta = routed["final"]["accuracy"] - anchor["final"]["accuracy"]
            lines.append(f"| {phase} | {seed} | {100*anchor['final']['accuracy']:.2f}% | "
                         f"{100*routed['final']['accuracy']:.2f}% | {100*delta:+.2f} pp |")
    lines += ["", "## Separate phase summaries", "",
              "| Phase | Seeds | Mean gain [descriptive 95% CI] | Nonpositive signs |",
              "|---|---:|---:|---:|"]
    for name, values in (("original", original), ("replication", replication),
                         ("combined descriptive", combined)):
        mean, half = interval(values)
        lines.append(f"| {name} | {len(values)} | {100*mean:+.2f} pp "
                     f"[{100*(mean-half):+.2f}, {100*(mean+half):+.2f}] | "
                     f"{sum(value <= 0 for value in values)}/{len(values)} |")
    lines += ["", "## Frozen directional seed-consistency heuristic", "",
              "This 4-of-5 sign-and-mean rule is not an alpha-controlled significance test; it only checks whether the observed direction repeats across new seeds.", ""]
    if gate:
        lines.append(f"The predeclared seed-consistency heuristic passes: {nonpositive}/5 new differences are nonpositive and the new-seed mean is {100*statistics.mean(replication):+.2f} pp.")
    else:
        lines.append(f"The predeclared seed-consistency heuristic fails: {nonpositive}/5 new differences are nonpositive and the new-seed mean is {100*statistics.mean(replication):+.2f} pp. The original small negative effect is not directionally consistent across the new seeds.")
    lines += ["", "The combined eight-seed estimate is descriptive and must not replace the original predeclared three-seed result.", "", "## Source records", ""]
    for path in paths:
        lines.append(f"- `{path}` (SHA-256 `{sha256(path)}`)")
    lines.append("")
    args.output.write_text("\n".join(lines))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
