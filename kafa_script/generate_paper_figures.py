#!/usr/bin/env python3
"""Generate validated PGFPlots coordinates for the paper evidence figure."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from scipy import stats

FAMILIES = ("qwen", "smol", "tiny")
SEVERITIES = ("clean", "medium", "severe")
SEVERITY_SEEDS = (72, 73, 74)
NATURAL = {
    "hausa": {"seeds": (81, 82, 83), "train": 2045, "dev": 290,
              "noise": 0.5036674816625917},
    "yoruba": {"seeds": (84, 85, 86), "train": 1340, "dev": 189,
               "noise": 0.3328358208955224},
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def estimate(values: list[float]) -> tuple[float, float]:
    """Mean and descriptive 95% Student-t half-width."""
    half = (stats.t.ppf(.975, len(values) - 1) * statistics.stdev(values) /
            math.sqrt(len(values)))
    return statistics.mean(values), half


def coordinate(x: int, values: list[float]) -> str:
    mean, half = estimate(values)
    return f"({x},{100 * mean:.4f}) +- (0,{100 * half:.4f})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path,
                        default=Path("kafa_script/analysis_outputs"))
    parser.add_argument("--output", type=Path,
                        default=Path("paper/figures/evidence_data.tex"))
    args = parser.parse_args()
    macros: list[str] = [
        "% Generated from raw JSON by kafa_script/generate_paper_figures.py.",
        "% Values are paired mean gains and descriptive 95% t half-widths.",
    ]

    for family in FAMILIES:
        points = []
        for x, severity in enumerate(SEVERITIES):
            gains = []
            for seed in SEVERITY_SEEDS:
                stem = f"gsm8k_severity_{family}_{{method}}_{severity}_seed{seed}.json"
                naive = load(args.records / stem.format(method="naive"))
                teacher = load(args.records / stem.format(method="teacher"))
                expected = {"seed": seed, "evaluation_examples": 1024,
                            "evaluation_split": "secondary_holdout"}
                for record in (naive, teacher):
                    for key, value in expected.items():
                        if record.get(key) != value:
                            raise ValueError(f"severity {family}/{severity}/{seed}: {key}")
                if naive["evaluation_manifest_sha256"] != teacher["evaluation_manifest_sha256"]:
                    raise ValueError("severity evaluation hash mismatch")
                gains.append(teacher["final"]["exact_match"] -
                             naive["final"]["exact_match"])
            points.append(coordinate(x, gains))
        macros.append(f"\\def\\{family.title()}Severity{{{' '.join(points)}}}")

    for family in FAMILIES:
        points = []
        for x, (corpus, config) in enumerate(NATURAL.items()):
            gains = []
            for seed in config["seeds"]:
                prefix = f"{corpus}_natural_{family}_{{method}}_dev_seed{seed}.json"
                naive = load(args.records / prefix.format(method="naive"))
                coteach = load(args.records / prefix.format(method="coteach"))
                for record in (naive, coteach):
                    expected = {"seed": seed, "evaluation_split": "dev",
                                "train_examples": config["train"],
                                "eval_examples": config["dev"]}
                    for key, value in expected.items():
                        if record.get(key) != value:
                            raise ValueError(f"natural {corpus}/{family}/{seed}: {key}")
                    if not math.isclose(record["measured_training_noise_rate"],
                                        config["noise"], abs_tol=1e-12):
                        raise ValueError("natural noise-rate mismatch")
                for key in ("model_weight_sha256", "training_file_sha256",
                            "clean_training_file_sha256", "evaluation_file_sha256"):
                    if naive[key] != coteach[key]:
                        raise ValueError(f"natural paired {key} mismatch")
                gains.append(coteach["final"]["accuracy"] -
                             naive["final"]["accuracy"])
            points.append(coordinate(x, gains))
        macros.append(f"\\def\\{family.title()}Natural{{{' '.join(points)}}}")

    args.output.write_text("\n".join(macros) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
