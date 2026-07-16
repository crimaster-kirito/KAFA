#!/usr/bin/env python3
"""Validate and summarize the frozen Hausa natural-noise development matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from scipy import stats

FAMILIES = ("qwen", "smol", "tiny")
CONFIGS = {
    "hausa": {
        "seeds": (81, 82, 83), "train": 2045, "dev": 290,
        "noise": 0.5036674816625917,
        "protocol": "kafa_script/hausa_natural_noise_development_protocol_v1.json",
        "name": "Hausa", "headlines": "2,045", "noise_percent": "50.37%",
        "data_dir": ".cache/data/natural_noise/hausa/COMPRISE_Data09_H-VOA-T opics_V1.0",
    },
    "yoruba": {
        "seeds": (84, 85, 86), "train": 1340, "dev": 189,
        "noise": 0.3328358208955224,
        "protocol": "kafa_script/yoruba_natural_noise_development_protocol_v1.json",
        "name": "Yoruba", "headlines": "1,340", "noise_percent": "33.28%",
        "data_dir": ".cache/data/natural_noise/yoruba/COMPRISE_Data13_Y-BBCTopics_V1.0",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    half = (stats.t.ppf(.975, len(values) - 1) * statistics.stdev(values) /
            math.sqrt(len(values)))
    return mean, half


def paired_greater_p(candidate, baseline) -> float:
    differences = [candidate_value - baseline_value
                   for candidate_value, baseline_value in zip(candidate, baseline)]
    if math.isclose(max(differences), min(differences),
                    rel_tol=0.0, abs_tol=1e-12):
        # Zero empirical seed variance makes the paired t test undefined.
        # Use a conservative value instead of reporting false certainty.
        return 1.0
    return float(stats.ttest_rel(candidate, baseline, alternative="greater").pvalue)


def holm(pvalues):
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path,
                        default=Path("kafa_script/analysis_outputs"))
    parser.add_argument("--corpus", choices=tuple(CONFIGS), default="hausa")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = CONFIGS[args.corpus]
    if args.protocol is None:
        args.protocol = Path(config["protocol"])
    data_dir = Path(config["data_dir"])
    dev_sha256 = sha256(data_dir / "dev.tsv")
    test_sha256 = sha256(data_dir / "test.tsv")
    if dev_sha256 == test_sha256:
        raise ValueError("development and test source hashes unexpectedly match")
    forbidden_test_records = list(args.directory.glob(
        f"{args.corpus}_natural_*_test_seed*.json"))
    if forbidden_test_records:
        raise ValueError(f"natural-noise test records exist: {forbidden_test_records}")
    records, paths = {}, []
    for family in FAMILIES:
        for method in ("naive", "coteach"):
            for seed in config["seeds"]:
                path = args.directory / f"{args.corpus}_natural_{family}_{method}_dev_seed{seed}.json"
                record = json.loads(path.read_text())
                expected = {"evidence_label": "benchmark_dev", "seed": seed,
                            "evaluation_split": "dev", "train_examples": config["train"],
                            "eval_examples": config["dev"], "steps": 256, "batch_size": 8}
                for key, value in expected.items():
                    if record.get(key) != value:
                        raise ValueError(f"{path}: {key} is not {value!r}")
                if not math.isclose(record["measured_training_noise_rate"],
                                    config["noise"], abs_tol=1e-12):
                    raise ValueError(f"{path}: natural-noise rate mismatch")
                if record.get("evaluation_file_sha256") != dev_sha256:
                    raise ValueError(f"{path}: evaluation hash is not the source dev.tsv")
                if record.get("evaluation_file_sha256") == test_sha256:
                    raise ValueError(f"{path}: clean test split was evaluated")
                records[(family, method, seed)] = record
                paths.append(path)
    for family in FAMILIES:
        for seed in config["seeds"]:
            naive = records[(family, "naive", seed)]
            coteach = records[(family, "coteach", seed)]
            for key in ("model_weight_sha256", "training_file_sha256",
                        "clean_training_file_sha256", "evaluation_file_sha256"):
                if naive[key] != coteach[key]:
                    raise ValueError(f"{family}/{seed}: {key} mismatch")

    def values(family, method, metric):
        return [records[(family, method, seed)]["final"][metric]
                for seed in config["seeds"]]

    deltas, raw = {}, {}
    for family in FAMILIES:
        naive = values(family, "naive", "accuracy")
        coteach = values(family, "coteach", "accuracy")
        deltas[family] = [candidate - baseline
                          for candidate, baseline in zip(coteach, naive)]
        raw[family] = paired_greater_p(coteach, naive)
    adjusted = holm(raw)
    gate = (all(statistics.mean(deltas[family]) >= 0 for family in FAMILIES)
            and any(adjusted[family] < .05 for family in FAMILIES))

    lines = [
        f"# {config['name']} natural-noise Co-Teaching development result", "",
        "**Status: frozen development-only evidence; the clean test split was not evaluated.**", "",
        "## Verification", "",
        f"- Protocol SHA-256: `{sha256(args.protocol)}`.",
        f"- All 18 records use all {config['headlines']} weakly labelled training headlines and the same {config['dev']}-example clean development split.",
        f"- The measured {config['noise_percent']} natural error rate matches the source data and is given to Co-Teaching as oracle information.",
        f"- Every evaluation hash equals source `dev.tsv` (`{dev_sha256}`), differs from `test.tsv` (`{test_sha256}`), and no natural-noise test record exists.",
        "- Three family-wise accuracy tests are Holm-adjusted together; macro-F1 is secondary.", "",
        "## Results", "",
        "| Client | Naive accuracy | Co-Teaching accuracy | Gain [95% CI] | Naive macro-F1 | Co-Teaching macro-F1 | Raw p | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        naive = values(family, "naive", "accuracy")
        coteach = values(family, "coteach", "accuracy")
        mean, half = interval(deltas[family])
        lines.append(
            f"| {family} | {100*statistics.mean(naive):.2f}% | "
            f"{100*statistics.mean(coteach):.2f}% | {100*mean:+.2f} pp "
            f"[{100*(mean-half):+.2f}, {100*(mean+half):+.2f}] | "
            f"{100*statistics.mean(values(family, 'naive', 'macro_f1')):.2f}% | "
            f"{100*statistics.mean(values(family, 'coteach', 'macro_f1')):.2f}% | "
            f"{raw[family]:.4f} | {adjusted[family]:.4f} |")
    lines += ["", "Per-seed accuracy gains:", ""]
    for family in FAMILIES:
        lines.append(f"- {family}: " + ", ".join(
            f"{100*gain:+.2f} pp" for gain in deltas[family]))

    lines += ["", "## Selection and cost audit", ""]
    for family in FAMILIES:
        clean_fractions, wall_ratios, adapter_ratios = [], [], []
        for seed in config["seeds"]:
            naive = records[(family, "naive", seed)]
            coteach = records[(family, "coteach", seed)]
            for counts in coteach["selected_kind_counts"]:
                clean_fractions.append(counts.get("clean_label_match", 0) /
                                       max(1, sum(counts.values())))
            wall_ratios.append(coteach["train_wall_seconds"] /
                               naive["train_wall_seconds"])
            adapter_ratios.append(coteach["serialized_adapter_bytes"] /
                                  naive["serialized_adapter_bytes"])
        lines.append(
            f"- {family}: selected clean-label agreement {100*statistics.mean(clean_fractions):.1f}%; "
            f"train-wall ratio {statistics.mean(wall_ratios):.2f}x; "
            f"adapter ratio {statistics.mean(adapter_ratios):.2f}x.")
    lines += ["", "## Gate decision", ""]
    if gate:
        lines.append("The predeclared development gate passes. A separately frozen clean-test run is permitted; this report itself remains development evidence.")
    else:
        lines.append("The predeclared development gate fails. The clean test split must not be evaluated, and this result cannot support a natural-noise holdout claim.")
    lines += ["", "This is a compatible Co-Teaching comparison, not Delora: Delora's published pipeline uses GPT-4o for positive-set construction and noisy-sample relabeling.", "", "## Source artifacts", ""]
    for path in sorted(paths):
        lines.append(f"- `{path}` (SHA-256 `{sha256(path)}`)")
    lines.append("")
    args.output.write_text("\n".join(lines))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
