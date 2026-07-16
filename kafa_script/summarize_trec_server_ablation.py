"""Generate a strict report for the post-primary TREC four-arm ablation."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.run_generic_classification_distillation import sha256_file
from kafa_script.verify_trec_server_ablation import verify_matrix


DISPLAY = {
    "public_anchor_only": "Anchor only",
    "clean_unfiltered_distillation": "Clean uniform",
    "severe_unfiltered_distillation": "Severe uniform",
    "severe_routed_distillation": "Severe routed",
}


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    center = statistics.mean(values)
    if len(values) < 2:
        return center, center, center
    half = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return center, center - half, center + half


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values or any(not math.isfinite(value) or not 0 <= value <= 1
                           for value in p_values):
        raise ValueError("Holm correction requires finite p-values in [0, 1]")
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_t_p(differences: list[float]) -> float:
    """Two-sided one-sample t p-value with an explicit all-zero policy."""
    if all(value == 0.0 for value in differences):
        return 1.0
    p_value = float(stats.ttest_1samp(differences, 0).pvalue)
    if not math.isfinite(p_value):
        raise ValueError(
            "paired t p-value is non-finite for a nonzero contrast")
    return p_value


def exact_sign_flip_p(differences: list[float]) -> float:
    observed = abs(statistics.mean(differences))
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        permuted = abs(statistics.mean(
            sign * value for sign, value in zip(signs, differences)))
        if permuted >= observed - 1e-15:
            extreme += 1
    return extreme / (2 ** len(differences))


def fmt_interval(center: float, low: float, high: float,
                 scale: float = 1.0, digits: int = 4) -> str:
    return (f"{center*scale:+.{digits}f} "
            f"[{low*scale:+.{digits}f}, {high*scale:+.{digits}f}]")


def summarize(phase: str, input_dir: Path, output: Path,
              protocol_path: Path) -> str:
    verification_phase = ("development" if phase == "development" else
                          "official_evaluation")
    verified = verify_matrix(verification_phase, input_dir, protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    section = protocol[verification_phase]
    records = {}
    raw_hashes = {}
    for seed in section["seeds"]:
        for method in section["comparators"]:
            path = input_dir / f"seed{seed}_{method}.json"
            records[(seed, method)] = json.loads(
                path.read_text(encoding="utf-8"))
            raw_hashes[path.name] = sha256_file(path)

    lines = [
        "# TREC Server Post-Primary Four-Arm Ablation",
        "",
        f"Phase: **{phase}**. Records: **{len(records)}**. Protocol SHA-256: "
        f"`{verified['protocol_sha256']}`.",
        f"Summarizer SHA-256: `{sha256_file(Path(__file__))}`. Verifier "
        f"SHA-256: `{sha256_file(ROOT / 'kafa_script/verify_trec_server_ablation.py')}`.",
        "",
        "## Evidence boundary",
        "",
        ("The complete TREC official test and primary 5-by-2 result were "
         "observed before this ablation was designed. The official phase "
         "reuses that test with ten new jointly assigned seeds. It is a "
         "post-primary mechanistic analysis, not a new holdout, independent "
         "replication, preregistration, or repair of the primary inference."),
        "",
        "## Arm summaries",
        "",
        "| Arm | Accuracy mean ± SD | Macro-F1 mean [CI] | Class NLL mean [CI] | ECE mean [CI] | Wall s mean [CI] | Peak GiB mean [CI] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in section["comparators"]:
        rows = [records[(seed, method)] for seed in section["seeds"]]
        accuracy = [row["final"]["accuracy"] for row in rows]
        macro = [row["final"]["macro_f1"] for row in rows]
        nll = [row["final"]["canonical_label_nll"] for row in rows]
        ece = [row["final"]["ece"] for row in rows]
        wall = [row["wall_seconds"] for row in rows]
        peak = [row["peak_allocated_gpu_bytes"] / (1024**3) for row in rows]
        lines.append(
            f"| {DISPLAY[method]} | {100*statistics.mean(accuracy):.2f} ± "
            f"{100*statistics.stdev(accuracy):.2f} | "
            f"{fmt_interval(*mean_ci(macro), 100, 2)} | "
            f"{fmt_interval(*mean_ci(nll), 1, 4)} | "
            f"{fmt_interval(*mean_ci(ece), 1, 4)} | "
            f"{fmt_interval(*mean_ci(wall), 1, 1)} | "
            f"{fmt_interval(*mean_ci(peak), 1, 2)} |")

    lines += ["", "## Per-seed accuracy (%)", "",
              "| Seed | Anchor | Clean uniform | Severe uniform | Severe routed |",
              "|---:|---:|---:|---:|---:|"]
    for seed in section["seeds"]:
        values = [100 * records[(seed, method)]["final"]["accuracy"]
                  for method in section["comparators"]]
        lines.append(f"| {seed} | " + " | ".join(f"{value:.2f}" for value in values) + " |")

    lines += [
        "",
        "## Complete per-seed scalar secondary values",
        "",
        "| Seed | Arm | Macro-F1 | Class NLL | ECE | Tensor bytes | Wall s | Peak GiB |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in section["seeds"]:
        for method in section["comparators"]:
            row = records[(seed, method)]
            lines.append(
                f"| {seed} | {DISPLAY[method]} | {100*row['final']['macro_f1']:.2f} | "
                f"{row['final']['canonical_label_nll']:.4f} | "
                f"{row['final']['ece']:.4f} | "
                f"{row['total_client_to_server_payload_bytes']} | "
                f"{row['wall_seconds']:.1f} | "
                f"{row['peak_allocated_gpu_bytes']/(1024**3):.2f} |")

    if phase == "official":
        contrasts = protocol["official_evaluation"]["primary_contrasts"]
        analyses = []
        for contrast in contrasts:
            left, right = contrast["left"], contrast["right"]
            differences = [
                records[(seed, left)]["final"]["accuracy"] -
                records[(seed, right)]["final"]["accuracy"]
                for seed in section["seeds"]]
            count_differences = []
            for seed in section["seeds"]:
                left_matrix = records[(seed, left)]["final"]["confusion_matrix"]
                right_matrix = records[(seed, right)]["final"]["confusion_matrix"]
                count_differences.append(
                    sum(left_matrix[index][index] for index in range(6)) -
                    sum(right_matrix[index][index] for index in range(6)))
            center, low, high = mean_ci(differences)
            p_value = paired_t_p(differences)
            analyses.append({
                **contrast,
                "differences": differences,
                "mean": center,
                "ci_low": low,
                "ci_high": high,
                "p_raw": p_value,
                "p_sign_flip": exact_sign_flip_p(count_differences),
            })
        adjusted = holm_adjust([item["p_raw"] for item in analyses])
        for item, value in zip(analyses, adjusted):
            item["p_holm"] = value
        lines += [
            "",
            "## Frozen primary accuracy family",
            "",
            ("All three adjacent contrasts use paired two-sided Student-t "
             "tests over the same ten seeds and one joint Holm correction. "
             "Intervals describe seed variation. Exact two-sided sign-flip "
             "p-values are sensitivity analyses, not the decision gate."),
            "",
            "| Contrast (left − right) | Mean pp [95% CI] | Raw p | Holm p | Exact sign-flip p |",
            "|---|---:|---:|---:|---:|",
        ]
        for item in analyses:
            lines.append(
                f"| {item['name']} | "
                f"{fmt_interval(item['mean'], item['ci_low'], item['ci_high'], 100, 2)} | "
                f"{item['p_raw']:.6f} | {item['p_holm']:.6f} | "
                f"{item['p_sign_flip']:.6f} |")
        net = [
            records[(seed, "severe_routed_distillation")]["final"]["accuracy"] -
            records[(seed, "public_anchor_only")]["final"]["accuracy"]
            for seed in section["seeds"]]
        net_center, net_low, net_high = mean_ci(net)
        lines += [
            "",
            ("The descriptive net severe-routed minus anchor effect is "
             f"{fmt_interval(net_center, net_low, net_high, 100, 2)} pp. "
             "It is not a fourth primary test."),
            "",
            "## Secondary descriptive contrasts",
            "",
            "| Metric | Clean−anchor | Severe−clean | Routed−severe |",
            "|---|---:|---:|---:|",
        ]
        for metric, scale in (("macro_f1", 100),
                              ("canonical_label_nll", 1), ("ece", 1)):
            cells = []
            for contrast in contrasts:
                values = [
                    records[(seed, contrast["left"])]["final"][metric] -
                    records[(seed, contrast["right"])]["final"][metric]
                    for seed in section["seeds"]]
                cells.append(fmt_interval(*mean_ci(values), scale=scale,
                                          digits=2 if scale == 100 else 4))
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "No secondary p-value or confirmatory secondary claim is used.",
        ]

    routed_rows = [records[(seed, "severe_routed_distillation")]
                   for seed in section["seeds"]]
    triggers = sum(
        client["selection"].get("gate_triggered", False)
        for row in routed_rows for round_record in row["round_records"]
        for client in round_record["client_records"])
    client_rounds = len(routed_rows) * 2 * 3
    selected_counts = Counter(
        kind
        for row in routed_rows for round_record in row["round_records"]
        for client in round_record["client_records"]
        for kind, count in client["selected_kind_counts_audit_only"].items()
        for _ in range(count))
    lines += [
        "",
        "## Mechanism and cost audit",
        "",
        f"- Severe routing triggered in {triggers}/{client_rounds} client-rounds.",
        f"- Aggregate routed selected-kind counts: `{dict(sorted(selected_counts.items()))}`.",
        "- Every distillation arm records 19,830 bytes of six serialized probability tensors per run; transport metadata is excluded.",
        "- Anchor-only omits the client path. Severe uniform versus severe routed is the only comparison matching client models, mixture, optimizer steps, server work, and seed.",
        "- Wall times are scheduler-specific timed-region measurements, not isolated deployment throughput.",
        "",
        "### Routed selection composition by seed",
        "",
        "| Seed | Triggered/client-rounds | Selected clean | Selected conflict | Selected off-task |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed in section["seeds"]:
        row = records[(seed, "severe_routed_distillation")]
        clients = [client for round_record in row["round_records"]
                   for client in round_record["client_records"]]
        seed_triggers = sum(client["selection"].get("gate_triggered", False)
                            for client in clients)
        counts = Counter()
        for client in clients:
            counts.update(client["selected_kind_counts_audit_only"])
        lines.append(
            f"| {seed} | {seed_triggers}/6 | {counts['clean']} | "
            f"{counts['conflict']} | {counts['off_task']} |")
    lines += [
        "",
        "## Complete per-class recall means and intervals",
        "",
        "| Arm | ABBR | ENTY | DESC | HUM | LOC | NUM |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in section["comparators"]:
        recalls = []
        for label in range(6):
            values = [records[(seed, method)]["final"]["per_class_recall"][label]
                      for seed in section["seeds"]]
            recalls.append(fmt_interval(*mean_ci(values), 100, 2))
        lines.append(f"| {DISPLAY[method]} | " +
                     " | ".join(recalls) + " |")

    lines += ["", "## Raw artifact hashes", ""]
    for name, digest in sorted(raw_hashes.items()):
        lines.append(f"- `{name}`: `{digest}`")
    lines += ["", "## Claim boundary", "",
              *[f"- {boundary}" for boundary in protocol["evidence_boundaries"]], ""]
    text = "\n".join(lines)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["development", "official"], required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(
        "kafa_script/trec_server_post_primary_ablation_v1.json"))
    args = parser.parse_args()
    print(summarize(args.phase, args.input_dir, args.output, args.protocol))


if __name__ == "__main__":
    main()
