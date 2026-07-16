"""Summarize the frozen AG News post-primary four-arm ablation."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.run_generic_classification_distillation import sha256_file  # noqa: E402
from kafa_script.summarize_trec_server_ablation import (  # noqa: E402
    exact_sign_flip_p, fmt_interval, holm_adjust, mean_ci, paired_t_p)
from kafa_script.verify_agnews_server_ablation import verify_matrix  # noqa: E402


DISPLAY = {
    "public_anchor_only": "Anchor only",
    "clean_unfiltered_distillation": "Clean uniform",
    "severe_unfiltered_distillation": "Severe uniform",
    "severe_routed_distillation": "Severe routed",
}


def summarize(phase: str, input_dir: Path, output: Path,
              protocol_path: Path) -> str:
    verified = verify_matrix(phase, input_dir, protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    section = protocol[phase]
    records, raw_hashes = {}, {}
    for seed in section["seeds"]:
        for method in section["comparators"]:
            path = input_dir / f"seed{seed}_{method}.json"
            records[(seed, method)] = json.loads(path.read_text(encoding="utf-8"))
            raw_hashes[path.name] = sha256_file(path)

    lines = [
        "# AG News Server Post-Primary Four-Arm Ablation",
        "",
        f"Phase: **{phase}**. Records: **{len(records)}**. Protocol SHA-256: "
        f"`{verified['protocol_sha256']}`.",
        f"Summarizer SHA-256: `{sha256_file(Path(__file__))}`. Verifier "
        f"SHA-256: `{sha256_file(ROOT / 'kafa_script/verify_agnews_server_ablation.py')}`.",
        "",
        "## Evidence boundary",
        "",
        ("The AG News primary and official-test seed replication were observed "
         "before this analysis was designed. The 4,000 balanced evaluation "
         "records were selected by a frozen content hash from the AG News "
         "training remainder and excluded from every client pool in this new "
         "runner. A reconstruction audit found that 303 of these IDs had been "
         "allocated to clients in at least one earlier server seed. This is "
         "therefore post-primary, run-disjoint same-task resampling evidence, "
         "not globally untouched data, external replication, or preregistration."),
        "",
        "## Arm summaries",
        "",
        "| Arm | Accuracy mean ± SD | Macro-F1 mean [CI] | Label NLL mean [CI] | ECE mean [CI] | Wall s mean [CI] | Peak GiB mean [CI] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in section["comparators"]:
        rows = [records[(seed, method)] for seed in section["seeds"]]
        accuracy = [row["final"]["accuracy"] for row in rows]
        macro = [row["final"]["macro_f1"] for row in rows]
        nll = [row["final"]["label_nll"] for row in rows]
        ece = [row["final"]["ece"] for row in rows]
        wall = [row["wall_seconds"] for row in rows]
        peak = [row["peak_allocated_gpu_bytes"] / 2**30 for row in rows]
        lines.append(
            f"| {DISPLAY[method]} | {100*statistics.mean(accuracy):.2f} ± "
            f"{100*statistics.stdev(accuracy):.2f} | "
            f"{fmt_interval(*mean_ci(macro), 100, 2)} | "
            f"{fmt_interval(*mean_ci(nll), 1, 4)} | "
            f"{fmt_interval(*mean_ci(ece), 1, 4)} | "
            f"{fmt_interval(*mean_ci(wall), 1, 1)} | "
            f"{fmt_interval(*mean_ci(peak), 1, 2)} |")

    lines += [
        "", "## Per-seed accuracy (%)", "",
        "| Seed | Anchor | Clean uniform | Severe uniform | Severe routed |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed in section["seeds"]:
        values = [100 * records[(seed, method)]["final"]["accuracy"]
                  for method in section["comparators"]]
        lines.append(f"| {seed} | " + " | ".join(
            f"{value:.2f}" for value in values) + " |")

    lines += [
        "", "## Complete per-seed scalar secondary values", "",
        "| Seed | Arm | Macro-F1 | Label NLL | ECE | Tensor bytes | Wall s | Peak GiB |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in section["seeds"]:
        for method in section["comparators"]:
            row = records[(seed, method)]
            lines.append(
                f"| {seed} | {DISPLAY[method]} | "
                f"{100*row['final']['macro_f1']:.2f} | "
                f"{row['final']['label_nll']:.4f} | {row['final']['ece']:.4f} | "
                f"{row['total_client_to_server_payload_bytes']} | "
                f"{row['wall_seconds']:.1f} | "
                f"{row['peak_allocated_gpu_bytes']/2**30:.2f} |")

    if phase == "secondary_holdout":
        analyses = []
        for contrast in protocol["analysis"]["adjacent_contrasts"]:
            left, right = contrast["left"], contrast["right"]
            differences = [
                records[(seed, left)]["final"]["accuracy"] -
                records[(seed, right)]["final"]["accuracy"]
                for seed in section["seeds"]]
            evaluation_n = records[(section["seeds"][0], left)][
                "evaluation_examples"]
            count_differences = [
                round(records[(seed, left)]["final"]["accuracy"] * evaluation_n) -
                round(records[(seed, right)]["final"]["accuracy"] * evaluation_n)
                for seed in section["seeds"]]
            center, low, high = mean_ci(differences)
            analyses.append({
                **contrast,
                "mean": center,
                "low": low,
                "high": high,
                "raw_p": paired_t_p(differences),
                "exact_p": exact_sign_flip_p(count_differences),
            })
        adjusted = holm_adjust([item["raw_p"] for item in analyses])
        for item, value in zip(analyses, adjusted):
            item["holm_p"] = value
        lines += [
            "", "## Frozen adjacent accuracy family", "",
            ("The three adjacent contrasts use paired two-sided Student-t tests "
             "over ten common seeds and one joint Holm correction. Intervals "
             "describe seed variation. Exact sign-flip p-values from integer "
             "correct-count differences are sensitivity analyses."),
            "",
            "| Contrast (left − right) | Mean pp [95% CI] | Raw p | Holm p | Exact sign-flip p |",
            "|---|---:|---:|---:|---:|",
        ]
        for item in analyses:
            lines.append(
                f"| {item['id']} | "
                f"{fmt_interval(item['mean'], item['low'], item['high'], 100, 2)} | "
                f"{item['raw_p']:.6f} | {item['holm_p']:.6f} | "
                f"{item['exact_p']:.6f} |")
        net = [
            records[(seed, "severe_routed_distillation")]["final"]["accuracy"] -
            records[(seed, "public_anchor_only")]["final"]["accuracy"]
            for seed in section["seeds"]]
        lines += [
            "",
            ("The severe-routed minus anchor net accuracy is "
             f"{fmt_interval(*mean_ci(net), 100, 2)} pp. It is descriptive and "
             "is not a fourth test."),
            "", "## Secondary descriptive contrasts", "",
            "| Metric | Clean−anchor | Severe−clean | Routed−severe |",
            "|---|---:|---:|---:|",
        ]
        pairs = [
            ("clean_unfiltered_distillation", "public_anchor_only"),
            ("severe_unfiltered_distillation", "clean_unfiltered_distillation"),
            ("severe_routed_distillation", "severe_unfiltered_distillation"),
        ]
        for metric, scale, digits in (
                ("macro_f1", 100, 2), ("label_nll", 1, 4),
                ("ece", 1, 4)):
            formatted = []
            for left, right in pairs:
                values = [records[(seed, left)]["final"][metric] -
                          records[(seed, right)]["final"][metric]
                          for seed in section["seeds"]]
                formatted.append(fmt_interval(*mean_ci(values), scale, digits))
            lines.append(f"| {metric} | " + " | ".join(formatted) + " |")
        lines += ["", "No secondary p-value or confirmatory secondary claim is used."]

    routed_rows = [records[(seed, "severe_routed_distillation")]
                   for seed in section["seeds"]]
    selected = Counter()
    triggers = 0
    client_rounds = 0
    for row in routed_rows:
        for round_record in row["round_records"]:
            for client in round_record["client_records"]:
                client_rounds += 1
                triggers += int(client["selection"]["gate_triggered"])
                selected.update(client["selected_kind_counts_audit_only"])
    clean_fraction = selected["clean"] / sum(selected.values()) if selected else 0
    distill_payloads = {
        row["total_client_to_server_payload_bytes"]
        for (seed, method), row in records.items()
        if method != "public_anchor_only"
    }
    lines += [
        "", "## Mechanism and cost audit", "",
        f"- Severe routing triggered in {triggers}/{client_rounds} client-rounds.",
        f"- Aggregate routed selected-kind counts: `{dict(selected)}`.",
        f"- Routed selected clean fraction: {100*clean_fraction:.1f}% (audit only).",
        f"- Distillation-arm total serialized tensor bytes per run: `{sorted(distill_payloads)}`.",
        "- The anchor-only arm omits client execution and records zero client payload.",
        "- Wall time is a scheduler-specific timed-region measurement, not deployment throughput.",
        "", "## Raw artifact hashes", "",
    ]
    for name, digest in sorted(raw_hashes.items()):
        lines.append(f"- `{name}`: `{digest}`")
    lines += ["", "## Claim boundary", ""]
    for boundary in protocol["evidence_boundaries"]:
        lines.append(f"- {boundary}")
    text = "\n".join(lines) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
                        choices=["development", "secondary_holdout"])
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=Path(
        "kafa_script/agnews_server_post_primary_ablation_v1.json"))
    args = parser.parse_args()
    print(summarize(args.phase, args.input_dir, args.output, args.protocol))


if __name__ == "__main__":
    main()
