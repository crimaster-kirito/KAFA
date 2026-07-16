"""Summarize the prospective IMDb role-reversal server experiment."""

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
    exact_sign_flip_p, fmt_interval, mean_ci, paired_t_p)
from kafa_script.verify_imdb_server_role_reversal import verify_matrix  # noqa: E402


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
        "# IMDb Server Prospective Role-Reversal Experiment",
        "",
        f"Phase: **{phase}**. Records: **{len(records)}**. Protocol SHA-256: "
        f"`{verified['protocol_sha256']}`.",
        f"Summarizer SHA-256: `{sha256_file(Path(__file__))}`. Verifier "
        f"SHA-256: `{sha256_file(ROOT / 'kafa_script/verify_imdb_server_role_reversal.py')}`.",
        "", "## Evidence boundary", "",
        ("The complete 25,000-row IMDb test had no model-evaluation record in "
         "the workspace audit before protocol freeze. Manifest construction "
         "hashed and counted raw lines without decoding JSON. IMDb train and "
         "the sentiment interface had previously served as off-task inputs, "
         "so this is a prospective untouched-test role-reversal extension, "
         "not dataset-naive or independently preregistered replication."),
        "", "## Arm summaries", "",
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

    if phase == "official_evaluation":
        primary = protocol["analysis"]["single_primary_contrast"]
        left, right = primary["left"], primary["right"]
        differences = [
            records[(seed, left)]["final"]["accuracy"] -
            records[(seed, right)]["final"]["accuracy"]
            for seed in section["seeds"]]
        count_differences = [
            round(records[(seed, left)]["final"]["accuracy"] * 25000) -
            round(records[(seed, right)]["final"]["accuracy"] * 25000)
            for seed in section["seeds"]]
        center, low, high = mean_ci(differences)
        p_value = paired_t_p(differences)
        exact = exact_sign_flip_p(count_differences)
        lines += [
            "", "## Single frozen primary contrast", "",
            "| Contrast | Mean pp [95% CI] | Paired-t p | Exact sign-flip p |",
            "|---|---:|---:|---:|",
            f"| {primary['id']} | {fmt_interval(center, low, high, 100, 2)} | "
            f"{p_value:.6f} | {exact:.6f} |",
        ]
        routed_rows = [records[(seed, left)] for seed in section["seeds"]]
        selected = Counter()
        triggers = 0
        rounds = 0
        for row in routed_rows:
            for round_record in row["round_records"]:
                for client in round_record["client_records"]:
                    rounds += 1
                    triggers += int(client["selection"]["gate_triggered"])
                    selected.update(client["selected_kind_counts_audit_only"])
        clean_fraction = selected["clean"] / sum(selected.values())
        passes_composition = clean_fraction > 103 / 256
        passes_primary = high < 0 and p_value < primary["alpha"]
        lines += [
            "", "## Frozen confirmation gate", "",
            f"- Routing triggered in {triggers}/{rounds} client-rounds.",
            f"- Routed audit clean fraction: {100*clean_fraction:.1f}% versus "
            f"the severe baseline {100*103/256:.1f}%: "
            f"**{'pass' if passes_composition else 'fail'}**.",
            f"- Strictly negative primary interval with two-sided p<.05: "
            f"**{'pass' if passes_primary else 'fail'}**.",
            f"- Overall narrow role-reversal confirmation gate: "
            f"**{'pass' if passes_composition and passes_primary else 'fail'}**.",
            "", "All other arm and metric comparisons are descriptive; no other p-value is used.",
        ]

    routed_rows = [records[(seed, "severe_routed_distillation")]
                   for seed in section["seeds"]]
    selected = Counter()
    triggers = 0
    rounds = 0
    for row in routed_rows:
        for round_record in row["round_records"]:
            for client in round_record["client_records"]:
                rounds += 1
                triggers += int(client["selection"]["gate_triggered"])
                selected.update(client["selected_kind_counts_audit_only"])
    lines += [
        "", "## Mechanism and cost audit", "",
        f"- Routing triggered in {triggers}/{rounds} client-rounds.",
        f"- Aggregate routed selected-kind counts: `{dict(selected)}`.",
        "- The anchor arm omits the complete client path and records zero payload.",
        "- Wall time is scheduler-specific, not isolated deployment throughput.",
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
                        choices=["development", "official_evaluation"])
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(
        "kafa_script/imdb_server_role_reversal_protocol_v1.json"))
    args = parser.parse_args()
    print(summarize(args.phase, args.input_dir, args.output, args.protocol))


if __name__ == "__main__":
    main()
