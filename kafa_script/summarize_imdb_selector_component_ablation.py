"""Summarize post-hoc IMDb selector-component ablations."""

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
from kafa_script.summarize_imdb_matched_control_addendum import holm_adjust  # noqa: E402
from kafa_script.summarize_trec_server_ablation import (  # noqa: E402
    exact_sign_flip_p,
    fmt_interval,
    mean_ci,
    paired_t_p,
)
from kafa_script.verify_imdb_matched_control_addendum import (  # noqa: E402
    verify_matrix as verify_matched_matrix,
)
from kafa_script.verify_imdb_selector_component_ablation import (  # noqa: E402
    verify_matrix as verify_component_matrix,
)


DISPLAY = {
    "severe_routed_k": "Full routed-K",
    "gradient_only_k": "Gradient-only-K",
    "loss_only_k": "Loss-only-K",
    "margin_only_k": "Margin-only-K",
}


def _load_records(input_dir: Path, seeds: list[int],
                  methods: list[str]) -> dict[tuple[int, str], dict]:
    records = {}
    for seed in seeds:
        for method in methods:
            path = input_dir / f"seed{seed}_{method}.json"
            records[(seed, method)] = json.loads(path.read_text(encoding="utf-8"))
    return records


def _composition(records: dict[tuple[int, str], dict],
                 seeds: list[int], methods: list[str]) -> dict[str, Counter]:
    aggregate = {method: Counter() for method in methods}
    for method in methods:
        for seed in seeds:
            for rr in records[(seed, method)]["round_records"]:
                for client in rr["client_records"]:
                    aggregate[method].update(
                        client["selected_kind_counts_audit_only"])
    return aggregate


def summarize(input_dir: Path, matched_dir: Path, output: Path,
              protocol_path: Path, matched_protocol_path: Path) -> str:
    verified = verify_component_matrix(input_dir, protocol_path)
    matched_verified = verify_matched_matrix(
        "official_evaluation", matched_dir, matched_protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    seeds = protocol["official_evaluation"]["seeds"]
    component_methods = protocol["official_evaluation"]["comparators"]
    component_records = _load_records(input_dir, seeds, component_methods)
    matched_records = _load_records(matched_dir, seeds, ["severe_routed_k"])
    all_records = {**component_records, **matched_records}
    all_methods = ["severe_routed_k", *component_methods]

    hashes = {
        f"component/{path.name}": sha256_file(path)
        for path in input_dir.glob("seed*.json")
    }
    hashes.update({
        f"matched_reference/{path.name}": sha256_file(path)
        for path in matched_dir.glob("seed*_severe_routed_k.json")
    })

    lines = [
        "# IMDb Selector-Component Post-hoc Ablation",
        "",
        f"Records: **{verified['records']}** new component records plus "
        f"**10** existing full-routed matched-K reference records.",
        f"Selector protocol SHA-256: `{sha256_file(protocol_path)}`.",
        f"Matched-K verifier protocol SHA-256: `{matched_verified['protocol_sha256']}`.",
        f"Summarizer SHA-256: `{sha256_file(Path(__file__))}`.",
        "",
        "## Evidence boundary",
        "",
        "This is a post-hoc reused-test secondary diagnostic designed after "
        "the official IMDb primary and matched-K results were observed. It "
        "can diagnose which score component carries ranking signal in this "
        "setting, but it cannot alter the original IMDb primary gate, replace "
        "the matched-K Holm family, or establish a general selector.",
        "",
        "## Arm summaries",
        "",
        "| Arm | Accuracy mean ± SD | Macro-F1 mean [95% CI] | Class NLL mean [95% CI] | ECE mean [95% CI] | Wall s mean [95% CI] | Peak GiB mean [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in all_methods:
        rows = [all_records[(seed, method)] for seed in seeds]
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
        "",
        "## Per-seed accuracy (%)",
        "",
        "| Seed | Full routed-K | Gradient-only-K | Loss-only-K | Margin-only-K |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed in seeds:
        values = [100 * all_records[(seed, method)]["final"]["accuracy"]
                  for method in all_methods]
        lines.append(f"| {seed} | " + " | ".join(
            f"{value:.2f}" for value in values) + " |")

    contrasts = [
        ("full_routed_k_minus_gradient_only_k",
         "severe_routed_k", "gradient_only_k"),
        ("full_routed_k_minus_loss_only_k",
         "severe_routed_k", "loss_only_k"),
        ("full_routed_k_minus_margin_only_k",
         "severe_routed_k", "margin_only_k"),
    ]
    rows, raw_p = [], []
    for contrast_id, left, right in contrasts:
        differences = [
            all_records[(seed, left)]["final"]["accuracy"] -
            all_records[(seed, right)]["final"]["accuracy"]
            for seed in seeds
        ]
        count_differences = [
            round(all_records[(seed, left)]["final"]["accuracy"] * 25000) -
            round(all_records[(seed, right)]["final"]["accuracy"] * 25000)
            for seed in seeds
        ]
        rows.append((contrast_id, differences, count_differences,
                     mean_ci(differences)))
        raw_p.append(paired_t_p(differences))
    adjusted = holm_adjust(raw_p)
    lines += [
        "",
        "## Post-hoc selector-component Holm family",
        "",
        "| Contrast | Mean pp [95% CI] | Raw paired-t p | Holm p | Exact sign-flip p |",
        "|---|---:|---:|---:|---:|",
    ]
    for (contrast_id, _diff, count_diff, interval), raw, adj in zip(
            rows, raw_p, adjusted):
        lines.append(
            f"| {contrast_id} | {fmt_interval(*interval, 100, 2)} | "
            f"{raw:.6f} | {adj:.6f} | {exact_sign_flip_p(count_diff):.6f} |")

    composition = _composition(all_records, seeds, all_methods)
    lines += ["", "## Mechanism audit", ""]
    for method in all_methods:
        counts = composition[method]
        total = sum(counts.values())
        clean = 100 * counts["clean"] / total if total else float("nan")
        conflict = 100 * counts["conflict"] / total if total else float("nan")
        off_task = 100 * counts.get("off_task", 0) / total if total else float("nan")
        lines.append(
            f"- {DISPLAY[method]} selected-kind counts: `{dict(counts)}` "
            f"({clean:.1f}% clean, {conflict:.1f}% conflict, "
            f"{off_task:.1f}% off-task).")
    lines += [
        "- Each component arm selects exactly 128 target-eligible examples per "
        "client-round with 64 examples per claimed label.",
        "- Clean/conflict identity is audit-only and is not used by any "
        "component selector.",
        "",
        "## Raw artifact hashes",
        "",
    ]
    for name, digest in sorted(hashes.items()):
        lines.append(f"- `{name}`: `{digest}`")
    lines += ["", "## Claim boundary", ""]
    lines.extend(f"- {item}" for item in protocol["evidence_boundaries"])
    text = "\n".join(lines) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--matched-dir", type=Path, default=Path(
        "kafa_script/analysis_outputs/imdb_matched_control_addendum/"
        "official_evaluation"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(
        "kafa_script/imdb_selector_component_ablation_protocol_v1.json"))
    parser.add_argument("--matched-protocol", type=Path, default=Path(
        "kafa_script/imdb_matched_control_addendum_protocol_v1.json"))
    args = parser.parse_args()
    print(summarize(args.input_dir, args.matched_dir, args.output,
                    args.protocol, args.matched_protocol))


if __name__ == "__main__":
    main()
