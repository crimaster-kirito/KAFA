"""Summarize the frozen IMDb matched-K mechanistic addendum."""

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
    fmt_interval, mean_ci, paired_t_p)
from kafa_script.verify_imdb_matched_control_addendum import verify_matrix  # noqa: E402


DISPLAY = {
    "severe_random_k": "Severe random-K",
    "severe_routed_k": "Severe routed-K",
    "oracle_max_purity_k": "Oracle maximum-purity-K",
}


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def summarize(phase: str, input_dir: Path, output: Path,
              protocol_path: Path) -> str:
    verified = verify_matrix(phase, input_dir, protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    section = protocol[phase]
    records, hashes = {}, {}
    for seed in section["seeds"]:
        for method in section["comparators"]:
            path = input_dir / f"seed{seed}_{method}.json"
            records[(seed, method)] = json.loads(path.read_text(encoding="utf-8"))
            hashes[path.name] = sha256_file(path)
    lines = [
        "# IMDb Matched-K Secondary Mechanistic Addendum", "",
        f"Phase: **{phase}**. Records: **{len(records)}**. Protocol SHA-256: "
        f"`{verified['protocol_sha256']}`.", "",
        "This is a prospective secondary mechanistic addendum frozen after "
        "IMDb development results were observed but before any official IMDb "
        "model evaluation in the retained workspace. It does not alter or "
        "confirm the original role-reversal primary contrast.", "",
        "## Arm summaries", "",
        "| Arm | Accuracy mean ± SD | Macro-F1 mean [95% CI] | NLL mean [95% CI] | ECE mean [95% CI] |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in section["comparators"]:
        rows = [records[(seed, method)] for seed in section["seeds"]]
        accuracy = [row["final"]["accuracy"] for row in rows]
        macro = [row["final"]["macro_f1"] for row in rows]
        nll = [row["final"]["canonical_label_nll"] for row in rows]
        ece = [row["final"]["ece"] for row in rows]
        lines.append(
            f"| {DISPLAY[method]} | {100*statistics.mean(accuracy):.2f} ± "
            f"{100*statistics.stdev(accuracy):.2f} | "
            f"{fmt_interval(*mean_ci(macro), 100, 2)} | "
            f"{fmt_interval(*mean_ci(nll), 1, 4)} | "
            f"{fmt_interval(*mean_ci(ece), 1, 4)} |")
    lines += ["", "## Per-seed accuracy (%)", "",
              "| Seed | Random-K | Routed-K | Maximum-purity-K |",
              "|---:|---:|---:|---:|"]
    for seed in section["seeds"]:
        vals = [100 * records[(seed, method)]["final"]["accuracy"]
                for method in section["comparators"]]
        lines.append(f"| {seed} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")

    if phase == "official_evaluation":
        contrasts = protocol["analysis"]["secondary_holm_family"]
        rows, raw_p = [], []
        for contrast in contrasts:
            differences = [
                records[(seed, contrast["left"])]["final"]["accuracy"] -
                records[(seed, contrast["right"])]["final"]["accuracy"]
                for seed in section["seeds"]]
            rows.append((contrast, differences, mean_ci(differences)))
            raw_p.append(paired_t_p(differences))
        adjusted = holm_adjust(raw_p)
        lines += ["", "## Frozen secondary Holm family", "",
                  "| Contrast | Mean pp [95% CI] | Raw paired-t p | Holm p |",
                  "|---|---:|---:|---:|"]
        for (contrast, _differences, interval), raw, adj in zip(rows, raw_p, adjusted):
            lines.append(
                f"| {contrast['id']} | {fmt_interval(*interval, 100, 2)} | "
                f"{raw:.6f} | {adj:.6f} |")
        lines += ["", "These two tests are secondary and are reported in parallel with, not substituted for, the original primary analysis."]

    aggregate = {method: Counter() for method in section["comparators"]}
    for method in section["comparators"]:
        for seed in section["seeds"]:
            for rr in records[(seed, method)]["round_records"]:
                for client in rr["client_records"]:
                    aggregate[method].update(client["selected_kind_counts_audit_only"])
    lines += ["", "## Mechanism audit", ""]
    for method in section["comparators"]:
        counts = aggregate[method]
        lines.append(f"- {DISPLAY[method]} selected-kind counts: `{dict(counts)}`.")
    lines += [
        "- Every arm selects exactly 128 examples per client-round and uses the same 40 client steps.",
        "- Random-K uses the known task interface to exclude off-task rows and balances claimed labels; it isolates ranking quality only within the target-eligible pool.",
        "- Maximum-purity-K is not clean-only: it contains every 103 clean row and 25 deterministically chosen conflict rows per client-round.",
        "", "## Raw artifact hashes", ""]
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
    parser.add_argument("--phase", choices=["development", "official_evaluation"], required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(
        "kafa_script/imdb_matched_control_addendum_protocol_v1.json"))
    args = parser.parse_args()
    print(summarize(args.phase, args.input_dir, args.output, args.protocol))


if __name__ == "__main__":
    main()
