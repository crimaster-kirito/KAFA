"""Generate submission assets from the complete, verified IMDb matrices.

This module is deliberately fail-closed.  It creates no output directory until
both frozen official matrices and the cross-matrix routed-arm equivalence check
have succeeded.  The emitted JSON is the machine-readable source of the TeX
macros, tables, and vector figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.summarize_imdb_matched_control_addendum import holm_adjust
from kafa_script.summarize_trec_server_ablation import (
    exact_sign_flip_p,
    mean_ci,
    paired_t_p,
)
from kafa_script import verify_imdb_matched_control_addendum as matched_verifier
from kafa_script import verify_imdb_routed_equivalence as equivalence_verifier
from kafa_script import verify_imdb_server_role_reversal as original_verifier


SEEDS = tuple(range(321, 331))
ORIGINAL_METHODS = (
    "public_anchor_only",
    "clean_unfiltered_distillation",
    "severe_unfiltered_distillation",
    "severe_routed_distillation",
)
MATCHED_METHODS = (
    "severe_random_k",
    "severe_routed_k",
    "oracle_max_purity_k",
)
ORIGINAL_DISPLAY = {
    "public_anchor_only": "Anchor only",
    "clean_unfiltered_distillation": "Clean uniform",
    "severe_unfiltered_distillation": "Severe uniform",
    "severe_routed_distillation": "Severe routed",
}
MATCHED_DISPLAY = {
    "severe_random_k": r"Random-$K$",
    "severe_routed_k": r"Routed-$K$",
    "oracle_max_purity_k": r"Maximum-purity-$K$",
}
METRICS = ("accuracy", "macro_f1", "canonical_label_nll", "ece")
ASSET_NAMES = (
    "imdb_official_macros.tex",
    "imdb_official_tables.tex",
    "imdb_official_figure.tex",
    "imdb_official_figure.pdf",
    "imdb_official_results_block.tex",
    "imdb_official_assets.json",
)
PRIMARY_CLEAN_FRACTION_BASELINE = 103 / 256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _expected_names(methods: tuple[str, ...]) -> set[str]:
    return {f"seed{seed}_{method}.json" for seed in SEEDS for method in methods}


def _check_exact_matrix(directory: Path, methods: tuple[str, ...],
                        expected_count: int, label: str) -> None:
    observed = {path.name for path in directory.glob("seed*.json")}
    expected = _expected_names(methods)
    _require(len(observed) == expected_count and observed == expected,
             f"{label} must contain exactly {expected_count} frozen records; "
             f"missing={sorted(expected - observed)}, "
             f"extra={sorted(observed - expected)}")


def _load_records(directory: Path, methods: tuple[str, ...]) -> dict:
    records = {}
    for seed in SEEDS:
        for method in methods:
            path = directory / f"seed{seed}_{method}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            _require(record.get("seed") == seed and record.get("method") == method,
                     f"identity mismatch in {path.name}")
            final = record.get("final", {})
            for metric in METRICS:
                value = float(final.get(metric, math.nan))
                _require(math.isfinite(value),
                         f"non-finite or absent {metric} in {path.name}")
            records[(seed, method)] = record
    return records


def _summary(records: dict, methods: tuple[str, ...]) -> dict:
    result = {}
    for method in methods:
        result[method] = {}
        for metric in METRICS:
            values = [float(records[(seed, method)]["final"][metric])
                      for seed in SEEDS]
            result[method][metric] = {
                "mean": statistics.mean(values),
                "sd": statistics.stdev(values),
                "values": values,
            }
    return result


def _contrast(records: dict, left: str, right: str,
              exact_correct_counts: bool = False) -> dict:
    differences = [
        float(records[(seed, left)]["final"]["accuracy"])
        - float(records[(seed, right)]["final"]["accuracy"])
        for seed in SEEDS
    ]
    center, low, high = mean_ci(differences)
    result = {
        "left": left,
        "right": right,
        "differences": differences,
        "mean": center,
        "ci95_low": low,
        "ci95_high": high,
        "paired_t_two_sided_p": paired_t_p(differences),
    }
    if exact_correct_counts:
        integer_differences = []
        for seed in SEEDS:
            left_record, right_record = records[(seed, left)], records[(seed, right)]
            left_n = int(left_record.get("evaluation_examples", 25000))
            right_n = int(right_record.get("evaluation_examples", 25000))
            _require(left_n == right_n,
                     f"evaluation count differs between primary arms at seed {seed}")
            integer_differences.append(
                round(float(left_record["final"]["accuracy"]) * left_n)
                - round(float(right_record["final"]["accuracy"]) * right_n))
        result["integer_correct_count_differences"] = integer_differences
        result["exact_two_sided_sign_flip_p"] = exact_sign_flip_p(
            integer_differences)
    return result


def _composition(records: dict, methods: tuple[str, ...]) -> dict:
    result = {}
    for method in methods:
        counts: Counter[str] = Counter()
        for seed in SEEDS:
            for round_record in records[(seed, method)].get("round_records", []):
                for client in round_record.get("client_records", []):
                    counts.update(client.get("selected_kind_counts_audit_only", {}))
        total = sum(counts.values())
        result[method] = {
            "counts": dict(sorted(counts.items())),
            "fractions": {
                kind: (counts[kind] / total if total else None)
                for kind in ("clean", "conflict", "off_task")
            },
            "total_selected": total,
        }
    return result


def compute_statistics(original_records: dict, matched_records: dict) -> dict:
    original_summary = _summary(original_records, ORIGINAL_METHODS)
    matched_summary = _summary(matched_records, MATCHED_METHODS)
    primary = _contrast(
        original_records, "severe_routed_distillation",
        "severe_unfiltered_distillation", exact_correct_counts=True)
    matched_contrasts = [
        _contrast(matched_records, "severe_routed_k", "severe_random_k"),
        _contrast(matched_records, "oracle_max_purity_k", "severe_routed_k"),
    ]
    adjusted = holm_adjust(
        [item["paired_t_two_sided_p"] for item in matched_contrasts])
    for item, value in zip(matched_contrasts, adjusted):
        item["holm_adjusted_p"] = value
    original_composition = _composition(original_records, ORIGINAL_METHODS)
    matched_composition = _composition(matched_records, MATCHED_METHODS)
    routed_clean_fraction = original_composition[
        "severe_routed_distillation"]["fractions"]["clean"]
    _require(routed_clean_fraction is not None,
             "primary routed arm has no auditable selected examples")
    clean_fraction_pass = (
        routed_clean_fraction > PRIMARY_CLEAN_FRACTION_BASELINE)
    negative_ci_pass = bool(primary["ci95_high"] < 0)
    paired_p_pass = bool(primary["paired_t_two_sided_p"] < 0.05)
    primary_gate = {
        "passes": clean_fraction_pass and negative_ci_pass and paired_p_pass,
        "routed_clean_fraction": routed_clean_fraction,
        "protocol_clean_fraction_baseline": PRIMARY_CLEAN_FRACTION_BASELINE,
        "clean_fraction_strictly_above_baseline": clean_fraction_pass,
        "ci_strictly_negative": negative_ci_pass,
        "paired_t_two_sided_p_below_0_05": paired_p_pass,
    }
    return {
        "schema_version": "imdb_official_paper_assets_v1",
        "evidence_scope": {
            "phase": "official_evaluation",
            "seeds": list(SEEDS),
            "original_records": 40,
            "matched_records": 30,
            "matched_family_classification": "secondary",
        },
        "original_arm_summaries": original_summary,
        "matched_arm_summaries": matched_summary,
        "primary_contrast": primary,
        "primary_confirmation_gate": primary_gate,
        "matched_secondary_holm_family": matched_contrasts,
        "original_composition": original_composition,
        "matched_composition": matched_composition,
    }


def _tex_number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _p_text(value: float) -> str:
    """Return TeX-safe content without opening or closing math mode."""
    return r"\ensuremath{<0.0001}" if value < 0.0001 else _tex_number(value, 4)


def _render_macros(stats: dict) -> str:
    p = stats["primary_contrast"]
    m1, m2 = stats["matched_secondary_holm_family"]
    original = stats["original_arm_summaries"]
    original_composition = stats["original_composition"]
    matched_composition = stats["matched_composition"]
    gate = stats["primary_confirmation_gate"]
    macros = {
        "IMDbPrimaryGateStatus": "passes" if gate["passes"] else "fails",
        "IMDbPrimaryGatePassFlag": "1" if gate["passes"] else "0",
        "IMDbArmAAccuracyPct": _tex_number(
            100 * original["public_anchor_only"]["accuracy"]["mean"], 2),
        "IMDbArmBAccuracyPct": _tex_number(
            100 * original["clean_unfiltered_distillation"]["accuracy"]["mean"], 2),
        "IMDbArmCAccuracyPct": _tex_number(
            100 * original["severe_unfiltered_distillation"]["accuracy"]["mean"], 2),
        "IMDbArmDAccuracyPct": _tex_number(
            100 * original["severe_routed_distillation"]["accuracy"]["mean"], 2),
        "IMDbSevereBaselineCleanPct": _tex_number(
            100 * gate["protocol_clean_fraction_baseline"], 2),
        "IMDbSevereUniformCleanPct": _tex_number(100 * original_composition[
            "severe_unfiltered_distillation"]["fractions"]["clean"], 2),
        "IMDbSevereRoutedCleanPct": _tex_number(
            100 * gate["routed_clean_fraction"], 2),
        "IMDbPrimaryDeltaPP": _tex_number(100 * p["mean"], 2),
        "IMDbPrimaryCILowPP": _tex_number(100 * p["ci95_low"], 2),
        "IMDbPrimaryCIHighPP": _tex_number(100 * p["ci95_high"], 2),
        "IMDbPrimaryPairedTP": _p_text(p["paired_t_two_sided_p"]),
        "IMDbPrimarySignFlipP": _p_text(p["exact_two_sided_sign_flip_p"]),
        "IMDbMatchedRandomCleanPct": _tex_number(100 * matched_composition[
            "severe_random_k"]["fractions"]["clean"], 2),
        "IMDbMatchedRoutedCleanPct": _tex_number(100 * matched_composition[
            "severe_routed_k"]["fractions"]["clean"], 2),
        "IMDbMatchedMaxPurityCleanPct": _tex_number(100 * matched_composition[
            "oracle_max_purity_k"]["fractions"]["clean"], 2),
        "IMDbRoutedRandomDeltaPP": _tex_number(100 * m1["mean"], 2),
        "IMDbRoutedRandomCILowPP": _tex_number(100 * m1["ci95_low"], 2),
        "IMDbRoutedRandomCIHighPP": _tex_number(100 * m1["ci95_high"], 2),
        "IMDbRoutedRandomRawP": _p_text(m1["paired_t_two_sided_p"]),
        "IMDbRoutedRandomHolmP": _p_text(m1["holm_adjusted_p"]),
        "IMDbMaxPurityRoutedDeltaPP": _tex_number(100 * m2["mean"], 2),
        "IMDbMaxPurityRoutedCILowPP": _tex_number(100 * m2["ci95_low"], 2),
        "IMDbMaxPurityRoutedCIHighPP": _tex_number(100 * m2["ci95_high"], 2),
        "IMDbMaxPurityRoutedRawP": _p_text(m2["paired_t_two_sided_p"]),
        "IMDbMaxPurityRoutedHolmP": _p_text(m2["holm_adjusted_p"]),
    }
    lines = ["% Auto-generated only after all 70 official records verify."]
    lines.extend(rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items())
    return "\n".join(lines) + "\n"


def _render_results_block(stats: dict) -> str:
    gate_status = "passes" if stats["primary_confirmation_gate"]["passes"] else "fails"
    return "\n".join([
        "% Auto-generated narrative; do not edit reported values by hand.",
        r"\input{generated/imdb_official_macros.tex}",
        r"\subsection{Prospective IMDb primary analysis}",
        r"The complete frozen ten-seed matrix gives mean accuracies of "
        r"\IMDbArmAAccuracyPct\%, \IMDbArmBAccuracyPct\%, "
        r"\IMDbArmCAccuracyPct\%, and \IMDbArmDAccuracyPct\% for arms A--D, "
        r"respectively.  The predeclared narrow cross-task gate \textbf{" +
        gate_status + r"}: the frozen severe-mixture baseline is "
        r"\IMDbSevereBaselineCleanPct\% clean, the observed severe-uniform "
        r"share is \IMDbSevereUniformCleanPct\%, and the routed share is "
        r"\IMDbSevereRoutedCleanPct\%.  The primary D--C effect is "
        r"\IMDbPrimaryDeltaPP\ pp (95\% CI "
        r"[\IMDbPrimaryCILowPP, \IMDbPrimaryCIHighPP]; paired two-sided "
        r"$t$-test $p=\IMDbPrimaryPairedTP$; exact sign-flip sensitivity "
        r"$p=\IMDbPrimarySignFlipP$).",
        r"\begin{table*}[t]",
        r"\centering",
        r"\input{generated/imdb_official_tables.tex}",
        r"\caption{Complete official IMDb results over ten common seeds. "
        r"Values in the arm panel are mean $\pm$ sample SD.  Confidence "
        r"intervals are paired two-sided 95\% Student-$t$ intervals.}",
        r"\label{tab:imdb-official}",
        r"\end{table*}",
        r"\subsection{Matched-$K$ secondary mechanistic analysis}",
        r"This separately frozen secondary family holds $K=128$, claimed-label "
        r"quotas, local steps, and the severe mixture fixed.  Audited clean "
        r"shares are \IMDbMatchedRandomCleanPct\% for random-$K$, "
        r"\IMDbMatchedRoutedCleanPct\% for routed-$K$, and "
        r"\IMDbMatchedMaxPurityCleanPct\% for maximum-purity-$K$.  The "
        r"routed-$K$ minus random-$K$ effect is \IMDbRoutedRandomDeltaPP\ pp "
        r"(95\% CI [\IMDbRoutedRandomCILowPP, "
        r"\IMDbRoutedRandomCIHighPP]; raw $p=\IMDbRoutedRandomRawP$, "
        r"Holm $p=\IMDbRoutedRandomHolmP$).  The maximum-purity-$K$ minus "
        r"routed-$K$ effect is \IMDbMaxPurityRoutedDeltaPP\ pp (95\% CI "
        r"[\IMDbMaxPurityRoutedCILowPP, \IMDbMaxPurityRoutedCIHighPP]; raw "
        r"$p=\IMDbMaxPurityRoutedRawP$, Holm "
        r"$p=\IMDbMaxPurityRoutedHolmP$).  These are secondary mechanistic "
        r"tests and do not confirm or replace the primary analysis.",
        r"\begin{figure*}[t]",
        r"\centering",
        r"\includegraphics[width=\textwidth]{generated/imdb_official_figure.pdf}",
        r"\caption{Paired IMDb effects and audited selected-set composition. "
        r"Points are mean paired accuracy differences and bars are unadjusted "
        r"95\% intervals; the two matched-$K$ tests form the frozen secondary "
        r"Holm family.}",
        r"\label{fig:imdb-official}",
        r"\end{figure*}",
    ]) + "\n"


def _arm_row(label: str, values: dict) -> str:
    cells = []
    for metric in METRICS:
        scale = 100 if metric in ("accuracy", "macro_f1") else 1
        digits = 2 if scale == 100 else 4
        cells.append(
            f"{scale * values[metric]['mean']:.{digits}f} $\\pm$ "
            f"{scale * values[metric]['sd']:.{digits}f}")
    return f"{label} & " + " & ".join(cells) + r" \\"


def _contrast_row(label: str, item: dict, adjusted: bool) -> str:
    p = item["paired_t_two_sided_p"]
    auxiliary_p = (item["holm_adjusted_p"] if adjusted else
                   item["exact_two_sided_sign_flip_p"])
    return (f"{label} & {100 * item['mean']:+.2f} "
            f"[{100 * item['ci95_low']:+.2f}, {100 * item['ci95_high']:+.2f}] "
            f"& {_p_text(p)} & {_p_text(auxiliary_p)} " + r"\\")


def _render_tables(stats: dict) -> str:
    lines = [
        "% Auto-generated only after all 70 official records verify.",
        r"\noindent\resizebox{0.99\linewidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Arm & Accuracy (\%) & Macro-F1 (\%) & Score NLL & Score ECE \\",
        r"\midrule",
    ]
    for method in ORIGINAL_METHODS:
        lines.append(_arm_row(ORIGINAL_DISPLAY[method],
                              stats["original_arm_summaries"][method]))
    lines.extend([r"\midrule"])
    for method in MATCHED_METHODS:
        lines.append(_arm_row(MATCHED_DISPLAY[method],
                              stats["matched_arm_summaries"][method]))
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        "",
        r"\noindent\resizebox{0.99\linewidth}{!}{%",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Frozen contrast & Mean pp [95\% CI] & Paired-$t$ $p$ & Holm / exact $p$ \\",
        r"\midrule",
        _contrast_row(r"Routed $-$ severe uniform (primary)",
                      stats["primary_contrast"], False),
        _contrast_row(r"Routed-$K$ $-$ random-$K$ (secondary)",
                      stats["matched_secondary_holm_family"][0], True),
        _contrast_row(r"Maximum-purity-$K$ $-$ routed-$K$ (secondary)",
                      stats["matched_secondary_holm_family"][1], True),
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
    ])
    return "\n".join(lines) + "\n"


def _x_map(value: float, low: float, high: float, start: float, width: float) -> float:
    return start + width * (value - low) / (high - low)


def _render_figure_tex(stats: dict) -> str:
    contrasts = [stats["primary_contrast"],
                 *stats["matched_secondary_holm_family"]]
    labels = [r"Primary: routed $-$ severe", r"Routed-$K$ $-$ random-$K$",
              r"Max-purity-$K$ $-$ routed-$K$"]
    all_limits = [100 * item[key] for item in contrasts
                  for key in ("ci95_low", "ci95_high")]
    extent = max(0.25, max(abs(value) for value in all_limits)) * 1.18
    tick_step = 0.25 if extent <= 1.25 else (0.5 if extent <= 2.5 else 1.0)
    extent = math.ceil(extent / tick_step) * tick_step
    x0, width = 1.4, 6.1
    lines = [
        r"\pdfinfoomitdate=1", r"\pdftrailerid{}",
        r"\documentclass[tikz,border=2pt]{standalone}",
        r"\usepackage{newtxtext}", r"\usepackage{newtxmath}",
        r"\definecolor{kafablue}{RGB}{38,92,140}",
        r"\definecolor{kafaorange}{RGB}{221,132,43}",
        r"\definecolor{kafagray}{RGB}{166,166,166}",
        r"\begin{document}", r"\begin{tikzpicture}[font=\sffamily\scriptsize]",
        r"\node[anchor=west,font=\sffamily\bfseries] at (0,4.7) {(a) Paired accuracy effects};",
        rf"\draw[->] ({x0:.3f},0.75) -- ({x0+width+0.15:.3f},0.75);",
        rf"\node at ({x0+width/2:.3f},0.18) {{accuracy difference (pp)}};",
    ]
    zero_x = _x_map(0, -extent, extent, x0, width)
    lines.append(rf"\draw[densely dashed,kafagray] ({zero_x:.3f},1.05) -- ({zero_x:.3f},4.35);")
    tick = -extent
    while tick <= extent + 1e-9:
        xpos = _x_map(tick, -extent, extent, x0, width)
        lines.append(rf"\draw ({xpos:.3f},0.70)--({xpos:.3f},0.80) node[below=2pt] {{{tick:g}}};")
        tick += tick_step
    for index, (label, item) in enumerate(zip(labels, contrasts)):
        y = 3.85 - index * 1.15
        lo = _x_map(100 * item["ci95_low"], -extent, extent, x0, width)
        hi = _x_map(100 * item["ci95_high"], -extent, extent, x0, width)
        center = _x_map(100 * item["mean"], -extent, extent, x0, width)
        color = "kafablue" if index == 0 else "kafaorange"
        lines.extend([
            rf"\node[anchor=east] at ({x0-0.12:.3f},{y:.3f}) {{{label}}};",
            rf"\draw[line width=0.8pt,{color}] ({lo:.3f},{y:.3f})--({hi:.3f},{y:.3f});",
            rf"\draw[{color}] ({lo:.3f},{y-0.10:.3f})--({lo:.3f},{y+0.10:.3f}) ({hi:.3f},{y-0.10:.3f})--({hi:.3f},{y+0.10:.3f});",
            rf"\fill[{color}] ({center:.3f},{y:.3f}) circle (2.1pt);",
        ])

    bx, bw = 9.3, 5.1
    lines.extend([
        r"\node[anchor=west,font=\sffamily\bfseries] at (8.25,4.7) {(b) Selected-set composition};",
        rf"\draw[->] ({bx:.3f},0.75)--({bx+bw+0.15:.3f},0.75);",
        rf"\node at ({bx+bw/2:.3f},0.18) {{selected share (\%)}};",
        r"\fill[kafablue] (9.25,4.32) rectangle +(0.28,0.20); \node[anchor=west] at (9.58,4.42) {clean};",
        r"\fill[kafaorange] (11.0,4.32) rectangle +(0.28,0.20); \node[anchor=west] at (11.33,4.42) {conflict};",
        r"\fill[kafagray] (13.0,4.32) rectangle +(0.28,0.20); \node[anchor=west] at (13.33,4.42) {off-task};",
    ])
    for tick in (0, 25, 50, 75, 100):
        xpos = bx + bw * tick / 100
        lines.append(rf"\draw ({xpos:.3f},0.70)--({xpos:.3f},0.80) node[below=2pt] {{{tick}}};")
    composition_rows = [
        ("Clean uniform", stats["original_composition"]["clean_unfiltered_distillation"],
         stats["original_arm_summaries"]["clean_unfiltered_distillation"]["accuracy"]["mean"]),
        ("Severe uniform", stats["original_composition"]["severe_unfiltered_distillation"],
         stats["original_arm_summaries"]["severe_unfiltered_distillation"]["accuracy"]["mean"]),
        (r"Random-$K$", stats["matched_composition"]["severe_random_k"],
         stats["matched_arm_summaries"]["severe_random_k"]["accuracy"]["mean"]),
        (r"Routed-$K$", stats["matched_composition"]["severe_routed_k"],
         stats["matched_arm_summaries"]["severe_routed_k"]["accuracy"]["mean"]),
        (r"Max-purity-$K$", stats["matched_composition"]["oracle_max_purity_k"],
         stats["matched_arm_summaries"]["oracle_max_purity_k"]["accuracy"]["mean"]),
    ]
    for index, (label, composition, accuracy) in enumerate(composition_rows):
        y = 4.05 - index * 0.67
        fractions = composition["fractions"]
        clean = 100 * (fractions["clean"] or 0)
        conflict = 100 * (fractions["conflict"] or 0)
        off_task = 100 * (fractions["off_task"] or 0)
        a = bx + bw * clean / 100
        b = a + bw * conflict / 100
        c = b + bw * off_task / 100
        lines.extend([
            rf"\node[anchor=east] at ({bx-0.10:.3f},{y:.3f}) {{{label}}};",
            rf"\fill[kafablue] ({bx:.3f},{y-0.16:.3f}) rectangle ({a:.3f},{y+0.16:.3f});",
            rf"\fill[kafaorange] ({a:.3f},{y-0.16:.3f}) rectangle ({b:.3f},{y+0.16:.3f});",
            rf"\fill[kafagray] ({b:.3f},{y-0.16:.3f}) rectangle ({c:.3f},{y+0.16:.3f});",
            rf"\draw ({bx:.3f},{y-0.16:.3f}) rectangle ({bx+bw:.3f},{y+0.16:.3f});",
            rf"\node[anchor=west] at ({bx+bw+0.10:.3f},{y:.3f}) {{{100*accuracy:.2f}\% acc.}};",
        ])
    lines.extend([
        r"\end{tikzpicture}", r"\end{document}",
    ])
    return "\n".join(lines) + "\n"


def _compile_figure(stage: Path) -> None:
    command = ["pdflatex", "-halt-on-error", "-interaction=batchmode",
               "imdb_official_figure.tex"]
    completed = subprocess.run(command, cwd=stage, text=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError("vector figure compilation failed:\n" + completed.stdout)
    pdf = stage / "imdb_official_figure.pdf"
    _require(pdf.is_file() and pdf.stat().st_size > 1000,
             "pdflatex did not create a nonempty vector PDF")


def generate_assets(
    original_dir: Path,
    matched_dir: Path,
    output_dir: Path,
    original_protocol: Path = ROOT / "kafa_script/imdb_server_role_reversal_protocol_v1.json",
    matched_protocol: Path = ROOT / "kafa_script/imdb_matched_control_addendum_protocol_v1.json",
    original_verify: Callable = original_verifier.verify_matrix,
    matched_verify: Callable = matched_verifier.verify_matrix,
    equivalence_verify: Callable = equivalence_verifier.verify_phase,
) -> dict:
    """Validate all gates, then atomically publish the six paper assets."""
    original_dir, matched_dir, output_dir = map(
        Path, (original_dir, matched_dir, output_dir))
    _check_exact_matrix(original_dir, ORIGINAL_METHODS, 40, "original matrix")
    _check_exact_matrix(matched_dir, MATCHED_METHODS, 30, "matched matrix")

    original_gate = original_verify(
        "official_evaluation", original_dir, Path(original_protocol))
    matched_gate = matched_verify(
        "official_evaluation", matched_dir, Path(matched_protocol))
    equivalence_gate = equivalence_verify(
        "official_evaluation", original_dir, matched_dir)
    _require(original_gate.get("phase") == "official_evaluation"
             and original_gate.get("records") == 40,
             "original official verifier did not certify 40 records")
    _require(matched_gate.get("phase") == "official_evaluation"
             and matched_gate.get("records") == 30,
             "matched official verifier did not certify 30 records")
    _require(equivalence_gate.get("phase") == "official_evaluation"
             and equivalence_gate.get("pairs") == 10
             and equivalence_gate.get("functional_equivalence") is True,
             "routed equivalence verifier did not certify all ten pairs")

    original_records = _load_records(original_dir, ORIGINAL_METHODS)
    matched_records = _load_records(matched_dir, MATCHED_METHODS)
    stats = compute_statistics(original_records, matched_records)
    stats["verification"] = {
        "original": original_gate,
        "matched": matched_gate,
        "routed_equivalence": equivalence_gate,
    }
    stats["raw_record_sha256"] = {
        "original": {path.name: _sha256(path) for path in sorted(original_dir.glob("seed*.json"))},
        "matched": {path.name: _sha256(path) for path in sorted(matched_dir.glob("seed*.json"))},
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".imdb-assets-", dir=output_dir.parent))
    try:
        (stage / "imdb_official_macros.tex").write_text(
            _render_macros(stats), encoding="utf-8")
        (stage / "imdb_official_tables.tex").write_text(
            _render_tables(stats), encoding="utf-8")
        (stage / "imdb_official_figure.tex").write_text(
            _render_figure_tex(stats), encoding="utf-8")
        (stage / "imdb_official_results_block.tex").write_text(
            _render_results_block(stats), encoding="utf-8")
        (stage / "imdb_official_assets.json").write_text(
            json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _compile_figure(stage)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ASSET_NAMES:
            os.replace(stage / name, output_dir / name)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return {
        "status": "generated_from_verified_complete_official_matrices",
        "output_dir": str(output_dir),
        "assets": list(ASSET_NAMES),
        "original_records": 40,
        "matched_records": 30,
        "routed_equivalence_pairs": 10,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dir", type=Path, default=ROOT /
                        "kafa_script/analysis_outputs/imdb_server_role_reversal/official_evaluation")
    parser.add_argument("--matched-dir", type=Path, default=ROOT /
                        "kafa_script/analysis_outputs/imdb_matched_control_addendum/official_evaluation")
    parser.add_argument("--output-dir", type=Path, default=ROOT /
                        "paper/generated")
    parser.add_argument("--original-protocol", type=Path, default=ROOT /
                        "kafa_script/imdb_server_role_reversal_protocol_v1.json")
    parser.add_argument("--matched-protocol", type=Path, default=ROOT /
                        "kafa_script/imdb_matched_control_addendum_protocol_v1.json")
    args = parser.parse_args()
    result = generate_assets(
        args.original_dir, args.matched_dir, args.output_dir,
        args.original_protocol, args.matched_protocol)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
