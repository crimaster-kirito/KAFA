#!/usr/bin/env python3
"""Recompute headline table and narrative values from raw JSON and match TeX."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kafa_script/analysis_outputs"
PAPER = ROOT / "paper"
REPORT_SOURCES = [PAPER / "main.tex", PAPER / "supplement.tex",
                  PAPER / "evidence_ledger.tex"]
for generated_name in ("imdb_official_macros.tex",
                       "imdb_official_tables.tex",
                       "imdb_official_figure.tex"):
    generated = PAPER / "generated" / generated_name
    if generated.exists():
        REPORT_SOURCES.append(generated)
TEX = "\n".join(path.read_text() for path in REPORT_SOURCES)


def read(name: str) -> dict:
    return json.loads((OUT / name).read_text())


def values(template: str, seeds, metric: str,
           result_key: str = "final") -> list[float]:
    return [read(template.format(seed=seed))[result_key][metric] for seed in seeds]


def mean(template: str, seeds, metric: str,
         result_key: str = "final") -> float:
    return statistics.mean(values(template, seeds, metric, result_key))


def require_text(text: str) -> None:
    if text not in TEX and " ".join(text.split()) not in " ".join(TEX.split()):
        raise AssertionError(f"reported value is absent or stale: {text}")


def interval(values: list[float]) -> tuple[float, float]:
    center = statistics.mean(values)
    half = stats.t.ppf(.975, len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))
    return center, half


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    adjusted, running = {}, 0.0
    ordered = sorted(pvalues, key=pvalues.get)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def main() -> None:
    # AG News frozen official test.
    seeds = (142, 143, 144)
    anchor_template = "final_anchor_only_strict_seed{seed}.json"
    routed_template = "final_routed_seed{seed}.json"
    anchor_acc = values(anchor_template, seeds, "accuracy")
    routed_acc = values(routed_template, seeds, "accuracy")
    require_text(
        f"AG News & Anchor-only & \\({100*statistics.mean(anchor_acc):.2f}\\pm"
        f"{100*statistics.stdev(anchor_acc):.2f}\\) & "
        f"{100*mean(anchor_template, seeds, 'macro_f1'):.2f} & "
        f"{mean(anchor_template, seeds, 'ece'):.3f}\\\\")
    require_text(
        f"AG News & Routed & \\({100*statistics.mean(routed_acc):.2f}\\pm"
        f"{100*statistics.stdev(routed_acc):.2f}\\) & "
        f"{100*mean(routed_template, seeds, 'macro_f1'):.2f} & "
        f"{mean(routed_template, seeds, 'ece'):.3f}\\\\")
    ag_delta = 100 * statistics.mean([r - a for a, r in zip(anchor_acc, routed_acc)])
    require_text(f"\\({ag_delta:.2f}\\pp\\)")
    delta_values = [r - a for a, r in zip(anchor_acc, routed_acc)]
    delta_mean, delta_half = interval(delta_values)
    require_text(f"\\({100*(delta_mean-delta_half):.2f}\\) to "
                 f"\\({100*(delta_mean+delta_half):.2f}\\pp\\)")
    anchor_wall = statistics.mean(read(anchor_template.format(seed=seed))["total_wall_seconds"] for seed in seeds)
    routed_wall = statistics.mean(read(routed_template.format(seed=seed))["total_wall_seconds"] for seed in seeds)
    require_text(f"\\({routed_wall/anchor_wall:.2f}\\times\\)")

    # AG News local development diagnostics cited in the narrative.
    local_prefix = "agnews_dev/mixtures/"
    local_families = {
        "qwen05b": ("qwen05b_clean_adaptive_relative_v2_seed42.json",
                    "qwen05b_c40_o20_seed42.json"),
        "smollm360m": ("smollm360m_clean_seed42.json",
                       "smollm360m_c40_o20_seed42.json"),
        "tinyllama11b": ("tinyllama11b_clean_seed42.json",
                         "tinyllama11b_c40_o20_seed42.json"),
    }
    local_drops = []
    for clean_name, mixed_name in local_families.values():
        clean = read(local_prefix + clean_name)["final"]["accuracy"]
        mixed = read(local_prefix + mixed_name)["final"]["accuracy"]
        local_drops.append(100 * (clean - mixed))
    require_text(f"{min(local_drops):.2f}--{max(local_drops):.2f} points")
    qwen_router_gains = []
    for seed in (42, 43, 44):
        naive_name = ("qwen05b_c40_o20_seed42.json" if seed == 42 else
                      f"qwen05b_mixed_naive_seed{seed}.json")
        routed_name = ("qwen05b_mixed_adaptive_relative_v2_seed42.json" if seed == 42 else
                       f"qwen05b_mixed_adaptive_seed{seed}.json")
        qwen_router_gains.append(100 * (
            read(local_prefix + routed_name)["final"]["accuracy"] -
            read(local_prefix + naive_name)["final"]["accuracy"]))
    require_text(f"\\({min(qwen_router_gains):+.2f}\\) to "
                 f"\\({max(qwen_router_gains):+.2f}\\pp\\)")

    # AG News Co-Teaching development table.
    for family, display in (("qwen", "Qwen"), ("smol", "SmolLM2"),
                            ("tiny", "TinyLlama")):
        cells = []
        for condition in ("conflict40", "joint"):
            naive = mean(f"agnews_coteach_{family}_naive_{condition}_seed{{seed}}.json",
                         (75, 76, 77), "accuracy")
            coteach = mean(f"agnews_coteach_{family}_coteach_{condition}_seed{{seed}}.json",
                           (75, 76, 77), "accuracy", "ensemble_final")
            cells.append(f"{100*naive:.2f} / {100*coteach:.2f} ({100*(coteach-naive):+.2f})")
        require_text(f"{display} & {cells[0]} & {cells[1]}\\\\")

    # Natural weak-label headline gains.
    natural = {
        "hausa": (81, 82, 83),
        "yoruba": (84, 85, 86),
    }
    natural_gains = {}
    natural_means = {}
    natural_costs, natural_purity = {}, {}
    for corpus, corpus_seeds in natural.items():
        natural_gains[corpus] = []
        natural_means[corpus], natural_costs[corpus], natural_purity[corpus] = [], [], []
        for family in ("qwen", "smol", "tiny"):
            naive_template = f"{corpus}_natural_{family}_naive_dev_seed{{seed}}.json"
            coteach_template = f"{corpus}_natural_{family}_coteach_dev_seed{{seed}}.json"
            naive = mean(naive_template, corpus_seeds, "accuracy")
            coteach = mean(coteach_template, corpus_seeds, "accuracy")
            natural_gains[corpus].append(100 * (coteach - naive))
            natural_means[corpus].append((100 * naive, 100 * coteach))
            natural_costs[corpus].append(statistics.mean(
                read(coteach_template.format(seed=seed))["train_wall_seconds"] /
                read(naive_template.format(seed=seed))["train_wall_seconds"]
                for seed in corpus_seeds))
            clean_fractions = []
            for seed in corpus_seeds:
                record = read(coteach_template.format(seed=seed))
                for counts in record["selected_kind_counts"]:
                    clean_fractions.append(counts.get("clean_label_match", 0) / sum(counts.values()))
            natural_purity[corpus].append(100 * statistics.mean(clean_fractions))
    for gain in natural_gains["hausa"] + natural_gains["yoruba"]:
        require_text(f"{gain:+.2f}")
    hausa_pairs = natural_means["hausa"]
    formatted_pairs = [f"{naive:.2f}/{coteach:.2f} for {display}"
                       for (naive, coteach), display in
                       zip(hausa_pairs, ("Qwen", "SmolLM2", "TinyLlama"))]
    require_text("naive/Co-Teaching accuracies are " +
                 ", ".join(formatted_pairs[:2]) + ", and " + formatted_pairs[2])
    require_text(f"{100*(1-read('hausa_natural_qwen_naive_dev_seed81.json')['measured_training_noise_rate']):.2f}\\% of training examples")
    require_text(f"{min(natural_purity['hausa']):.1f}\\%--{max(natural_purity['hausa']):.1f}\\%")
    require_text(f"{min(natural_purity['yoruba']):.1f}\\%--{max(natural_purity['yoruba']):.1f}\\%")
    require_text(f"\\({min(natural_costs['hausa']):.2f}\\)--\\({max(natural_costs['hausa']):.2f}\\times\\)")

    # GSM8K frozen Qwen and non-Qwen official tests.
    qwen_naive = mean("gsm8k_final_naive_severe_seed{seed}.json",
                      (158, 159, 160), "exact_match")
    qwen_teacher = mean("gsm8k_final_teacher_severe_seed{seed}.json",
                        (158, 159, 160), "exact_match")
    require_text(f"{100*qwen_naive:.2f}\\% EM")
    require_text(f"{100*qwen_teacher:.2f}\\%")
    qwen_clean_naive = mean("gsm8k_final_naive_clean_seed{seed}.json",
                            (158, 159, 160), "exact_match")
    qwen_clean_teacher = mean("gsm8k_final_teacher_clean_seed{seed}.json",
                              (158, 159, 160), "exact_match")
    require_text(f"{100*qwen_clean_naive:.2f}\\% to {100*qwen_clean_teacher:.2f}\\%")
    cross_p = {}
    for family, display in (("smol", "SmolLM2"), ("tiny", "TinyLlama")):
        for condition in ("clean", "severe"):
            naive = mean(
                f"gsm8k_crossfamily_final_{family}_naive_{condition}_seed{{seed}}.json",
                (164, 165, 166), "exact_match")
            teacher = mean(
                f"gsm8k_crossfamily_final_{family}_teacher_{condition}_seed{{seed}}.json",
                (164, 165, 166), "exact_match")
            require_text(f"{display} & {condition}  & {100*naive:.2f} & "
                         f"{100*teacher:.2f} & {100*(teacher-naive):+.2f}\\\\")
        naive_values = values(f"gsm8k_crossfamily_final_{family}_naive_severe_seed{{seed}}.json",
                              (164, 165, 166), "exact_match")
        teacher_values = values(f"gsm8k_crossfamily_final_{family}_teacher_severe_seed{{seed}}.json",
                                (164, 165, 166), "exact_match")
        cross_p[family] = float(stats.ttest_rel(teacher_values, naive_values,
                                               alternative="greater").pvalue)
    adjusted = holm(cross_p)
    require_text(f"\\(p={next(iter(adjusted.values())):.4f}\\)")

    # Post-hoc matched selection controls on development data.
    control_gains = {}
    for family in ("smol", "tiny"):
        teacher = mean(f"gsm8k_crossfamily_{family}_teacher_severe_seed{{seed}}.json",
                       (61, 62, 63), "exact_match")
        control_gains[family] = []
        for selector in ("numeric_random", "client_small_loss"):
            control = mean(f"gsm8k_matched_{family}_{selector}_severe_seed{{seed}}.json",
                           (61, 62, 63), "exact_match")
            control_gains[family].append(100 * (teacher - control))
    require_text(f"\\({control_gains['smol'][0]:+.2f}/{control_gains['smol'][1]:+.2f}\\pp\\)")
    require_text(f"\\({control_gains['tiny'][0]:+.2f}/{control_gains['tiny'][1]:+.2f}\\pp\\)")

    # Post-observation AG News seed replication, kept separate from original.
    replication_deltas = []
    for seed in (145, 146, 147, 148, 149):
        anchor = read(f"replication_anchor_only_seed{seed}.json")["final"]["accuracy"]
        routed = read(f"replication_routed_seed{seed}.json")["final"]["accuracy"]
        replication_deltas.append(routed - anchor)
    replication_mean, replication_half = interval(replication_deltas)
    require_text(f"\\({100*replication_mean:.2f}\\pp\\)")
    require_text(f"\\({100*(replication_mean-replication_half):.2f}\\) to "
                 f"\\({100*(replication_mean+replication_half):.2f}\\pp\\)")
    original_deltas = [r - a for a, r in zip(anchor_acc, routed_acc)]
    combined_mean = statistics.mean(original_deltas + replication_deltas)
    require_text(f"\\({100*combined_mean:.2f}\\pp\\)")

    # Independently frozen TREC server-transfer matrix.
    trec_seeds = (251, 252, 253, 254, 255)
    trec_anchor = "trec_server/official/seed{seed}_public_anchor_only.json"
    trec_routed = "trec_server/official/seed{seed}_routed_distillation.json"
    for template, display in ((trec_anchor, "Anchor-only"),
                              (trec_routed, "Routed")):
        accuracies = values(template, trec_seeds, "accuracy")
        require_text(
            f"TREC & {display} & \\({100*statistics.mean(accuracies):.2f}\\pm"
            f"{100*statistics.stdev(accuracies):.2f}\\) & "
            f"{100*mean(template, trec_seeds, 'macro_f1'):.2f} & "
            f"{mean(template, trec_seeds, 'ece'):.3f}\\\\")
    trec_metrics = {
        "accuracy": "accuracy",
        "macro_f1": "macro_f1",
        "class_nll": "canonical_label_nll",
        "ece": "ece",
    }
    trec_deltas = {}
    for name, metric in trec_metrics.items():
        anchor_values = values(trec_anchor, trec_seeds, metric)
        routed_values = values(trec_routed, trec_seeds, metric)
        trec_deltas[name] = [routed - anchor for anchor, routed in
                             zip(anchor_values, routed_values)]
    acc_center, acc_half = interval(trec_deltas["accuracy"])
    require_text(f"\\({100*acc_center:+.2f}\\pp\\)")
    require_text(f"\\([{100*(acc_center-acc_half):+.2f},"
                 f"{100*(acc_center+acc_half):+.2f}]\\pp\\)")
    acc_p = stats.ttest_1samp(trec_deltas["accuracy"], 0).pvalue
    require_text(f"\\(p={acc_p:.3f}\\)")
    nll_center = statistics.mean(trec_deltas["class_nll"])
    nll_p = stats.ttest_1samp(trec_deltas["class_nll"], 0).pvalue
    require_text(f"decreases by {abs(nll_center):.4f} (\\(p={nll_p:.4f}\\))")
    ece_center, ece_half = interval(trec_deltas["ece"])
    ece_p = stats.ttest_1samp(trec_deltas["ece"], 0).pvalue
    require_text(f"by {abs(ece_center):.4f} in every seed")
    require_text(f"\\([{ece_center-ece_half:.4f},{ece_center+ece_half:.4f}]\\)")
    require_text(f"\\(p={ece_p:.5f}\\)")
    routed_records = [read(trec_routed.format(seed=seed)) for seed in trec_seeds]
    gate_triggers = sum(record["routing"]["gate_triggered"]
                        for run in routed_records
                        for round_record in run["round_records"]
                        for record in round_record["client_records"])
    require_text(f"{gate_triggers}/30 client-rounds")
    require_text(
        f"{routed_records[0]['total_client_to_server_payload_bytes']:,} bytes of serialized probability tensors per run"
    )
    trec_anchor_wall = statistics.mean(
        read(trec_anchor.format(seed=seed))["wall_seconds"] for seed in trec_seeds)
    trec_routed_wall = statistics.mean(
        read(trec_routed.format(seed=seed))["wall_seconds"] for seed in trec_seeds)
    require_text(f"\\({trec_routed_wall/trec_anchor_wall:.2f}\\times\\)")

    # Post-primary TREC four-arm reused-test mechanism ablation.
    post_seeds = tuple(range(261, 271))
    post_templates = {
        "anchor": "trec_server_ablation/official_reused_test/seed{seed}_public_anchor_only.json",
        "clean": "trec_server_ablation/official_reused_test/seed{seed}_clean_unfiltered_distillation.json",
        "severe": "trec_server_ablation/official_reused_test/seed{seed}_severe_unfiltered_distillation.json",
        "routed": "trec_server_ablation/official_reused_test/seed{seed}_severe_routed_distillation.json",
    }
    post_displays = {
        "anchor": "Anchor-only",
        "clean": "Clean uniform",
        "severe": "Severe uniform",
        "routed": "Severe routed",
    }
    for key, template in post_templates.items():
        accuracies = values(template, post_seeds, "accuracy")
        require_text(
            f"TREC post & {post_displays[key]} & "
            f"\\({100*statistics.mean(accuracies):.2f}\\pm"
            f"{100*statistics.stdev(accuracies):.2f}\\) & "
            f"{100*mean(template, post_seeds, 'macro_f1'):.2f} & "
            f"{mean(template, post_seeds, 'ece'):.3f}\\\\")
    contrast_specs = {
        "clean": ("clean", "anchor"),
        "severe": ("severe", "clean"),
        "routed": ("routed", "severe"),
    }
    post_contrasts = {}
    raw_p = {}
    for name, (left, right) in contrast_specs.items():
        left_values = values(post_templates[left], post_seeds, "accuracy")
        right_values = values(post_templates[right], post_seeds, "accuracy")
        differences = [a - b for a, b in zip(left_values, right_values)]
        post_contrasts[name] = differences
        raw_p[name] = float(stats.ttest_1samp(differences, 0).pvalue)
    adjusted = holm(raw_p)
    for name in ("clean", "severe", "routed"):
        center, half = interval(post_contrasts[name])
        reported_center = (f"{100*center:+.2f}" if name == "clean"
                           else f"{abs(100*center):.2f}")
        require_text(f"\\({reported_center}\\pp\\)")
        require_text(
            f"\\([{100*(center-half):+.2f},{100*(center+half):+.2f}]\\)")
        p_digits = 4 if name == "routed" else 3
        p_text = f"{adjusted[name]:.{p_digits}f}".lstrip("0")
        require_text(f"Holm \\(p={p_text}\\)")
    routed_rows = [read(post_templates["routed"].format(seed=seed))
                   for seed in post_seeds]
    routed_triggers = sum(
        client["selection"].get("gate_triggered", False)
        for row in routed_rows for round_record in row["round_records"]
        for client in round_record["client_records"])
    require_text(f"{routed_triggers}/60 client-rounds")
    selected_counts = {"clean": 0, "conflict": 0, "off_task": 0}
    for row in routed_rows:
        for round_record in row["round_records"]:
            for client in round_record["client_records"]:
                for kind, count in client["selected_kind_counts_audit_only"].items():
                    selected_counts[kind] += count
    selected_fraction = selected_counts["clean"] / sum(selected_counts.values())
    require_text(f"{100*selected_fraction:.1f}\\%")
    net = [
        read(post_templates["routed"].format(seed=seed))["final"]["accuracy"] -
        read(post_templates["anchor"].format(seed=seed))["final"]["accuracy"]
        for seed in post_seeds]
    net_center, net_half = interval(net)
    require_text(f"\\({100*net_center:+.2f}\\pp\\)")
    require_text(
        f"\\([{100*(net_center-net_half):+.2f},{100*(net_center+net_half):+.2f}]\\)")
    for metric, expected_digits in (("canonical_label_nll", 4), ("ece", 4)):
        differences = [
            read(post_templates["routed"].format(seed=seed))["final"][metric] -
            read(post_templates["severe"].format(seed=seed))["final"][metric]
            for seed in post_seeds]
        require_text(f"by {abs(statistics.mean(differences)):.{expected_digits}f}")

    # Secondary-holdout severity table.
    for family, display in (("qwen", "Qwen"), ("smol", "SmolLM2"),
                            ("tiny", "TinyLlama")):
        cells = []
        for condition in ("clean", "medium", "severe"):
            naive = mean(f"gsm8k_severity_{family}_naive_{condition}_seed{{seed}}.json",
                         (72, 73, 74), "exact_match")
            teacher = mean(f"gsm8k_severity_{family}_teacher_{condition}_seed{{seed}}.json",
                           (72, 73, 74), "exact_match")
            cells.append(f"{100*naive:.2f} & {100*teacher:.2f} ({100*(teacher-naive):+.2f})")
        require_text(f"{display} & {' & '.join(cells)}\\\\")

    print("Verified headline table/narrative values against raw JSON: AG News, "
          "primary/post-primary TREC, GSM8K, Co-Teaching, severity, Hausa, "
          "and Yoruba")


if __name__ == "__main__":
    main()
