"""Generate the final cross-task server audit table, figure, and ledger.

The generator is intentionally release-gated.  It first checks the exact file
sets and then calls every frozen verifier.  It does not deserialize a result
record, create the output directory, or stage an asset until all five verifier
gates (TREC, IMDb original, IMDb matched, IMDb routed equivalence, and AG News)
have positively certified their complete matrices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
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

from kafa_script import verify_agnews_server_ablation as ag_verifier  # noqa: E402
from kafa_script import verify_imdb_matched_control_addendum as matched_verifier  # noqa: E402
from kafa_script import verify_imdb_routed_equivalence as equivalence_verifier  # noqa: E402
from kafa_script import verify_imdb_server_role_reversal as imdb_verifier  # noqa: E402
from kafa_script import verify_trec_server_ablation as trec_verifier  # noqa: E402
from kafa_script.summarize_trec_server_ablation import (  # noqa: E402
    exact_sign_flip_p,
    holm_adjust,
    mean_ci,
    paired_t_p,
)


FOUR_ARMS = (
    "public_anchor_only",
    "clean_unfiltered_distillation",
    "severe_unfiltered_distillation",
    "severe_routed_distillation",
)
ARM_LETTERS = dict(zip(FOUR_ARMS, "ABCD"))
ADJACENT = (
    ("B-A", "clean_unfiltered_distillation", "public_anchor_only"),
    ("C-B", "severe_unfiltered_distillation", "clean_unfiltered_distillation"),
    ("D-C", "severe_routed_distillation", "severe_unfiltered_distillation"),
)
MATCHED_ARMS = ("severe_random_k", "severe_routed_k", "oracle_max_purity_k")
MATCHED_LABELS = {
    "severe_random_k": r"Random-$K$",
    "severe_routed_k": r"Routed-$K$",
    "oracle_max_purity_k": r"Maximum-purity-$K$",
}
TASKS = {
    "trec": {
        "name": "TREC",
        "scope": "post-primary reused-test discovery",
        "phase": "official_evaluation",
        "seeds": tuple(range(261, 271)),
        "methods": FOUR_ARMS,
        "records": 40,
    },
    "imdb": {
        "name": "IMDb",
        "scope": "prospective role-reversal",
        "phase": "official_evaluation",
        "seeds": tuple(range(321, 331)),
        "methods": FOUR_ARMS,
        "records": 40,
    },
    "agnews": {
        "name": "AG News",
        "scope": "post-primary resampling",
        "phase": "secondary_holdout",
        "seeds": tuple(range(281, 291)),
        "methods": FOUR_ARMS,
        "records": 40,
    },
}
ASSET_NAMES = (
    "server_audit_table.tex",
    "server_audit_figure.tex",
    "server_audit_figure.pdf",
    "server_audit_ledger.json",
    "server_audit_current_inputs.json",
)

KNOWN_COMPOSITION_KINDS = frozenset(("clean", "conflict", "off_task"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expect_equal(actual, expected, field: str) -> None:
    _require(actual == expected,
             f"frozen protocol semantic mismatch at {field}: "
             f"expected={expected!r}, observed={actual!r}")


def _expect_tokens(value: object, tokens: tuple[str, ...], field: str) -> None:
    _require(isinstance(value, str) and
             all(token.lower() in value.lower() for token in tokens),
             f"frozen protocol semantic tokens missing at {field}: {tokens}")


def _load_protocol(path: Path, expected_name: str, expected_status: str) -> dict:
    _require(path.is_file(), f"frozen protocol is missing: {path}")
    try:
        protocol = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(f"cannot parse frozen protocol {path}: {error}") from error
    _require(isinstance(protocol, dict) and len(protocol) >= 10,
             f"frozen protocol is an empty or incomplete shell: {path}")
    _expect_equal(protocol.get("name"), expected_name, f"{expected_name}.name")
    _expect_equal(protocol.get("status"), expected_status, f"{expected_name}.status")
    boundaries = protocol.get("evidence_boundaries")
    _require(isinstance(boundaries, list) and len(boundaries) >= 5 and
             all(isinstance(item, str) and item.strip() for item in boundaries),
             f"{expected_name}.evidence_boundaries is incomplete")
    return protocol


def _bind_protocol_semantics(trec_path: Path, imdb_path: Path,
                             matched_path: Path, ag_path: Path) -> dict:
    """Parse and bind every inference semantic consumed by this generator.

    The verifier digest binds the remaining frozen implementation/data fields
    byte-for-byte.  This explicit projection prevents a verifier-shaped shell
    from silently changing seeds, arms, estimands, multiplicity, evidence
    timing, matched K, or the IMDb confirmation rule.
    """
    trec = _load_protocol(
        trec_path, "trec_server_post_primary_ablation_v1",
        "refrozen_after_selected_composition_instrumentation_repair_before_any_valid_official_ablation_record")
    imdb = _load_protocol(
        imdb_path, "imdb_server_role_reversal_protocol_v1",
        "refrozen_after_pre_record_training_line_audit_repair_before_any_imdb_metric_or_test_json_decoding")
    matched = _load_protocol(
        matched_path, "imdb_matched_control_addendum_v1",
        "frozen_after_imdb_development_observation_before_any_matched_control_or_original_official_imdb_model_evaluation_with_pre_evaluation_cpu_contract_repairs")
    ag = _load_protocol(
        ag_path, "agnews_server_post_primary_ablation_v1",
        "refrozen_after_prior_training_exposure_audit_with_all_previous_records_invalidated")

    trec_section = trec.get("official_evaluation", {})
    _expect_equal(trec.get("confirmatory"), False, "trec.confirmatory")
    _expect_equal(trec.get("role"),
                  "post_primary_reused_official_test_mechanistic_ablation",
                  "trec.role")
    _expect_equal(trec.get("official_test_access_history", {}).get(
        "previously_accessed"), True, "trec.official_test.previously_accessed")
    _expect_equal(trec_section.get("seeds"), list(TASKS["trec"]["seeds"]),
                  "trec.official.seeds")
    _expect_equal(trec_section.get("comparators"), list(FOUR_ARMS),
                  "trec.official.comparators")
    _expect_equal(trec_section.get("primary_metric"), "accuracy",
                  "trec.official.primary_metric")
    _expect_equal(
        [(item.get("left"), item.get("right"))
         for item in trec_section.get("primary_contrasts", [])],
        [(left, right) for _, left, right in ADJACENT],
        "trec.official.primary_contrasts")
    _expect_tokens(trec_section.get("analysis"),
                   ("two-sided", "paired", "holm", "sign-flip"),
                   "trec.official.analysis")
    _expect_tokens(trec_section.get("net_effect"), ("descriptive", "fourth"),
                   "trec.official.net_effect")
    _expect_tokens(trec_section.get("equivalence_policy"),
                   ("no equivalence", "nonsignificant"),
                   "trec.official.equivalence_policy")

    imdb_section, imdb_analysis = (imdb.get("official_evaluation", {}),
                                    imdb.get("analysis", {}))
    _expect_equal(imdb_section.get("seeds"), list(TASKS["imdb"]["seeds"]),
                  "imdb.official.seeds")
    _expect_equal(imdb_section.get("comparators"), list(FOUR_ARMS),
                  "imdb.official.comparators")
    primary = imdb_analysis.get("single_primary_contrast", {})
    _expect_equal(
        (primary.get("id"), primary.get("left"), primary.get("right"),
         primary.get("metric"), primary.get("alpha")),
        ("routing_accuracy_role_reversal", "severe_routed_distillation",
         "severe_unfiltered_distillation", "accuracy", 0.05),
        "imdb.analysis.single_primary_contrast")
    _expect_tokens(primary.get("test"), ("two-sided", "paired", "ten"),
                   "imdb.primary.test")
    _expect_tokens(primary.get("confidence_interval"), ("95%", "paired"),
                   "imdb.primary.confidence_interval")
    _expect_tokens(primary.get("sensitivity"), ("exact", "sign-flip", "integer"),
                   "imdb.primary.sensitivity")
    _expect_tokens(imdb_analysis.get("confirmation_gate"),
                   ("103/256", "strictly negative", "p<0.05"),
                   "imdb.analysis.confirmation_gate")
    _require(any("clean-minus-anchor" in item for item in
                 imdb_analysis.get("descriptive_only", [])),
             "imdb descriptive-only B-A/C-B binding is missing")
    _require(any("No additional p-values" in item for item in
                 imdb_analysis.get("prohibited", [])),
             "imdb additional-p prohibition is missing")

    matched_section, matched_analysis = (matched.get("official_evaluation", {}),
                                          matched.get("analysis", {}))
    _expect_equal(matched_section.get("seeds"), list(TASKS["imdb"]["seeds"]),
                  "matched.official.seeds")
    _expect_equal(matched_section.get("comparators"), list(MATCHED_ARMS),
                  "matched.official.comparators")
    fixed = matched.get("fixed_configuration", {})
    _expect_equal((fixed.get("rounds"), len(fixed.get("client_models", [])),
                   fixed.get("matched_k"), fixed.get("matched_quota_per_claimed_label"),
                   fixed.get("severe_clean_examples_each"),
                   fixed.get("severe_conflict_examples_each"),
                   fixed.get("severe_off_task_examples_each")),
                  (2, 3, 128, 64, 103, 102, 51),
                  "matched.fixed_configuration")
    _expect_equal(matched_analysis.get("classification"),
                  "prospective secondary mechanistic addendum",
                  "matched.analysis.classification")
    _expect_equal(
        [(item.get("id"), item.get("left"), item.get("right"),
          item.get("metric"))
         for item in matched_analysis.get("secondary_holm_family", [])],
        [("routed_k_minus_random_k", "severe_routed_k", "severe_random_k",
          "accuracy"),
         ("oracle_max_purity_k_minus_routed_k", "oracle_max_purity_k",
          "severe_routed_k", "accuracy")],
        "matched.analysis.secondary_holm_family")
    _expect_tokens(matched_analysis.get("multiplicity"), ("holm", "two", "0.05"),
                   "matched.analysis.multiplicity")
    _expect_tokens(matched_analysis.get("confidence_intervals"),
                   ("95%", "paired", "secondary"),
                   "matched.analysis.confidence_intervals")

    ag_section, ag_analysis = (ag.get("secondary_holdout", {}),
                               ag.get("analysis", {}))
    _expect_equal(ag_section.get("seeds"), list(TASKS["agnews"]["seeds"]),
                  "ag.secondary.seeds")
    _expect_equal(ag_section.get("comparators"), list(FOUR_ARMS),
                  "ag.secondary.comparators")
    _expect_equal((ag_analysis.get("primary_metric"), ag_analysis.get("paired_unit")),
                  ("accuracy", "common seed"), "ag.analysis.metric_and_unit")
    _expect_tokens(ag_analysis.get("tests"), ("two-sided", "paired"),
                   "ag.analysis.tests")
    _expect_tokens(ag_analysis.get("multiplicity"), ("holm", "three"),
                   "ag.analysis.multiplicity")
    _expect_tokens(ag_analysis.get("sensitivity"),
                   ("exact", "sign-flip", "integer"),
                   "ag.analysis.sensitivity")
    _expect_equal(
        [(item.get("left"), item.get("right"))
         for item in ag_analysis.get("adjacent_contrasts", [])],
        [(left, right) for _, left, right in ADJACENT],
        "ag.analysis.adjacent_contrasts")
    _require(any("post-primary" in item.lower() for item in
                 ag.get("evidence_boundaries", [])) and
             any("303" in item for item in ag.get("evidence_boundaries", [])),
             "AG post-primary resampling boundary is incomplete")

    return {
        "trec": {"name": trec["name"], "status": trec["status"],
                 "confirmatory": False, "role": trec["role"]},
        "imdb_original": {"name": imdb["name"], "status": imdb["status"],
                          "primary_alpha": primary["alpha"]},
        "imdb_matched_k": {"name": matched["name"], "status": matched["status"],
                           "matched_k": fixed["matched_k"],
                           "expected_client_records": (
                               len(TASKS["imdb"]["seeds"]) * fixed["rounds"] *
                               len(fixed["client_models"]))},
        "agnews": {"name": ag["name"], "status": ag["status"],
                   "scope": "post-primary resampling"},
    }


def _expected_names(seeds: tuple[int, ...], methods: tuple[str, ...]) -> set[str]:
    return {f"seed{seed}_{method}.json" for seed in seeds for method in methods}


def _check_exact_matrix(directory: Path, seeds: tuple[int, ...],
                        methods: tuple[str, ...], expected_count: int,
                        label: str) -> None:
    """Count filenames only; never deserialize an incomplete matrix."""
    expected = _expected_names(seeds, methods)
    observed = {path.name for path in directory.glob("*.json")}
    _require(len(observed) == expected_count and observed == expected,
             f"{label} must contain exactly {expected_count} records; "
             f"missing={sorted(expected - observed)}, "
             f"extra={sorted(observed - expected)}")
    _require(not list(directory.glob("*.tmp*")),
             f"{label} contains temporary records")


def _positive_gate(gate: dict, phase: str, count_key: str,
                   expected_count: int, label: str,
                   expected_protocol_sha256: str,
                   expected_seeds: tuple[int, ...],
                   expected_methods: tuple[str, ...]) -> None:
    _require(isinstance(gate, dict) and gate.get("phase") == phase and
             gate.get(count_key) == expected_count,
             f"{label} verifier did not certify {expected_count} records")
    _require(gate.get("protocol_sha256") == expected_protocol_sha256,
             f"{label} verifier protocol digest does not match the local "
             "frozen protocol")
    _require(gate.get("seeds") == list(expected_seeds) and
             gate.get("methods") == list(expected_methods),
             f"{label} verifier certified the wrong seed/method identity")


def _load_records(directory: Path, seeds: tuple[int, ...],
                  methods: tuple[str, ...], label: str) -> dict:
    records = {}
    for seed in seeds:
        for method in methods:
            path = directory / f"seed{seed}_{method}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            _require(record.get("seed") == seed and record.get("method") == method,
                     f"{label} identity mismatch: {path.name}")
            accuracy = float(record.get("final", {}).get("accuracy", math.nan))
            _require(math.isfinite(accuracy) and 0 <= accuracy <= 1,
                     f"{label} invalid accuracy: {path.name}")
            records[(seed, method)] = record
    return records


def _arm_summaries(records: dict, seeds: tuple[int, ...],
                   methods: tuple[str, ...]) -> dict:
    summaries = {}
    for method in methods:
        values = [float(records[(seed, method)]["final"]["accuracy"])
                  for seed in seeds]
        summaries[method] = {
            "arm": ARM_LETTERS.get(method),
            "values": values,
            "mean": statistics.mean(values),
            "sd": statistics.stdev(values),
        }
    return summaries


def _correct_count(record: dict, context: str) -> int:
    matrix = record.get("final", {}).get("confusion_matrix")
    _require(isinstance(matrix, list) and matrix and
             all(isinstance(row, list) and len(row) == len(matrix)
                 for row in matrix),
             f"{context}: confusion_matrix must be nonempty and square")
    values = [value for row in matrix for value in row]
    _require(all(isinstance(value, int) and not isinstance(value, bool) and
                 value >= 0 for value in values),
             f"{context}: confusion_matrix counts must be nonnegative integers")
    total = sum(values)
    evaluation_examples = int(record.get("evaluation_examples", -1))
    _require(total == evaluation_examples,
             f"{context}: confusion_matrix total differs from evaluation_examples")
    correct = sum(matrix[index][index] for index in range(len(matrix)))
    accuracy = float(record["final"]["accuracy"])
    _require(math.isclose(correct / total, accuracy, rel_tol=0, abs_tol=1e-12),
             f"{context}: confusion_matrix trace differs from accuracy")
    return correct


def _contrast(records: dict, seeds: tuple[int, ...], contrast_id: str,
              left: str, right: str, inference: str,
              exact_counts: bool = False) -> dict:
    differences = [
        float(records[(seed, left)]["final"]["accuracy"]) -
        float(records[(seed, right)]["final"]["accuracy"])
        for seed in seeds
    ]
    center, low, high = mean_ci(differences)
    item = {
        "id": contrast_id,
        "left": left,
        "right": right,
        "seed_paired_differences": differences,
        "mean": center,
        "ci95_low": low,
        "ci95_high": high,
        "inference": inference,
        "paired_t_two_sided_p": None,
        "holm_adjusted_p": None,
    }
    if inference != "descriptive_no_p_by_frozen_protocol":
        item["paired_t_two_sided_p"] = paired_t_p(differences)
    if exact_counts:
        count_differences = []
        for seed in seeds:
            left_record = records[(seed, left)]
            right_record = records[(seed, right)]
            count_differences.append(
                _correct_count(left_record, f"{contrast_id}, seed {seed}, left") -
                _correct_count(right_record, f"{contrast_id}, seed {seed}, right"))
        item["integer_correct_count_differences"] = count_differences
        item["exact_two_sided_sign_flip_p"] = exact_sign_flip_p(count_differences)
    return item


def _task_statistics(task_id: str, records: dict) -> dict:
    spec = TASKS[task_id]
    if task_id in {"trec", "agnews"}:
        inference = "holm_family_of_three_frozen_adjacent_accuracy_contrasts"
    else:
        inference = "mixed_by_frozen_imdb_protocol"
    contrasts = []
    for contrast_id, left, right in ADJACENT:
        if task_id == "imdb" and contrast_id != "D-C":
            contrast_inference = "descriptive_no_p_by_frozen_protocol"
        elif task_id == "imdb":
            contrast_inference = "single_frozen_primary_two_sided_p"
        else:
            contrast_inference = inference
        contrasts.append(_contrast(
            records, spec["seeds"], contrast_id, left, right,
            contrast_inference,
            exact_counts=(task_id in {"trec", "agnews"} or
                          (task_id == "imdb" and contrast_id == "D-C"))))
    if task_id in {"trec", "agnews"}:
        adjusted = holm_adjust(
            [item["paired_t_two_sided_p"] for item in contrasts])
        for item, value in zip(contrasts, adjusted):
            item["holm_adjusted_p"] = value
    return {
        "task": spec["name"],
        "evidence_scope": spec["scope"],
        "phase": spec["phase"],
        "seeds": list(spec["seeds"]),
        "arm_accuracy": _arm_summaries(records, spec["seeds"], FOUR_ARMS),
        "adjacent_accuracy_contrasts": contrasts,
    }


def _selection_composition(records: dict, seeds: tuple[int, ...], method: str,
                           expected_k: int | None, label: str) -> dict:
    counts: Counter[str] = Counter()
    client_records = 0
    for seed in seeds:
        rounds = records[(seed, method)].get("round_records", [])
        _require(isinstance(rounds, list) and rounds,
                 f"{label}: round_records is empty at seed {seed}")
        for round_record in rounds:
            clients = round_record.get("client_records", [])
            _require(isinstance(clients, list),
                     f"{label}: client_records is malformed at seed {seed}")
            for client in clients:
                raw = client.get("selected_kind_counts_audit_only")
                _require(isinstance(raw, dict) and raw,
                         f"{label}: selected composition is empty at seed {seed}")
                unknown = set(raw) - KNOWN_COMPOSITION_KINDS
                _require(not unknown,
                         f"{label}: unknown composition kinds: {sorted(unknown)}")
                _require(all(isinstance(value, int) and not isinstance(value, bool)
                             and value >= 0 for value in raw.values()),
                         f"{label}: composition counts must be nonnegative integers")
                selected = sum(raw.values())
                if expected_k is not None:
                    _require(selected == expected_k,
                             f"{label}: selected {selected}, expected K={expected_k}")
                counts.update(raw)
                client_records += 1
    _require(client_records > 0 and sum(counts.values()) > 0,
             f"{label}: no auditable selected composition")
    total = sum(counts.values())
    return {
        "counts": {kind: counts[kind] for kind in sorted(KNOWN_COMPOSITION_KINDS)},
        "fractions": {kind: counts[kind] / total
                      for kind in sorted(KNOWN_COMPOSITION_KINDS)},
        "total_selected": total,
        "client_records": client_records,
        "expected_k_per_client": expected_k,
    }


def _matched_statistics(records: dict, matched_k: int = 128,
                        expected_client_records: int = 60) -> dict:
    summaries = _arm_summaries(records, TASKS["imdb"]["seeds"], MATCHED_ARMS)
    compositions = {}
    for method in MATCHED_ARMS:
        compositions[method] = _selection_composition(
            records, TASKS["imdb"]["seeds"], method, matched_k,
            f"IMDb matched-K {method}")
        _require(compositions[method]["client_records"] == expected_client_records and
                 compositions[method]["total_selected"] ==
                 expected_client_records * matched_k,
                 f"IMDb matched-K {method}: aggregate K total changed")
    pairs = (
        ("routed_k_minus_random_k", "severe_routed_k", "severe_random_k"),
        ("maximum_purity_k_minus_routed_k", "oracle_max_purity_k", "severe_routed_k"),
    )
    contrasts = [_contrast(
        records, TASKS["imdb"]["seeds"], contrast_id, left, right,
        "secondary_holm_family_of_two") for contrast_id, left, right in pairs]
    for item, value in zip(
            contrasts,
            holm_adjust([item["paired_t_two_sided_p"] for item in contrasts])):
        item["holm_adjusted_p"] = value
    return {
        "classification": "prospective_secondary_mechanistic_addendum",
        "seeds": list(TASKS["imdb"]["seeds"]),
        "arm_accuracy": summaries,
        "selection_composition": compositions,
        "secondary_holm_family": contrasts,
    }


def compute_statistics(trec_records: dict, imdb_records: dict,
                       matched_records: dict, ag_records: dict,
                       equivalence_certified: bool = True,
                       matched_k: int = 128,
                       matched_expected_client_records: int = 60) -> dict:
    tasks = {
        "trec": _task_statistics("trec", trec_records),
        "imdb": _task_statistics("imdb", imdb_records),
        "agnews": _task_statistics("agnews", ag_records),
    }
    trec_dc = tasks["trec"]["adjacent_accuracy_contrasts"][2]["mean"]
    ag_dc = tasks["agnews"]["adjacent_accuracy_contrasts"][2]["mean"]
    tasks["agnews"]["direction_matches_trec_dc"] = (
        (trec_dc > 0) - (trec_dc < 0) == (ag_dc > 0) - (ag_dc < 0))
    tasks["agnews"]["direction_comparison"] = {
        "trec_dc_mean": trec_dc,
        "agnews_dc_mean": ag_dc,
        "definition": "sign(AG D-C mean) equals sign(TREC D-C mean)",
    }
    imdb_dc = tasks["imdb"]["adjacent_accuracy_contrasts"][2]
    routed_composition = _selection_composition(
        imdb_records, TASKS["imdb"]["seeds"],
        "severe_routed_distillation", None, "IMDb original routed")
    clean_fraction = routed_composition["fractions"]["clean"]
    gate_components = {
        "routed_clean_fraction_gt_103_over_256": bool(clean_fraction > 103 / 256),
        "dc_ci95_strictly_negative": bool(imdb_dc["ci95_high"] < 0),
        "dc_two_sided_p_lt_0_05": bool(
            imdb_dc["paired_t_two_sided_p"] < 0.05),
        "routed_equivalence_verifier_certified": bool(equivalence_certified),
    }
    imdb_gate = {
        "classification": "frozen prospective primary claim gate",
        "threshold_clean_fraction": 103 / 256,
        "observed_routed_clean_fraction": clean_fraction,
        "components": gate_components,
        "passed": all(gate_components.values()),
        "claim_status": ("narrow routing-harm pattern confirmed"
                         if all(gate_components.values()) else
                         "narrow routing-harm pattern not confirmed"),
        "failure_does_not_suppress_complete_results": True,
    }
    return {
        "schema_version": "server_audit_assets_v2",
        "accuracy_unit": "fraction; TeX renders percentage points",
        "paired_unit": "common frozen seed within task",
        "evidence_order": ["post-primary reused-test discovery",
                           "prospective role-reversal",
                           "post-primary resampling"],
        "tasks": tasks,
        "imdb_primary_gate": imdb_gate,
        "imdb_original_routed_composition": routed_composition,
        "imdb_matched_k": _matched_statistics(
            matched_records, matched_k, matched_expected_client_records),
    }


def _p_text(item: dict) -> str:
    if item["inference"] == "descriptive_no_p_by_frozen_protocol":
        return r"\textit{descriptive; no $p$}"
    value = (item["holm_adjusted_p"] if item["holm_adjusted_p"] is not None
             else item["paired_t_two_sided_p"])
    symbol = r"p_{\mathrm{H}}" if item["holm_adjusted_p"] is not None else "p"
    return _p_value_tex(value, symbol)


def _p_value_tex(value: float, symbol: str = "p") -> str:
    operator = "<" if value < 0.0001 else "="
    rendered = r"10^{-4}" if value < 0.0001 else f"{value:.4f}"
    return rf"${symbol}{operator}{rendered}$"


def _effect_text(item: dict) -> str:
    classification = _inference_classification(item)
    return (f"{100 * item['mean']:+.2f} "
            f"[{100 * item['ci95_low']:+.2f},{100 * item['ci95_high']:+.2f}]; "
            f"{_p_text(item)} ({classification})")


def _inference_classification(item: dict) -> str:
    return {
        "descriptive_no_p_by_frozen_protocol": "descriptive",
        "single_frozen_primary_two_sided_p": "primary",
        "holm_family_of_three_frozen_adjacent_accuracy_contrasts":
            "post-primary Holm",
        "secondary_holm_family_of_two": "secondary Holm",
    }[item["inference"]]


def _render_table(ledger: dict) -> str:
    lines = [
        "% Auto-generated only after all 150 official/holdout records verify.",
        r"\begin{table*}[t]",
        r"\centering\scriptsize",
        r"\caption{Cross-task server audit. Arm cells are accuracy mean $\pm$ SD (\%). Effects are paired mean percentage points [95\% CI]. IMDb $B-A$ and $C-B$ remain descriptive because its frozen protocol authorizes only the $D-C$ primary test.}",
        r"\label{tab:server-audit}",
        r"\resizebox{0.98\textwidth}{!}{%",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Task & Evidence scope & A: anchor & B: clean uniform & C: severe uniform & D: severe routed \\",
        r"\midrule",
    ]
    for task_id in ("trec", "imdb", "agnews"):
        task = ledger["tasks"][task_id]
        arm_cells = []
        for method in FOUR_ARMS:
            arm = task["arm_accuracy"][method]
            arm_cells.append(f"{100*arm['mean']:.2f} $\\pm$ {100*arm['sd']:.2f}")
        lines.append(" & ".join(
            [task["task"], task["evidence_scope"], *arm_cells]) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%", r"}",
        r"\par\vspace{2pt}",
        r"\resizebox{0.98\textwidth}{!}{%",
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r"Task & Evidence timing & Contrast & Paired mean pp [95\% CI] & Inference \\",
        r"\midrule",
    ])
    for task_id in ("trec", "imdb", "agnews"):
        task = ledger["tasks"][task_id]
        for item in task["adjacent_accuracy_contrasts"]:
            effect = (f"{100*item['mean']:+.2f} "
                      f"[{100*item['ci95_low']:+.2f},"
                      f"{100*item['ci95_high']:+.2f}]")
            inference = (f"{_p_text(item)} "
                         f"({_inference_classification(item)})")
            lines.append(" & ".join(
                [task["task"], task["evidence_scope"], item["id"],
                 effect, inference]) + r" \\")
    lines.extend([
        r"\bottomrule", r"\end{tabular}%", r"}",
        r"\par\vspace{2pt}",
        r"\resizebox{0.94\textwidth}{!}{%",
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r"IMDb matched-$K$ secondary arm & Accuracy (\%) & Clean/conflict/off-task (\%) & Frozen paired effect [95\% CI] & Holm $p$ \\",
        r"\midrule",
    ])
    matched = ledger["imdb_matched_k"]
    gate = ledger["imdb_primary_gate"]
    imdb_dc = ledger["tasks"]["imdb"]["adjacent_accuracy_contrasts"][2]
    secondary_by_left = {item["left"]: item
                         for item in matched["secondary_holm_family"]}
    matched_lines = []
    for method in MATCHED_ARMS:
        arm = matched["arm_accuracy"][method]
        comp = matched["selection_composition"][method]["fractions"]
        item = secondary_by_left.get(method)
        effect = (_effect_text(item) if item is not None else
                  r"\textit{reference arm}")
        holm = (_p_text(item) if item is not None else "---")
        matched_lines.append(
            f"{MATCHED_LABELS[method]} & {100*arm['mean']:.2f} $\\pm$ "
            f"{100*arm['sd']:.2f} & {100*comp['clean']:.1f}/"
            f"{100*comp['conflict']:.1f}/{100*comp['off_task']:.1f} & "
            f"{effect} & {holm} " + r"\\")
    lines.extend(matched_lines + [
        r"\bottomrule", r"\end{tabular}%", r"}",
        (r"\par\scriptsize IMDb frozen primary gate: \textbf{" +
         ("passed" if gate["passed"] else "not passed") +
         f"}}; routed clean {100*gate['observed_routed_clean_fraction']:.1f}\\% "
         f"$>$ {100*gate['threshold_clean_fraction']:.1f}\\%, "
         f"$D-C$ CI upper {100*imdb_dc['ci95_high']:+.2f} pp, "
         f"{_p_value_tex(imdb_dc['paired_t_two_sided_p'])}, routed equivalence "
         + ("certified" if gate["components"][
             "routed_equivalence_verifier_certified"] else "not certified") + "."),
        r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def _nice_extent(values: list[float]) -> float:
    maximum = max([0.5, *(abs(value) for value in values)]) * 1.12
    step = 0.25 if maximum <= 1.0 else (0.5 if maximum <= 2.5 else 1.0)
    return math.ceil(maximum / step) * step


def _map_x(value: float, extent: float, x0: float, width: float) -> float:
    return x0 + width * (value + extent) / (2 * extent)


def _render_figure(ledger: dict) -> str:
    def inference_color(item: dict) -> str:
        if item["inference"] == "descriptive_no_p_by_frozen_protocol":
            return "kafagray"
        if item["inference"] == "single_frozen_primary_two_sided_p":
            return "kafablue"
        return "kafaorange"

    rows = []
    limits = []
    for task_id in ("trec", "imdb", "agnews"):
        task = ledger["tasks"][task_id]
        for item in task["adjacent_accuracy_contrasts"]:
            rows.append((task, item))
            limits.extend([100 * item["ci95_low"], 100 * item["ci95_high"],
                           *(100 * value for value in item["seed_paired_differences"])])
    extent = _nice_extent(limits)
    x0, width = 2.7, 5.15
    lines = [
        r"\pdfinfoomitdate=1", r"\pdftrailerid{}",
        r"\documentclass[tikz,border=2pt]{standalone}",
        r"\usepackage{newtxtext}", r"\usepackage{newtxmath}",
        r"\definecolor{kafapurple}{RGB}{117,72,141}",
        r"\definecolor{kafablue}{RGB}{38,92,140}",
        r"\definecolor{kafagreen}{RGB}{55,126,92}",
        r"\definecolor{kafateal}{RGB}{30,132,137}",
        r"\definecolor{kafaorange}{RGB}{221,132,43}",
        r"\definecolor{kafagray}{RGB}{166,166,166}",
        r"\begin{document}",
        r"\begin{tikzpicture}[font=\sffamily\scriptsize]",
        r"\node[anchor=west,font=\sffamily\bfseries] at (0,7.10) {(a) Seed-paired adjacent effects by evidence class};",
    ]
    zero = _map_x(0, extent, x0, width)
    lines.append(rf"\draw[densely dashed,kafagray] ({zero:.3f},1.10)--({zero:.3f},6.65);")
    for index, (task, item) in enumerate(rows):
        y = 6.35 - index * 0.58
        color = inference_color(item)
        label = f"{task['task']} {item['id']}"
        for dot_index, difference in enumerate(item["seed_paired_differences"]):
            xpos = _map_x(100 * difference, extent, x0, width)
            jitter = (dot_index - 4.5) * 0.012
            lines.append(rf"\fill[{color},opacity=.28] ({xpos:.3f},{y+jitter:.3f}) circle (0.75pt);")
        lo = _map_x(100 * item["ci95_low"], extent, x0, width)
        hi = _map_x(100 * item["ci95_high"], extent, x0, width)
        center = _map_x(100 * item["mean"], extent, x0, width)
        lines.extend([
            rf"\node[anchor=east] at ({x0-0.10:.3f},{y:.3f}) {{{label}}};",
            rf"\draw[line width=.85pt,{color}] ({lo:.3f},{y:.3f})--({hi:.3f},{y:.3f});",
            rf"\draw[{color}] ({lo:.3f},{y-.07:.3f})--({lo:.3f},{y+.07:.3f}) ({hi:.3f},{y-.07:.3f})--({hi:.3f},{y+.07:.3f});",
            rf"\fill[{color}] ({center:.3f},{y:.3f}) circle (2pt);",
        ])
    axis_y = 0.90
    lines.append(rf"\draw[->] ({x0:.3f},{axis_y:.3f})--({x0+width+.14:.3f},{axis_y:.3f});")
    for tick in (-extent, -extent/2, 0, extent/2, extent):
        xpos = _map_x(tick, extent, x0, width)
        lines.append(rf"\draw ({xpos:.3f},{axis_y-.05:.3f})--({xpos:.3f},{axis_y+.05:.3f}) node[below=2pt] {{{tick:g}}};")
    lines.extend([
        rf"\node at ({x0+width/2:.3f},0.30) {{paired accuracy difference (pp)}};",
        r"\fill[kafaorange] (0.15,0.50) circle (2pt); \node[anchor=west] at (.30,.50) {post-primary};",
        r"\fill[kafablue] (1.85,0.50) circle (2pt); \node[anchor=west] at (2.00,.50) {primary};",
        r"\fill[kafagray] (2.95,0.50) circle (2pt); \node[anchor=west] at (3.10,.50) {descriptive};",
        r"\node[anchor=west,font=\sffamily\bfseries] at (8.55,7.10) {(b) IMDb matched-$K$ composition and accuracy};",
        r"\fill[kafablue] (9.20,6.55) rectangle +(0.25,.18); \node[anchor=west] at (9.50,6.64) {clean};",
        r"\fill[kafaorange] (10.65,6.55) rectangle +(0.25,.18); \node[anchor=west] at (10.95,6.64) {conflict};",
        r"\fill[kafagray] (12.35,6.55) rectangle +(0.25,.18); \node[anchor=west] at (12.65,6.64) {off-task};",
    ])
    matched = ledger["imdb_matched_k"]
    bx, bw = 9.55, 4.55
    for index, method in enumerate(MATCHED_ARMS):
        y = 5.65 - index * 1.45
        composition = matched["selection_composition"][method]["fractions"]
        clean = composition["clean"]
        conflict = composition["conflict"]
        off_task = composition["off_task"]
        a, b, c = bx + bw*clean, bx + bw*(clean+conflict), bx + bw*(clean+conflict+off_task)
        accuracy = matched["arm_accuracy"][method]
        lines.extend([
            rf"\node[anchor=east] at ({bx-.10:.3f},{y:.3f}) {{{MATCHED_LABELS[method]}}};",
            rf"\fill[kafablue] ({bx:.3f},{y-.20:.3f}) rectangle ({a:.3f},{y+.20:.3f});",
            rf"\fill[kafaorange] ({a:.3f},{y-.20:.3f}) rectangle ({b:.3f},{y+.20:.3f});",
            rf"\fill[kafagray] ({b:.3f},{y-.20:.3f}) rectangle ({c:.3f},{y+.20:.3f});",
            rf"\draw ({bx:.3f},{y-.20:.3f}) rectangle ({bx+bw:.3f},{y+.20:.3f});",
            rf"\node[anchor=west] at ({bx+bw+.10:.3f},{y:.3f}) {{{100*accuracy['mean']:.2f}$\pm${100*accuracy['sd']:.2f}\% acc.}};",
        ])
    lines.append(rf"\draw[->] ({bx:.3f},1.15)--({bx+bw+.15:.3f},1.15);")
    for tick in (0, 25, 50, 75, 100):
        xpos = bx + bw * tick / 100
        lines.append(rf"\draw ({xpos:.3f},1.10)--({xpos:.3f},1.20) node[below=2pt] {{{tick}}};")
    lines.extend([
        rf"\node at ({bx+bw/2:.3f},0.75) {{aggregate selected-set share (\%)}};",
    ])
    for index, item in enumerate(matched["secondary_holm_family"]):
        label = (r"Routed-$K$ $-$ random-$K$" if index == 0 else
                 r"Maximum-purity-$K$ $-$ routed-$K$")
        p_value = item["holm_adjusted_p"]
        p_rendered = (r"p_{\mathrm H}<10^{-4}" if p_value < 0.0001 else
                      rf"p_{{\mathrm H}}={p_value:.4f}")
        lines.append(
            rf"\node[anchor=west,text=kafateal] at (8.65,{0.43-index*.28:.2f}) "
            rf"{{secondary: {label}: {100*item['mean']:+.2f} "
            rf"[{100*item['ci95_low']:+.2f},{100*item['ci95_high']:+.2f}] pp, "
            rf"${p_rendered}$}};")
    gate = ledger["imdb_primary_gate"]
    imdb_dc = ledger["tasks"]["imdb"]["adjacent_accuracy_contrasts"][2]
    gate_p = _p_value_tex(imdb_dc["paired_t_two_sided_p"])
    ag_match = ledger["tasks"]["agnews"]["direction_matches_trec_dc"]
    lines.extend([
        rf"\node[anchor=west] at (8.65,-0.20) {{IMDb primary gate: "
        rf"\textbf{{{'passed' if gate['passed'] else 'not passed'}}}; clean "
        rf"{100*gate['observed_routed_clean_fraction']:.1f}$>${100*gate['threshold_clean_fraction']:.1f}\%, "
        rf"CI upper {100*imdb_dc['ci95_high']:+.2f} pp, "
        rf"{gate_p}, equiv. yes}};",
        rf"\node[anchor=west] at (0,-0.20) {{AG $D-C$ direction matches TREC: "
        rf"\textbf{{{'yes' if ag_match else 'no'}}}}};",
        r"\end{tikzpicture}", r"\end{document}",
    ])
    return "\n".join(lines) + "\n"


def _compile_figure(stage: Path) -> None:
    """Compile both production TeX assets, rejecting every diagnostic."""
    (stage / "server_audit_table_compile.tex").write_text(
        r"\documentclass{article}" "\n"
        r"\usepackage[margin=0.25in]{geometry}" "\n"
        r"\usepackage{booktabs}" "\n"
        r"\usepackage{graphicx}" "\n"
        r"\begin{document}" "\n"
        r"\input{server_audit_table.tex}" "\n"
        r"\end{document}" "\n", encoding="utf-8")
    sources = (("server_audit_figure.tex", 1),
               ("server_audit_table_compile.tex", 2))
    for source, passes in sources:
        completed = None
        for _ in range(passes):
            completed = subprocess.run(
                ["pdflatex", "-halt-on-error", "-interaction=batchmode", source],
                cwd=stage, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"server audit TeX compilation failed for {source}:\n" +
                    completed.stdout)
        log = (stage / f"{Path(source).stem}.log").read_text(
            encoding="utf-8", errors="replace")
        diagnostic = re.search(
            r"(?:LaTeX|Package \S+) Warning|Overfull |Underfull |"
            r"Undefined control sequence", log)
        if diagnostic:
            raise RuntimeError(
                f"server audit {source} has TeX diagnostic: "
                f"{diagnostic.group(0)}")
    pdf = stage / "server_audit_figure.pdf"
    _require(pdf.is_file() and pdf.stat().st_size > 1000,
             "server audit figure PDF is absent or empty")


def _publish(stage: Path, output_dir: Path) -> None:
    """Publish from a complete stage; rollback target files on any failure."""
    output_existed = output_dir.exists()
    output_dir.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix=".server-audit-backup-",
                                  dir=output_dir.parent))
    replaced = []
    try:
        for name in ASSET_NAMES:
            target = output_dir / name
            if target.exists():
                os.replace(target, backup / name)
            os.replace(stage / name, target)
            replaced.append(name)
    except BaseException:
        for name in replaced:
            target = output_dir / name
            if target.exists():
                target.unlink()
        for saved in backup.iterdir():
            os.replace(saved, output_dir / saved.name)
        if not output_existed and not any(output_dir.iterdir()):
            output_dir.rmdir()
        raise
    finally:
        shutil.rmtree(backup, ignore_errors=True)


def _current_input_snapshot(trec_dir: Path, imdb_dir: Path, matched_dir: Path,
                            ag_dir: Path, trec_protocol: Path,
                            imdb_protocol: Path, matched_protocol: Path,
                            ag_protocol: Path) -> dict:
    return {
        "schema_version": "server_audit_current_inputs_v1",
        "generator_sha256": _sha256(Path(__file__)),
        "protocol_sha256": {
            "trec": _sha256(trec_protocol),
            "imdb_original": _sha256(imdb_protocol),
            "imdb_matched_k": _sha256(matched_protocol),
            "agnews": _sha256(ag_protocol),
        },
        "verifier_sha256": {
            "trec": _sha256(ROOT / "kafa_script/verify_trec_server_ablation.py"),
            "imdb_original": _sha256(ROOT / "kafa_script/verify_imdb_server_role_reversal.py"),
            "imdb_matched_k": _sha256(ROOT / "kafa_script/verify_imdb_matched_control_addendum.py"),
            "imdb_routed_equivalence": _sha256(ROOT / "kafa_script/verify_imdb_routed_equivalence.py"),
            "agnews": _sha256(ROOT / "kafa_script/verify_agnews_server_ablation.py"),
        },
        "matrix_sha256": {
            "trec": {path.name: _sha256(path) for path in sorted(trec_dir.glob("*.json"))},
            "imdb_original": {path.name: _sha256(path) for path in sorted(imdb_dir.glob("*.json"))},
            "imdb_matched_k": {path.name: _sha256(path) for path in sorted(matched_dir.glob("*.json"))},
            "agnews": {path.name: _sha256(path) for path in sorted(ag_dir.glob("*.json"))},
        },
    }


def verify_published_assets(
    trec_dir: Path,
    imdb_dir: Path,
    matched_dir: Path,
    ag_dir: Path,
    output_dir: Path,
    trec_protocol: Path = ROOT / "kafa_script/trec_server_post_primary_ablation_v1.json",
    imdb_protocol: Path = ROOT / "kafa_script/imdb_server_role_reversal_protocol_v1.json",
    matched_protocol: Path = ROOT / "kafa_script/imdb_matched_control_addendum_protocol_v1.json",
    ag_protocol: Path = ROOT / "kafa_script/agnews_server_post_primary_ablation_v1.json",
) -> dict:
    """Reject stale assets whose current protocols, records, or code differ."""
    trec_dir, imdb_dir, matched_dir, ag_dir, output_dir = map(
        Path, (trec_dir, imdb_dir, matched_dir, ag_dir, output_dir))
    trec_protocol, imdb_protocol, matched_protocol, ag_protocol = map(
        Path, (trec_protocol, imdb_protocol, matched_protocol, ag_protocol))
    _check_exact_matrix(trec_dir, TASKS["trec"]["seeds"], FOUR_ARMS, 40, "TREC")
    _check_exact_matrix(imdb_dir, TASKS["imdb"]["seeds"], FOUR_ARMS, 40,
                        "IMDb original")
    _check_exact_matrix(matched_dir, TASKS["imdb"]["seeds"], MATCHED_ARMS, 30,
                        "IMDb matched-K")
    _check_exact_matrix(ag_dir, TASKS["agnews"]["seeds"], FOUR_ARMS, 40,
                        "AG News")
    _bind_protocol_semantics(
        trec_protocol, imdb_protocol, matched_protocol, ag_protocol)
    manifest_path = output_dir / "server_audit_current_inputs.json"
    _require(manifest_path.is_file(),
             "server audit assets lack a current-input manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_inputs = _current_input_snapshot(
        trec_dir, imdb_dir, matched_dir, ag_dir, trec_protocol,
        imdb_protocol, matched_protocol, ag_protocol)
    _expect_equal(manifest.get("inputs"), expected_inputs,
                  "server_audit_current_inputs.inputs")
    published = manifest.get("published_asset_sha256")
    _require(isinstance(published, dict) and set(published) ==
             set(ASSET_NAMES) - {"server_audit_current_inputs.json"},
             "server audit manifest has an invalid published asset set")
    for name, digest in published.items():
        path = output_dir / name
        _require(path.is_file() and _sha256(path) == digest,
                 f"stale or missing server audit asset: {name}")
    return {"status": "current_server_audit_assets_verified",
            "assets": sorted(published), "records": 150}


def generate_assets(
    trec_dir: Path,
    imdb_dir: Path,
    matched_dir: Path,
    ag_dir: Path,
    output_dir: Path,
    trec_protocol: Path = ROOT / "kafa_script/trec_server_post_primary_ablation_v1.json",
    imdb_protocol: Path = ROOT / "kafa_script/imdb_server_role_reversal_protocol_v1.json",
    matched_protocol: Path = ROOT / "kafa_script/imdb_matched_control_addendum_protocol_v1.json",
    ag_protocol: Path = ROOT / "kafa_script/agnews_server_post_primary_ablation_v1.json",
    trec_verify: Callable = trec_verifier.verify_matrix,
    imdb_verify: Callable = imdb_verifier.verify_matrix,
    matched_verify: Callable = matched_verifier.verify_matrix,
    equivalence_verify: Callable = equivalence_verifier.verify_phase,
    ag_verify: Callable = ag_verifier.verify_matrix,
) -> dict:
    """Verify all complete matrices, then atomically publish final assets."""
    trec_dir, imdb_dir, matched_dir, ag_dir, output_dir = map(
        Path, (trec_dir, imdb_dir, matched_dir, ag_dir, output_dir))
    trec_protocol, imdb_protocol, matched_protocol, ag_protocol = map(
        Path, (trec_protocol, imdb_protocol, matched_protocol, ag_protocol))

    # All filename/count gates precede every verifier and every JSON read.
    _check_exact_matrix(trec_dir, TASKS["trec"]["seeds"], FOUR_ARMS, 40, "TREC")
    _check_exact_matrix(imdb_dir, TASKS["imdb"]["seeds"], FOUR_ARMS, 40,
                        "IMDb original")
    _check_exact_matrix(matched_dir, TASKS["imdb"]["seeds"], MATCHED_ARMS, 30,
                        "IMDb matched-K")
    _check_exact_matrix(ag_dir, TASKS["agnews"]["seeds"], FOUR_ARMS, 40,
                        "AG News")

    protocol_semantics = _bind_protocol_semantics(
        trec_protocol, imdb_protocol, matched_protocol, ag_protocol)
    protocol_hashes = {
        "trec": _sha256(trec_protocol),
        "imdb_original": _sha256(imdb_protocol),
        "imdb_matched_k": _sha256(matched_protocol),
        "agnews": _sha256(ag_protocol),
    }

    gates = {
        "trec": trec_verify("official_evaluation", trec_dir, trec_protocol),
        "imdb_original": imdb_verify(
            "official_evaluation", imdb_dir, imdb_protocol),
        "imdb_matched_k": matched_verify(
            "official_evaluation", matched_dir, matched_protocol),
        "imdb_routed_equivalence": equivalence_verify(
            "official_evaluation", imdb_dir, matched_dir),
        "agnews": ag_verify("secondary_holdout", ag_dir, ag_protocol),
    }
    _positive_gate(gates["trec"], "official_evaluation", "records", 40, "TREC",
                   protocol_hashes["trec"], TASKS["trec"]["seeds"], FOUR_ARMS)
    _positive_gate(gates["imdb_original"], "official_evaluation", "records", 40,
                   "IMDb original", protocol_hashes["imdb_original"],
                   TASKS["imdb"]["seeds"], FOUR_ARMS)
    _positive_gate(gates["imdb_matched_k"], "official_evaluation", "records", 30,
                   "IMDb matched-K", protocol_hashes["imdb_matched_k"],
                   TASKS["imdb"]["seeds"], MATCHED_ARMS)
    _require(gates["imdb_routed_equivalence"].get("phase") ==
             "official_evaluation" and
             gates["imdb_routed_equivalence"].get("pairs") == 10 and
             gates["imdb_routed_equivalence"].get("seeds") ==
             list(TASKS["imdb"]["seeds"]) and
             gates["imdb_routed_equivalence"].get("functional_equivalence") is True,
             "IMDb routed equivalence verifier did not certify ten pairs")
    _positive_gate(gates["agnews"], "secondary_holdout", "records", 40,
                   "AG News", protocol_hashes["agnews"],
                   TASKS["agnews"]["seeds"], FOUR_ARMS)

    records = {
        "trec": _load_records(trec_dir, TASKS["trec"]["seeds"], FOUR_ARMS, "TREC"),
        "imdb": _load_records(imdb_dir, TASKS["imdb"]["seeds"], FOUR_ARMS,
                              "IMDb original"),
        "matched": _load_records(matched_dir, TASKS["imdb"]["seeds"], MATCHED_ARMS,
                                 "IMDb matched-K"),
        "agnews": _load_records(ag_dir, TASKS["agnews"]["seeds"], FOUR_ARMS,
                                "AG News"),
    }
    ledger = compute_statistics(records["trec"], records["imdb"],
                                records["matched"], records["agnews"],
                                equivalence_certified=True,
                                matched_k=protocol_semantics[
                                    "imdb_matched_k"]["matched_k"],
                                matched_expected_client_records=protocol_semantics[
                                    "imdb_matched_k"]["expected_client_records"])
    ledger["imdb_primary_gate"]["equivalence_verifier_receipt"] = dict(
        gates["imdb_routed_equivalence"])
    gates["imdb_routed_equivalence"]["generator_bound_protocol_sha256"] = {
        "imdb_original": protocol_hashes["imdb_original"],
        "imdb_matched_k": protocol_hashes["imdb_matched_k"],
    }
    ledger["verification"] = gates
    ledger["frozen_protocol_semantics"] = protocol_semantics
    ledger["provenance"] = {
        "protocol_sha256": protocol_hashes,
        "verifier_sha256": {
            "trec": _sha256(ROOT / "kafa_script/verify_trec_server_ablation.py"),
            "imdb_original": _sha256(ROOT / "kafa_script/verify_imdb_server_role_reversal.py"),
            "imdb_matched_k": _sha256(ROOT / "kafa_script/verify_imdb_matched_control_addendum.py"),
            "imdb_routed_equivalence": _sha256(ROOT / "kafa_script/verify_imdb_routed_equivalence.py"),
            "agnews": _sha256(ROOT / "kafa_script/verify_agnews_server_ablation.py"),
        },
        "raw_record_sha256": {
            "trec": {path.name: _sha256(path) for path in sorted(trec_dir.glob("*.json"))},
            "imdb_original": {path.name: _sha256(path) for path in sorted(imdb_dir.glob("*.json"))},
            "imdb_matched_k": {path.name: _sha256(path) for path in sorted(matched_dir.glob("*.json"))},
            "agnews": {path.name: _sha256(path) for path in sorted(ag_dir.glob("*.json"))},
        },
    }
    current_inputs = _current_input_snapshot(
        trec_dir, imdb_dir, matched_dir, ag_dir, trec_protocol,
        imdb_protocol, matched_protocol, ag_protocol)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".server-audit-assets-",
                                 dir=output_dir.parent))
    try:
        (stage / "server_audit_table.tex").write_text(
            _render_table(ledger), encoding="utf-8")
        (stage / "server_audit_figure.tex").write_text(
            _render_figure(ledger), encoding="utf-8")
        (stage / "server_audit_ledger.json").write_text(
            json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _compile_figure(stage)
        publish_without_manifest = (
            set(ASSET_NAMES) - {"server_audit_current_inputs.json"})
        manifest = {
            "schema_version": "server_audit_published_manifest_v1",
            "inputs": current_inputs,
            "published_asset_sha256": {
                name: _sha256(stage / name)
                for name in sorted(publish_without_manifest)
            },
        }
        (stage / "server_audit_current_inputs.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        _publish(stage, output_dir)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return {
        "status": "generated_from_all_verified_complete_server_matrices",
        "records": {"trec": 40, "imdb_original": 40,
                    "imdb_matched_k": 30, "agnews": 40},
        "output_dir": str(output_dir),
        "assets": list(ASSET_NAMES),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trec-dir", type=Path, default=ROOT /
                        "kafa_script/analysis_outputs/trec_server_ablation/official_reused_test")
    parser.add_argument("--imdb-dir", type=Path, default=ROOT /
                        "kafa_script/analysis_outputs/imdb_server_role_reversal/official_evaluation")
    parser.add_argument("--matched-dir", type=Path, default=ROOT /
                        "kafa_script/analysis_outputs/imdb_matched_control_addendum/official_evaluation")
    parser.add_argument("--ag-dir", type=Path, default=ROOT /
                        "kafa_script/analysis_outputs/agnews_server_ablation/secondary_holdout")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/generated")
    parser.add_argument("--trec-protocol", type=Path, default=ROOT /
                        "kafa_script/trec_server_post_primary_ablation_v1.json")
    parser.add_argument("--imdb-protocol", type=Path, default=ROOT /
                        "kafa_script/imdb_server_role_reversal_protocol_v1.json")
    parser.add_argument("--matched-protocol", type=Path, default=ROOT /
                        "kafa_script/imdb_matched_control_addendum_protocol_v1.json")
    parser.add_argument("--ag-protocol", type=Path, default=ROOT /
                        "kafa_script/agnews_server_post_primary_ablation_v1.json")
    parser.add_argument(
        "--verify-only", action="store_true",
        help="fail unless published assets match every current frozen input")
    args = parser.parse_args()
    function = verify_published_assets if args.verify_only else generate_assets
    print(json.dumps(function(
        args.trec_dir, args.imdb_dir, args.matched_dir, args.ag_dir,
        args.output_dir, args.trec_protocol, args.imdb_protocol,
        args.matched_protocol, args.ag_protocol), indent=2))


if __name__ == "__main__":
    main()
