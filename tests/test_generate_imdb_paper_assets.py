import json
import math
import subprocess
from pathlib import Path

import pytest

from kafa_script.generate_imdb_paper_assets import (
    ASSET_NAMES,
    MATCHED_METHODS,
    ORIGINAL_METHODS,
    SEEDS,
    generate_assets,
)


def _record(seed: int, method: str, accuracy: float, counts: dict[str, int]) -> dict:
    clients = [] if method == "public_anchor_only" else [
        {"selected_kind_counts_audit_only": counts}
    ]
    return {
        "seed": seed,
        "method": method,
        "evaluation_examples": 10_000,
        "final": {
            "accuracy": accuracy,
            "macro_f1": accuracy - 0.001,
            "canonical_label_nll": 1.0 - accuracy,
            "ece": 0.05 + (accuracy - 0.8) / 10,
        },
        "round_records": [{"client_records": clients}],
    }


def _write_synthetic_matrices(
        root: Path, primary_gate_pass: bool = True) -> tuple[Path, Path]:
    original = root / "original"
    matched = root / "matched"
    original.mkdir()
    matched.mkdir()
    original_counts = {
        "public_anchor_only": {},
        "clean_unfiltered_distillation": {"clean": 256},
        "severe_unfiltered_distillation": {
            "clean": 103, "conflict": 102, "off_task": 51},
        "severe_routed_distillation": {"clean": 96, "conflict": 32},
    }
    matched_counts = {
        "severe_random_k": {"clean": 64, "conflict": 64},
        "severe_routed_k": {"clean": 96, "conflict": 32},
        "oracle_max_purity_k": {"clean": 103, "conflict": 25},
    }
    for index, seed in enumerate(SEEDS):
        original_accuracy = {
            "public_anchor_only": 0.805 + index * 0.0002,
            "clean_unfiltered_distillation": 0.815 + index * 0.0002,
            "severe_unfiltered_distillation": 0.800 + index * 0.0001,
            # Pass: frozen primary differences are -0.10, ..., -1.00 pp.
            # Fail: the same magnitudes point in the opposite direction.
            "severe_routed_distillation": (
                0.799 - index * 0.0009 if primary_gate_pass
                else 0.801 + index * 0.0011),
        }
        matched_accuracy = {
            "severe_random_k": 0.790 + index * 0.0001,
            "severe_routed_k": 0.795 + index * 0.0011,
            "oracle_max_purity_k": 0.797 + index * 0.0013,
        }
        for method in ORIGINAL_METHODS:
            path = original / f"seed{seed}_{method}.json"
            path.write_text(json.dumps(_record(
                seed, method, original_accuracy[method], original_counts[method])))
        for method in MATCHED_METHODS:
            path = matched / f"seed{seed}_{method}.json"
            path.write_text(json.dumps(_record(
                seed, method, matched_accuracy[method], matched_counts[method])))
    return original, matched


def _original_ok(*_args) -> dict:
    return {"phase": "official_evaluation", "records": 40,
            "protocol_sha256": "synthetic-original"}


def _matched_ok(*_args) -> dict:
    return {"phase": "official_evaluation", "records": 30,
            "protocol_sha256": "synthetic-matched"}


def _equivalence_ok(*_args) -> dict:
    return {"phase": "official_evaluation", "pairs": 10,
            "functional_equivalence": True}


def test_generator_fails_closed_on_incomplete_matrix(tmp_path: Path) -> None:
    original, matched = _write_synthetic_matrices(tmp_path)
    (original / "seed330_severe_routed_distillation.json").unlink()
    output = tmp_path / "assets"
    verifier_called = False

    def forbidden(*_args):
        nonlocal verifier_called
        verifier_called = True
        raise AssertionError("verifier must not run before exact-count gate")

    with pytest.raises(RuntimeError, match="exactly 40"):
        generate_assets(original, matched, output,
                        original_verify=forbidden,
                        matched_verify=forbidden,
                        equivalence_verify=forbidden)
    assert not verifier_called
    assert not output.exists()


def test_generator_fails_closed_when_verifier_rejects(tmp_path: Path) -> None:
    original, matched = _write_synthetic_matrices(tmp_path)
    output = tmp_path / "assets"

    def reject(*_args):
        raise AssertionError("synthetic verifier rejection")

    with pytest.raises(AssertionError, match="verifier rejection"):
        generate_assets(original, matched, output,
                        original_verify=reject,
                        matched_verify=_matched_ok,
                        equivalence_verify=_equivalence_ok)
    assert not output.exists()


@pytest.mark.parametrize("failed_gate", ["original", "matched", "equivalence"])
def test_generator_requires_positive_certification_from_every_gate(
        tmp_path: Path, failed_gate: str) -> None:
    original, matched = _write_synthetic_matrices(tmp_path)
    output = tmp_path / "assets"
    callbacks = {
        "original_verify": _original_ok,
        "matched_verify": _matched_ok,
        "equivalence_verify": _equivalence_ok,
    }
    callbacks[f"{failed_gate}_verify"] = lambda *_args: {}
    with pytest.raises(RuntimeError, match="did not certify"):
        generate_assets(original, matched, output, **callbacks)
    assert not output.exists()


def test_statistics_and_submission_assets_from_verified_matrices(
        tmp_path: Path) -> None:
    original, matched = _write_synthetic_matrices(tmp_path)
    output = tmp_path / "paper" / "generated"
    result = generate_assets(
        original, matched, output,
        original_verify=_original_ok,
        matched_verify=_matched_ok,
        equivalence_verify=_equivalence_ok)

    assert result["original_records"] == 40
    assert result["matched_records"] == 30
    assert set(result["assets"]) == set(ASSET_NAMES)
    assert all((output / name).is_file() for name in ASSET_NAMES)
    assert (output / "imdb_official_figure.pdf").read_bytes().startswith(b"%PDF-")

    audit = json.loads((output / "imdb_official_assets.json").read_text())
    primary = audit["primary_contrast"]
    assert primary["mean"] == pytest.approx(-0.0055)
    assert primary["ci95_low"] < primary["mean"] < primary["ci95_high"]
    assert 0 <= primary["paired_t_two_sided_p"] <= 1
    assert 0 <= primary["exact_two_sided_sign_flip_p"] <= 1
    assert primary["integer_correct_count_differences"] == [
        -10, -20, -30, -40, -50, -60, -70, -80, -90, -100]
    gate = audit["primary_confirmation_gate"]
    assert gate == {
        "ci_strictly_negative": True,
        "clean_fraction_strictly_above_baseline": True,
        "paired_t_two_sided_p_below_0_05": True,
        "passes": True,
        "protocol_clean_fraction_baseline": pytest.approx(103 / 256),
        "routed_clean_fraction": pytest.approx(96 / 128),
    }

    secondary = audit["matched_secondary_holm_family"]
    raw = [item["paired_t_two_sided_p"] for item in secondary]
    adjusted = [item["holm_adjusted_p"] for item in secondary]
    assert all(math.isfinite(value) and 0 <= value <= 1
               for value in raw + adjusted)
    assert all(adj >= p for adj, p in zip(adjusted, raw))
    assert audit["matched_composition"]["oracle_max_purity_k"][
        "fractions"]["clean"] == pytest.approx(103 / 128)

    macros = (output / "imdb_official_macros.tex").read_text()
    tables = (output / "imdb_official_tables.tex").read_text()
    assert r"\newcommand{\IMDbPrimaryGateStatus}{passes}" in macros
    assert r"\newcommand{\IMDbPrimaryDeltaPP}{-0.55}" in macros
    assert r"\newcommand{\IMDbArmAAccuracyPct}{80.59}" in macros
    assert r"\newcommand{\IMDbSevereBaselineCleanPct}{40.23}" in macros
    assert r"\newcommand{\IMDbSevereUniformCleanPct}{40.23}" in macros
    assert r"\newcommand{\IMDbMatchedMaxPurityCleanPct}{80.47}" in macros
    assert "$" not in macros
    assert "Anchor only" in tables
    assert r"Maximum-purity-$K$" in tables

    block = (output / "imdb_official_results_block.tex").read_text()
    assert r"gate \textbf{passes}" in block
    assert "secondary mechanistic tests" in block
    assert "primary confirmation" not in block.lower()
    assert r"\input{generated/imdb_official_tables.tex}" in block
    assert r"{generated/imdb_official_figure.pdf}" in block

    smoke = output.parent / "submission_asset_smoke.tex"
    smoke.write_text(
        r"\documentclass{article}" "\n"
        r"\usepackage{graphicx}" "\n"
        r"\usepackage{booktabs}" "\n"
        r"\begin{document}" "\n"
        r"\input{generated/imdb_official_results_block.tex}" "\n"
        r"\end{document}" "\n")
    compiled = subprocess.run(
        ["pdflatex", "-halt-on-error", "-interaction=batchmode",
         smoke.name], cwd=output.parent, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False)
    assert compiled.returncode == 0, compiled.stdout


def test_primary_gate_failure_is_reported_without_upgrading_secondary(
        tmp_path: Path) -> None:
    original, matched = _write_synthetic_matrices(
        tmp_path, primary_gate_pass=False)
    output = tmp_path / "paper" / "generated"
    generate_assets(
        original, matched, output,
        original_verify=_original_ok,
        matched_verify=_matched_ok,
        equivalence_verify=_equivalence_ok)

    audit = json.loads((output / "imdb_official_assets.json").read_text())
    gate = audit["primary_confirmation_gate"]
    assert gate["clean_fraction_strictly_above_baseline"] is True
    assert gate["ci_strictly_negative"] is False
    assert gate["paired_t_two_sided_p_below_0_05"] is True
    assert gate["passes"] is False
    macros = (output / "imdb_official_macros.tex").read_text()
    block = (output / "imdb_official_results_block.tex").read_text()
    assert r"\newcommand{\IMDbPrimaryGateStatus}{fails}" in macros
    assert r"gate \textbf{fails}" in block
    assert "secondary mechanistic tests" in block
    assert "primary confirmation" not in block.lower()
