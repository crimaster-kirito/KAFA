import json
import hashlib
import math
import subprocess
from pathlib import Path

import pytest

from kafa_script import generate_server_audit_assets as generator
from kafa_script.generate_server_audit_assets import (
    ASSET_NAMES,
    FOUR_ARMS,
    MATCHED_ARMS,
    TASKS,
    generate_assets,
)


def _record(seed: int, method: str, accuracy: float,
            counts: dict[str, int]) -> dict:
    clients = [] if method == "public_anchor_only" else [
        {"selected_kind_counts_audit_only": counts} for _ in range(3)
    ]
    evaluation_examples = 10_000
    correct = round(accuracy * evaluation_examples)
    return {
        "seed": seed,
        "method": method,
        "evaluation_examples": evaluation_examples,
        "final": {
            "accuracy": correct / evaluation_examples,
            "confusion_matrix": [[correct, 0],
                                 [evaluation_examples - correct, 0]],
        },
        "round_records": [{"client_records": clients} for _ in range(2)],
    }


def _write_matrix(directory: Path, seeds: tuple[int, ...],
                  methods: tuple[str, ...], task_offset: float = 0.0,
                  imdb_harm: bool = False) -> None:
    directory.mkdir(parents=True)
    counts = {
        "public_anchor_only": {},
        "clean_unfiltered_distillation": {"clean": 256},
        "severe_unfiltered_distillation": {
            "clean": 103, "conflict": 102, "off_task": 51},
        "severe_routed_distillation": {"clean": 96, "conflict": 32},
        "severe_random_k": {"clean": 64, "conflict": 64},
        "severe_routed_k": {"clean": 96, "conflict": 32},
        "oracle_max_purity_k": {"clean": 103, "conflict": 25},
    }
    for index, seed in enumerate(seeds):
        anchor = 0.80 + task_offset + 0.0002 * index
        values = {
            "public_anchor_only": anchor,
            "clean_unfiltered_distillation": anchor + 0.005 + 0.0002 * index,
            "severe_unfiltered_distillation": anchor - 0.005 + 0.0003 * index,
            "severe_routed_distillation": anchor - 0.003 + 0.0006 * index,
            "severe_random_k": anchor - 0.008 + 0.0001 * index,
            "severe_routed_k": anchor - 0.002 + 0.0005 * index,
            "oracle_max_purity_k": anchor + 0.001 + 0.0007 * index,
        }
        if imdb_harm and "severe_routed_distillation" in methods:
            values["severe_routed_distillation"] = (
                values["severe_unfiltered_distillation"] - 0.006 -
                0.0002 * index)
        for method in methods:
            (directory / f"seed{seed}_{method}.json").write_text(
                json.dumps(_record(seed, method, values[method], counts[method])),
                encoding="utf-8")


def _synthetic_inputs(root: Path) -> dict[str, Path]:
    paths = {
        "trec": root / "trec",
        "imdb": root / "imdb",
        "matched": root / "matched",
        "agnews": root / "agnews",
    }
    _write_matrix(paths["trec"], TASKS["trec"]["seeds"], FOUR_ARMS)
    _write_matrix(paths["imdb"], TASKS["imdb"]["seeds"], FOUR_ARMS, 0.03,
                  imdb_harm=True)
    _write_matrix(paths["matched"], TASKS["imdb"]["seeds"], MATCHED_ARMS, 0.03)
    _write_matrix(paths["agnews"], TASKS["agnews"]["seeds"], FOUR_ARMS, 0.06)
    protocols = _synthetic_protocols()
    for name, value in protocols.items():
        path = root / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    return paths


def _synthetic_protocols() -> dict[str, dict]:
    boundaries = [f"boundary {index}" for index in range(5)]
    adjacent = [{"left": left, "right": right}
                for _, left, right in generator.ADJACENT]
    return {
        "trec_protocol": {
            "name": "trec_server_post_primary_ablation_v1",
            "status": "refrozen_after_selected_composition_instrumentation_repair_before_any_valid_official_ablation_record",
            "confirmatory": False,
            "role": "post_primary_reused_official_test_mechanistic_ablation",
            "official_test_access_history": {"previously_accessed": True},
            "official_evaluation": {
                "seeds": list(TASKS["trec"]["seeds"]),
                "comparators": list(FOUR_ARMS),
                "primary_metric": "accuracy",
                "primary_contrasts": adjacent,
                "analysis": "two-sided paired tests, Holm, exact sign-flip",
                "net_effect": "descriptive and not a fourth test",
                "equivalence_policy": "No equivalence; nonsignificant is inconclusive",
            },
            "evidence_boundaries": boundaries,
            "purpose": "synthetic", "implementation": {}, "task": {},
            "data": {}, "selection_contract": {},
            "fixed_configuration": {}, "data": {},
        },
        "imdb_protocol": {
            "name": "imdb_server_role_reversal_protocol_v1",
            "status": "refrozen_after_pre_record_training_line_audit_repair_before_any_imdb_metric_or_test_json_decoding",
            "official_evaluation": {
                "seeds": list(TASKS["imdb"]["seeds"]),
                "comparators": list(FOUR_ARMS),
            },
            "analysis": {
                "single_primary_contrast": {
                    "id": "routing_accuracy_role_reversal",
                    "left": "severe_routed_distillation",
                    "right": "severe_unfiltered_distillation",
                    "metric": "accuracy", "alpha": 0.05,
                    "test": "two-sided paired Student test over ten seeds",
                    "confidence_interval": "95% paired interval",
                    "sensitivity": "exact sign-flip from integer counts",
                },
                "confirmation_gate": "clean 103/256; strictly negative; p<0.05",
                "descriptive_only": ["clean-minus-anchor and severe-minus-clean"],
                "prohibited": ["No additional p-values"],
            },
            "evidence_boundaries": boundaries,
            "purpose": "synthetic", "implementation": {}, "task": {},
            "data": {},
            "fixed_configuration": {},
        },
        "matched_protocol": {
            "name": "imdb_matched_control_addendum_v1",
            "status": "frozen_after_imdb_development_observation_before_any_matched_control_or_original_official_imdb_model_evaluation_with_pre_evaluation_cpu_contract_repairs",
            "official_evaluation": {
                "seeds": list(TASKS["imdb"]["seeds"]),
                "comparators": list(MATCHED_ARMS),
            },
            "fixed_configuration": {
                "rounds": 2,
                "client_models": ["client-a", "client-b", "client-c"],
                "matched_k": 128, "matched_quota_per_claimed_label": 64,
                "severe_clean_examples_each": 103,
                "severe_conflict_examples_each": 102,
                "severe_off_task_examples_each": 51,
            },
            "analysis": {
                "classification": "prospective secondary mechanistic addendum",
                "secondary_holm_family": [
                    {"id": "routed_k_minus_random_k", "left": "severe_routed_k",
                     "right": "severe_random_k", "metric": "accuracy"},
                    {"id": "oracle_max_purity_k_minus_routed_k",
                     "left": "oracle_max_purity_k", "right": "severe_routed_k",
                     "metric": "accuracy"},
                ],
                "multiplicity": "Holm over two tests at 0.05",
                "confidence_intervals": "95% paired secondary intervals",
            },
            "evidence_boundaries": boundaries,
            "purpose": "synthetic", "implementation": {}, "task": {},
            "data": {}, "selection_contract": {},
        },
        "ag_protocol": {
            "name": "agnews_server_post_primary_ablation_v1",
            "status": "refrozen_after_prior_training_exposure_audit_with_all_previous_records_invalidated",
            "secondary_holdout": {
                "seeds": list(TASKS["agnews"]["seeds"]),
                "comparators": list(FOUR_ARMS),
            },
            "analysis": {
                "primary_metric": "accuracy", "paired_unit": "common seed",
                "tests": "two-sided paired tests",
                "multiplicity": "one Holm family of three",
                "sensitivity": "exact sign-flip from integer counts",
                "adjacent_contrasts": adjacent,
            },
            "evidence_boundaries": [
                "This is post-primary evidence.",
                "303 IDs had earlier exposure.", *boundaries[:3],
            ],
            "purpose": "synthetic", "implementation": {},
            "fixed_configuration": {}, "data": {}, "task": {},
        },
    }


def _equivalence_ok(*_args) -> dict:
    return {"phase": "official_evaluation", "pairs": 10,
            "seeds": list(TASKS["imdb"]["seeds"]),
            "functional_equivalence": True}


def _call(paths: dict[str, Path], output: Path, **overrides):
    def gate(phase: str, records: int, seeds: tuple[int, ...],
             methods: tuple[str, ...]):
        def verify(_phase, _directory, protocol):
            assert _phase == phase
            digest = hashlib.sha256(Path(protocol).read_bytes()).hexdigest()
            return {"phase": phase, "records": records,
                    "protocol_sha256": digest, "seeds": list(seeds),
                    "methods": list(methods)}
        return verify

    callbacks = {
        "trec_verify": gate("official_evaluation", 40, TASKS["trec"]["seeds"],
                            FOUR_ARMS),
        "imdb_verify": gate("official_evaluation", 40, TASKS["imdb"]["seeds"],
                            FOUR_ARMS),
        "matched_verify": gate("official_evaluation", 30,
                               TASKS["imdb"]["seeds"], MATCHED_ARMS),
        "equivalence_verify": _equivalence_ok,
        "ag_verify": gate("secondary_holdout", 40, TASKS["agnews"]["seeds"],
                          FOUR_ARMS),
    }
    callbacks.update(overrides)
    return generate_assets(
        paths["trec"], paths["imdb"], paths["matched"], paths["agnews"],
        output, paths["trec_protocol"], paths["imdb_protocol"],
        paths["matched_protocol"], paths["ag_protocol"], **callbacks)


@pytest.mark.parametrize(
    ("matrix", "filename", "message"),
    [
        ("trec", "seed270_severe_routed_distillation.json", "TREC.*exactly 40"),
        ("imdb", "seed330_severe_routed_distillation.json", "IMDb original.*exactly 40"),
        ("matched", "seed330_oracle_max_purity_k.json", "IMDb matched-K.*exactly 30"),
        ("agnews", "seed290_severe_routed_distillation.json", "AG News.*exactly 40"),
    ],
)
def test_count_gates_fail_before_any_verifier_or_output(
        tmp_path: Path, matrix: str, filename: str, message: str) -> None:
    paths = _synthetic_inputs(tmp_path)
    (paths[matrix] / filename).unlink()
    output = tmp_path / "generated"
    calls = []

    def forbidden(*_args):
        calls.append(True)
        raise AssertionError("a verifier read an incomplete matrix")

    with pytest.raises(RuntimeError, match=message):
        _call(paths, output, trec_verify=forbidden, imdb_verify=forbidden,
              matched_verify=forbidden, equivalence_verify=forbidden,
              ag_verify=forbidden)
    assert calls == []
    assert not output.exists()


@pytest.mark.parametrize(
    "gate", ["trec_verify", "imdb_verify", "matched_verify",
             "equivalence_verify", "ag_verify"])
def test_every_verifier_rejection_is_fail_closed(
        tmp_path: Path, gate: str) -> None:
    paths = _synthetic_inputs(tmp_path)
    output = tmp_path / "generated"

    def reject(*_args):
        raise AssertionError(f"synthetic {gate} rejection")

    with pytest.raises(AssertionError, match="synthetic .* rejection"):
        _call(paths, output, **{gate: reject})
    assert not output.exists()


@pytest.mark.parametrize(
    ("gate", "bad_result", "message"),
    [
        ("trec_verify", {}, "TREC verifier did not certify"),
        ("imdb_verify", {}, "IMDb original verifier did not certify"),
        ("matched_verify", {}, "IMDb matched-K verifier did not certify"),
        ("equivalence_verify", {}, "equivalence verifier did not certify"),
        ("ag_verify", {}, "AG News verifier did not certify"),
    ],
)
def test_every_verifier_must_positively_certify(
        tmp_path: Path, gate: str, bad_result: dict, message: str) -> None:
    paths = _synthetic_inputs(tmp_path)
    output = tmp_path / "generated"
    with pytest.raises(RuntimeError, match=message):
        _call(paths, output, **{gate: lambda *_args: bad_result})
    assert not output.exists()


def test_protocol_shell_and_verifier_digest_mismatch_are_rejected(
        tmp_path: Path) -> None:
    paths = _synthetic_inputs(tmp_path)
    output = tmp_path / "generated"
    paths["ag_protocol"].write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="empty or incomplete shell"):
        _call(paths, output)
    assert not output.exists()

    paths = _synthetic_inputs(tmp_path / "digest")

    def wrong_digest(*_args):
        return {"phase": "official_evaluation", "records": 40,
                "protocol_sha256": "0" * 64}

    with pytest.raises(RuntimeError, match="protocol digest does not match"):
        _call(paths, output, trec_verify=wrong_digest)
    assert not output.exists()


@pytest.mark.parametrize("violation", ["unknown_kind", "wrong_k"])
def test_matched_composition_rejects_unknown_kinds_and_wrong_k(
        tmp_path: Path, violation: str) -> None:
    paths = _synthetic_inputs(tmp_path)
    path = paths["matched"] / "seed321_severe_random_k.json"
    record = json.loads(path.read_text())
    counts = record["round_records"][0]["client_records"][0][
        "selected_kind_counts_audit_only"]
    if violation == "unknown_kind":
        counts["mystery"] = 1
        message = "unknown composition kinds"
    else:
        counts["clean"] -= 1
        message = "expected K=128"
    path.write_text(json.dumps(record), encoding="utf-8")
    output = tmp_path / "generated"
    with pytest.raises(RuntimeError, match=message):
        _call(paths, output)
    assert not output.exists()


def test_primary_gate_failure_is_explicit_and_confusion_trace_is_authoritative(
        tmp_path: Path) -> None:
    paths = _synthetic_inputs(tmp_path)
    trec = generator._load_records(
        paths["trec"], TASKS["trec"]["seeds"], FOUR_ARMS, "TREC")
    imdb = generator._load_records(
        paths["imdb"], TASKS["imdb"]["seeds"], FOUR_ARMS, "IMDb")
    matched = generator._load_records(
        paths["matched"], TASKS["imdb"]["seeds"], MATCHED_ARMS, "matched")
    agnews = generator._load_records(
        paths["agnews"], TASKS["agnews"]["seeds"], FOUR_ARMS, "AG")
    for seed in TASKS["imdb"]["seeds"]:
        for round_record in imdb[(seed, "severe_routed_distillation")][
                "round_records"]:
            for client in round_record["client_records"]:
                client["selected_kind_counts_audit_only"] = {
                    "clean": 50, "conflict": 78}
    ledger = generator.compute_statistics(trec, imdb, matched, agnews)
    assert ledger["imdb_primary_gate"]["passed"] is False
    assert ledger["imdb_primary_gate"]["components"][
        "routed_clean_fraction_gt_103_over_256"] is False
    assert ledger["imdb_primary_gate"][
        "failure_does_not_suppress_complete_results"] is True

    broken = trec[(261, "clean_unfiltered_distillation")]["final"][
        "confusion_matrix"]
    broken[0][0] -= 1
    broken[1][0] += 1
    with pytest.raises(RuntimeError, match="trace differs from accuracy"):
        generator.compute_statistics(trec, imdb, matched, agnews)


def test_render_or_compile_failure_never_publishes_partial_assets(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _synthetic_inputs(tmp_path)
    output = tmp_path / "generated"

    def fail_compile(_stage: Path) -> None:
        raise RuntimeError("synthetic TeX compile failure")

    monkeypatch.setattr(generator, "_compile_figure", fail_compile)
    with pytest.raises(RuntimeError, match="synthetic TeX compile failure"):
        _call(paths, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".server-audit-assets-*"))


def test_publish_rolls_back_the_complete_asset_set_on_replace_failure(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stage = tmp_path / "stage"
    output = tmp_path / "generated"
    stage.mkdir()
    output.mkdir()
    for name in ASSET_NAMES:
        (stage / name).write_text(f"new:{name}", encoding="utf-8")
        (output / name).write_text(f"old:{name}", encoding="utf-8")

    real_replace = generator.os.replace
    failed = False

    def fail_once(source, target):
        nonlocal failed
        source = Path(source)
        if (not failed and source.parent == stage and
                source.name == "server_audit_figure.tex"):
            failed = True
            raise OSError("synthetic atomic replace failure")
        return real_replace(source, target)

    monkeypatch.setattr(generator.os, "replace", fail_once)
    with pytest.raises(OSError, match="synthetic atomic replace failure"):
        generator._publish(stage, output)
    assert failed
    for name in ASSET_NAMES:
        assert (output / name).read_text(encoding="utf-8") == f"old:{name}"
    assert not list(tmp_path.glob(".server-audit-backup-*"))


def test_statistics_atomic_assets_and_warning_free_tex_pdf(tmp_path: Path) -> None:
    paths = _synthetic_inputs(tmp_path)
    output = tmp_path / "generated"
    result = _call(paths, output)

    assert result["records"] == {
        "trec": 40, "imdb_original": 40, "imdb_matched_k": 30,
        "agnews": 40,
    }
    assert set(result["assets"]) == set(ASSET_NAMES)
    assert all((output / name).is_file() for name in ASSET_NAMES)
    assert (output / "server_audit_figure.pdf").read_bytes().startswith(b"%PDF-")

    ledger = json.loads((output / "server_audit_ledger.json").read_text())
    assert ledger["evidence_order"] == [
        "post-primary reused-test discovery", "prospective role-reversal",
        "post-primary resampling"]
    assert sum(len(values) for values in ledger["provenance"][
        "raw_record_sha256"].values()) == 150
    assert set(ledger["verification"]) == {
        "trec", "imdb_original", "imdb_matched_k",
        "imdb_routed_equivalence", "agnews",
    }

    trec = ledger["tasks"]["trec"]
    assert trec["arm_accuracy"]["public_anchor_only"]["mean"] == pytest.approx(
        0.8009)
    assert trec["adjacent_accuracy_contrasts"][0]["mean"] == pytest.approx(0.0059)
    assert trec["adjacent_accuracy_contrasts"][0]["ci95_low"] == pytest.approx(
        0.005466829882055663)
    assert trec["adjacent_accuracy_contrasts"][0]["ci95_high"] == pytest.approx(
        0.0063331701179443365)
    for item in trec["adjacent_accuracy_contrasts"]:
        assert item["ci95_low"] < item["mean"] < item["ci95_high"]
        assert 0 <= item["paired_t_two_sided_p"] <= 1
        assert item["holm_adjusted_p"] >= item["paired_t_two_sided_p"]
        assert len(item["integer_correct_count_differences"]) == 10
        assert 0 <= item["exact_two_sided_sign_flip_p"] <= 1
        assert item["exact_two_sided_sign_flip_p"] == pytest.approx(2 / 1024)
    assert trec["adjacent_accuracy_contrasts"][0][
        "integer_correct_count_differences"] == list(range(50, 70, 2))

    imdb_contrasts = ledger["tasks"]["imdb"]["adjacent_accuracy_contrasts"]
    assert imdb_contrasts[0]["paired_t_two_sided_p"] is None
    assert imdb_contrasts[1]["paired_t_two_sided_p"] is None
    assert imdb_contrasts[2]["inference"] == "single_frozen_primary_two_sided_p"
    assert math.isfinite(imdb_contrasts[2]["paired_t_two_sided_p"])
    assert len(imdb_contrasts[2]["integer_correct_count_differences"]) == 10
    assert imdb_contrasts[2]["mean"] == pytest.approx(-0.0069)
    gate = ledger["imdb_primary_gate"]
    assert gate["passed"] is True
    assert all(gate["components"].values())
    assert gate["observed_routed_clean_fraction"] == pytest.approx(96 / 128)
    assert gate["equivalence_verifier_receipt"]["pairs"] == 10
    assert ledger["verification"]["imdb_routed_equivalence"][
        "generator_bound_protocol_sha256"] == {
            "imdb_original": ledger["provenance"]["protocol_sha256"][
                "imdb_original"],
            "imdb_matched_k": ledger["provenance"]["protocol_sha256"][
                "imdb_matched_k"],
        }

    agnews = ledger["tasks"]["agnews"]
    assert agnews["evidence_scope"] == "post-primary resampling"
    assert agnews["direction_matches_trec_dc"] is True
    assert all("exact_two_sided_sign_flip_p" in item
               for item in agnews["adjacent_accuracy_contrasts"])

    matched = ledger["imdb_matched_k"]
    assert matched["selection_composition"]["oracle_max_purity_k"][
        "fractions"]["clean"] == pytest.approx(103 / 128)
    assert matched["selection_composition"]["oracle_max_purity_k"][
        "total_selected"] == 10 * 2 * 3 * 128
    assert len(matched["secondary_holm_family"]) == 2
    assert matched["secondary_holm_family"][0]["mean"] == pytest.approx(0.0078)
    assert matched["secondary_holm_family"][1]["mean"] == pytest.approx(0.0039)
    assert all(item["holm_adjusted_p"] >= item["paired_t_two_sided_p"]
               for item in matched["secondary_holm_family"])

    table = (output / "server_audit_table.tex").read_text()
    figure = (output / "server_audit_figure.tex").read_text()
    assert "post-primary reused-test discovery" in table
    assert "prospective role-reversal" in table
    assert "post-primary resampling" in table
    assert "descriptive" in table
    assert "$p$=$" not in table
    assert "$p<10^{-4}$" in table
    assert "secondary Holm" in table
    assert "Seed-paired adjacent effects" in figure
    assert "IMDb matched-$K$ composition and accuracy" in figure
    assert "IMDb primary gate" in figure
    assert "secondary: Routed-$K$" in figure
    assert figure.count("opacity=.28") == 90

    manifest = json.loads(
        (output / "server_audit_current_inputs.json").read_text())
    assert manifest["schema_version"] == "server_audit_published_manifest_v1"
    assert len(manifest["inputs"]["matrix_sha256"]["trec"]) == 40
    verified = generator.verify_published_assets(
        paths["trec"], paths["imdb"], paths["matched"], paths["agnews"],
        output, paths["trec_protocol"], paths["imdb_protocol"],
        paths["matched_protocol"], paths["ag_protocol"])
    assert verified["records"] == 150

    smoke = output / "server_audit_table_smoke.tex"
    smoke.write_text(
        r"\documentclass{article}" "\n"
        r"\usepackage{booktabs}" "\n"
        r"\usepackage{graphicx}" "\n"
        r"\begin{document}" "\n"
        r"\input{server_audit_table.tex}" "\n"
        r"\end{document}" "\n", encoding="utf-8")
    for source in (smoke.name, "server_audit_figure.tex"):
        for _ in range(2):
            compiled = subprocess.run(
                ["pdflatex", "-halt-on-error", "-interaction=batchmode", source],
                cwd=output, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=False)
            assert compiled.returncode == 0, compiled.stdout
        log = (output / f"{Path(source).stem}.log").read_text(
            encoding="utf-8", errors="replace")
        assert "Warning" not in log
        assert "Overfull " not in log
        assert "Underfull " not in log


def test_current_input_manifest_rejects_changed_record_and_build_wires_gate(
        tmp_path: Path) -> None:
    paths = _synthetic_inputs(tmp_path)
    output = tmp_path / "generated"
    _call(paths, output)
    table = output / "server_audit_table.tex"
    original_table = table.read_text(encoding="utf-8")
    table.write_text(original_table + "% stale edit\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale or missing server audit asset"):
        generator.verify_published_assets(
            paths["trec"], paths["imdb"], paths["matched"], paths["agnews"],
            output, paths["trec_protocol"], paths["imdb_protocol"],
            paths["matched_protocol"], paths["ag_protocol"])
    table.write_text(original_table, encoding="utf-8")

    changed = paths["agnews"] / "seed281_public_anchor_only.json"
    changed.write_text(changed.read_text(encoding="utf-8") + "\n",
                       encoding="utf-8")
    with pytest.raises(RuntimeError, match="server_audit_current_inputs.inputs"):
        generator.verify_published_assets(
            paths["trec"], paths["imdb"], paths["matched"], paths["agnews"],
            output, paths["trec_protocol"], paths["imdb_protocol"],
            paths["matched_protocol"], paths["ag_protocol"])

    build = (generator.ROOT / "paper/build_submission.sh").read_text()
    assert "generate_server_audit_assets.py --verify-only" in build
    assert "Refusing stale server_audit assets" in build
    assert "tests/test_generate_server_audit_assets.py" in build
