#!/usr/bin/env python3
"""Validate and summarize the frozen TREC server-transfer matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.classification_task import canonical_sha256, load_task_spec  # noqa: E402


PROTOCOL_PATH = ROOT / "kafa_script/trec_server_transfer_protocol_v1.json"
EXPECTED_PROTOCOL_SHA256 = (
    "ee43c7d4ee8d6ac4a3fc3caecc78e4ad3625157f223e27b37db49efe65dff099"
)
METHODS = ("public_anchor_only", "routed_distillation")
METRICS = ("accuracy", "macro_f1", "canonical_label_nll", "ece")
DISPLAY = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "canonical_label_nll": "Class NLL",
    "ece": "ECE",
}


class SummaryValidationError(ValueError):
    """Raised when raw evidence differs from the frozen evidence contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SummaryValidationError(message)


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SummaryValidationError(f"cannot read {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class PairedStatistics:
    mean: float
    lower: float
    upper: float
    pvalue: float | None


def paired_statistics(differences: list[float]) -> PairedStatistics:
    """Two-sided paired-t summary, expressed directly on paired differences."""
    require(len(differences) >= 2, "paired inference needs at least two seeds")
    require(all(math.isfinite(value) for value in differences),
            "paired differences must be finite")
    mean = statistics.mean(differences)
    standard_deviation = statistics.stdev(differences)
    half = (float(stats.t.ppf(.975, len(differences) - 1)) *
            standard_deviation / math.sqrt(len(differences)))
    if math.isclose(standard_deviation, 0.0, rel_tol=0.0, abs_tol=1e-15):
        # With no estimated seed variance, a Student-t statistic is undefined.
        pvalue = 1.0 if math.isclose(mean, 0.0, abs_tol=1e-15) else None
    else:
        pvalue = float(stats.ttest_1samp(differences, popmean=0.0,
                                        alternative="two-sided").pvalue)
    return PairedStatistics(mean, mean - half, mean + half, pvalue)


def _phase_contract(protocol: dict, phase: str) -> tuple[list[int], str, bool, int]:
    if phase == "development":
        return (protocol["development"]["seeds"], "benchmark_development",
                False, protocol["fixed_configuration"]["development_examples"])
    if phase == "official":
        return (protocol["official_evaluation"]["seeds"],
                "frozen_official_test", True,
                protocol["fixed_configuration"]["official_test_examples"])
    raise SummaryValidationError(f"unknown phase: {phase}")


def validate_frozen_artifacts(root: Path, protocol_path: Path) -> tuple[dict, dict]:
    protocol_hash = sha256_file(protocol_path)
    require(protocol_hash == EXPECTED_PROTOCOL_SHA256,
            "protocol file differs from the frozen summarizer digest")
    protocol = read_json(protocol_path)
    implementation = protocol["implementation"]
    runner = resolve(root, implementation["runner"])
    core = resolve(root, implementation["classification_task_core"])
    require(sha256_file(runner) == implementation["runner_file_sha256"],
            "current runner differs from frozen protocol")
    require(sha256_file(core) == implementation["classification_task_file_sha256"],
            "current classification core differs from frozen protocol")
    task = protocol["task"]
    manifest_path = resolve(root, task["manifest"])
    require(sha256_file(manifest_path) == task["manifest_file_sha256"],
            "current manifest differs from frozen protocol")
    spec = load_task_spec(resolve(root, task["task_spec"]))
    require(spec.sha256 == task["task_spec_semantic_sha256"],
            "current task spec differs semantically from protocol")
    manifest = read_json(manifest_path)
    require(manifest.get("task_spec_sha256") == spec.sha256,
            "manifest task hash differs from task spec")
    return protocol, {
        "protocol": protocol_hash,
        "runner": implementation["runner_file_sha256"],
        "core": implementation["classification_task_file_sha256"],
        "manifest": task["manifest_file_sha256"],
        "task_spec_file": sha256_file(resolve(root, task["task_spec"])),
    }


def _finite_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(value))


def _validate_record(path: Path, record: dict, *, phase: str, seed: int,
                     method: str, evidence_label: str, paper_result: bool,
                     evaluation_examples: int, protocol: dict,
                     hashes: dict) -> None:
    prefix = f"{path}: "
    require(record.get("schema_version") == "2.0", prefix + "schema is not v2")
    require(record.get("task_id") == protocol["task"]["task_id"],
            prefix + "wrong task_id")
    require(record.get("seed") == seed and record.get("method") == method,
            prefix + "seed/method mismatch")
    require(record.get("evidence_label") == evidence_label,
            prefix + "wrong evidence label")
    expected_split = "development" if phase == "development" else "official_test"
    require(record.get("evaluation_split") == expected_split,
            prefix + "wrong evaluation split")
    require(record.get("paper_result") is paper_result,
            prefix + "wrong paper-result boundary")
    require(record.get("evaluation_examples") == evaluation_examples,
            prefix + "wrong evaluation count")
    require(record.get("anchor_examples") == 72, prefix + "wrong anchor count")
    require(record.get("protocol_sha256") == hashes["protocol"],
            prefix + "protocol hash mismatch")
    require(record.get("runner_file_sha256") == hashes["runner"],
            prefix + "runner hash mismatch")
    require(record.get("classification_task_file_sha256") == hashes["core"],
            prefix + "classification core hash mismatch")
    require(record.get("manifest_sha256") == hashes["manifest"],
            prefix + "manifest hash mismatch")
    require(record.get("task_spec_file_sha256") == hashes["task_spec_file"],
            prefix + "task-spec file hash mismatch")
    require(record.get("task_spec_semantic_sha256") ==
            protocol["task"]["task_spec_semantic_sha256"],
            prefix + "task semantic hash mismatch")
    require(record.get("off_task_spec_semantic_sha256") ==
            protocol["task"]["off_task_spec_semantic_sha256"],
            prefix + "off-task semantic hash mismatch")

    full_args = record.get("full_args")
    require(isinstance(full_args, dict), prefix + "full_args missing")
    require(record.get("full_args_sha256") == canonical_sha256(full_args),
            prefix + "full_args hash mismatch")
    require(full_args.get("seed") == seed and full_args.get("method") == method,
            prefix + "full_args seed/method mismatch")
    require(full_args.get("evaluation_split") == expected_split,
            prefix + "full_args split mismatch")

    datasets = record.get("datasets", {})
    task = protocol["task"]
    require(datasets.get("target_train_sha256") == task["target_train_sha256"],
            prefix + "target-train hash mismatch")
    require(datasets.get("off_task_train_sha256") == task["off_task_train_sha256"],
            prefix + "off-task hash mismatch")
    require(datasets.get("target_official_test_sha256") ==
            task["target_official_test_sha256"],
            prefix + "official-test hash mismatch")
    expected_hash_source = ("frozen_protocol_not_reopened_in_development"
                            if phase == "development" else "verified_raw_bytes")
    require(datasets.get("target_official_test_hash_source") == expected_hash_source,
            prefix + "official-test access/hash source mismatch")

    require(isinstance(record.get("started_utc"), str) and record["started_utc"],
            prefix + "started_utc missing")
    git = record.get("git")
    require(isinstance(git, dict) and set(git) >= {"commit", "dirty"},
            prefix + "git provenance missing")
    require(isinstance(record.get("model_artifacts"), dict),
            prefix + "model hashes missing")
    split_hashes = record.get("split_source_ids_sha256")
    require(isinstance(split_hashes, dict) and
            set(split_hashes) == {"anchor", "evaluation", "client_0",
                                  "client_1", "client_2"},
            prefix + "split hashes incomplete")

    final = record.get("final")
    require(isinstance(final, dict), prefix + "final metrics missing")
    for metric in METRICS:
        require(_finite_number(final.get(metric)), prefix + f"non-finite {metric}")
    require(0.0 <= final["accuracy"] <= 1.0 and
            0.0 <= final["macro_f1"] <= 1.0 and
            final["canonical_label_nll"] >= 0.0 and 0.0 <= final["ece"] <= 1.0,
            prefix + "metric outside valid range")

    rounds = record.get("round_records")
    require(isinstance(rounds, list) and len(rounds) == 2,
            prefix + "expected two round records")
    calculated_wire = 0
    for round_index, round_record in enumerate(rounds, start=1):
        require(round_record.get("round") == round_index,
                prefix + "round index mismatch")
        clients = round_record.get("client_records")
        expected_clients = 0 if method == "public_anchor_only" else 3
        require(isinstance(clients, list) and len(clients) == expected_clients,
                prefix + "wrong client count in round")
        for client in clients:
            routing = client.get("routing")
            require(isinstance(routing, dict) and
                    isinstance(routing.get("gate_triggered"), bool),
                    prefix + "routing gate audit missing")
            canonical_payload = client.get("probability_payload", {})
            require(canonical_payload.get("shape") == [72, 6],
                    prefix + "canonical probability shape mismatch")
            require(canonical_payload.get("max_row_sum_error", math.inf) <= 1e-6,
                    prefix + "probability simplex audit failed")
            wire = client.get("serialized_probability_payload", {})
            require(wire.get("encoding") == "torch-save-tensor" and
                    isinstance(wire.get("payload_bytes"), int) and
                    wire["payload_bytes"] > 0 and
                    isinstance(wire.get("payload_sha256"), str) and
                    len(wire["payload_sha256"]) == 64,
                    prefix + "wire payload audit missing")
            calculated_wire += wire["payload_bytes"]
    require(record.get("total_client_to_server_payload_bytes") == calculated_wire,
            prefix + "total wire bytes mismatch")
    require(_finite_number(record.get("wall_seconds")) and
            record["wall_seconds"] > 0, prefix + "invalid wall time")


def load_validated_matrix(input_dir: Path, phase: str, *, root: Path = ROOT,
                          protocol_path: Path = PROTOCOL_PATH) \
        -> tuple[dict[tuple[int, str], tuple[Path, dict]], dict, dict]:
    protocol, hashes = validate_frozen_artifacts(root, protocol_path)
    seeds, evidence_label, paper_result, examples = _phase_contract(protocol, phase)
    expected = {(seed, method) for seed in seeds for method in METHODS}
    require(input_dir.is_dir(), f"input directory is missing: {input_dir}")
    paths = sorted(input_dir.glob("*.json"))
    require(len(paths) == len(expected),
            f"expected {len(expected)} raw JSON files, found {len(paths)}")
    matrix = {}
    for path in paths:
        record = read_json(path)
        key = (record.get("seed"), record.get("method"))
        require(key in expected, f"unexpected seed/method in {path}: {key}")
        require(key not in matrix, f"duplicate seed/method record: {key}")
        _validate_record(path, record, phase=phase, seed=key[0], method=key[1],
                         evidence_label=evidence_label,
                         paper_result=paper_result,
                         evaluation_examples=examples, protocol=protocol,
                         hashes=hashes)
        matrix[key] = (path, record)
    require(set(matrix) == expected, "raw matrix is incomplete")

    reference_args = None
    reference_models = None
    reference_splits = None
    for _, record in matrix.values():
        comparable_args = {key: value for key, value in record["full_args"].items()
                           if key not in {"seed", "method", "output"}}
        if reference_args is None:
            reference_args = comparable_args
            reference_models = record["model_artifacts"]
            reference_splits = record["split_source_ids_sha256"]
        require(comparable_args == reference_args,
                "full_args differ beyond seed/method/output")
        require(record["model_artifacts"] == reference_models,
                "model artifact hashes differ across matrix")
        for key in ("anchor", "client_0", "client_1", "client_2"):
            require(record["split_source_ids_sha256"][key] ==
                    reference_splits[key], f"{key} split differs across matrix")
        require(record["split_source_ids_sha256"]["evaluation"] ==
                reference_splits["evaluation"],
                "evaluation split differs across matrix")
    return matrix, protocol, hashes


def summarize_matrix(matrix: dict, protocol: dict, phase: str) -> str:
    seeds, _, paper_result, _ = _phase_contract(protocol, phase)
    deltas = {metric: [
        matrix[(seed, "routed_distillation")][1]["final"][metric] -
        matrix[(seed, "public_anchor_only")][1]["final"][metric]
        for seed in seeds] for metric in METRICS}
    summaries = {metric: paired_statistics(values)
                 for metric, values in deltas.items()}
    status = ("official frozen-test paper evidence" if paper_result else
              "development sanity evidence; non-paper result")
    lines = [
        "# TREC server-transfer result", "",
        f"**Status: {status}.**", "",
        "All confidence intervals and paired t tests below describe variation "
        "over the frozen training seeds only; they do not quantify test-sample "
        "or task-population uncertainty.", "", "## Per-seed results", "",
        "| Seed | Anchor acc. | Routed acc. | Acc. delta | Macro-F1 delta | "
        "Class-NLL delta | ECE delta | Gates triggered | Wire bytes | "
        "Anchor / routed wall (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in seeds:
        anchor = matrix[(seed, "public_anchor_only")][1]
        routed = matrix[(seed, "routed_distillation")][1]
        gates = sum(client["routing"]["gate_triggered"]
                    for round_record in routed["round_records"]
                    for client in round_record["client_records"])
        lines.append(
            f"| {seed} | {100*anchor['final']['accuracy']:.2f}% | "
            f"{100*routed['final']['accuracy']:.2f}% | "
            f"{100*(routed['final']['accuracy']-anchor['final']['accuracy']):+.2f} pp | "
            f"{100*(routed['final']['macro_f1']-anchor['final']['macro_f1']):+.2f} pp | "
            f"{routed['final']['canonical_label_nll']-anchor['final']['canonical_label_nll']:+.4f} | "
            f"{routed['final']['ece']-anchor['final']['ece']:+.4f} | "
            f"{gates}/6 | {routed['total_client_to_server_payload_bytes']} | "
            f"{anchor['wall_seconds']:.1f} / {routed['wall_seconds']:.1f} |")
    lines += ["", "## Paired routed-minus-anchor inference", "",
              "| Metric | Mean delta | Descriptive 95% t-CI | Two-sided paired t p |",
              "|---|---:|---:|---:|"]
    for metric in METRICS:
        result = summaries[metric]
        scale = 100.0 if metric in {"accuracy", "macro_f1"} else 1.0
        unit = " pp" if scale == 100.0 else ""
        pvalue = "NA (zero seed variance)" if result.pvalue is None \
            else f"{result.pvalue:.4g}"
        lines.append(
            f"| {DISPLAY[metric]} | {scale*result.mean:+.4f}{unit} | "
            f"[{scale*result.lower:+.4f}, {scale*result.upper:+.4f}]{unit} | "
            f"{pvalue} |")
    anchor_walls = [matrix[(seed, "public_anchor_only")][1]["wall_seconds"]
                    for seed in seeds]
    routed_walls = [matrix[(seed, "routed_distillation")][1]["wall_seconds"]
                    for seed in seeds]
    total_gates = sum(
        client["routing"]["gate_triggered"] for seed in seeds
        for round_record in matrix[(seed, "routed_distillation")][1]["round_records"]
        for client in round_record["client_records"])
    wire = [matrix[(seed, "routed_distillation")][1]
            ["total_client_to_server_payload_bytes"] for seed in seeds]
    lines += ["", "## Routing and cost audit", "",
              f"- Routing gate triggers: {total_gates}/{len(seeds)*6} client-rounds.",
              f"- Routed wire payload: mean {statistics.mean(wire):.0f} bytes "
              f"(range {min(wire)}--{max(wire)}). Anchor-only sends 0 bytes.",
              f"- Mean wall time: anchor {statistics.mean(anchor_walls):.2f} s; "
              f"routed {statistics.mean(routed_walls):.2f} s; ratio "
              f"{statistics.mean(routed_walls)/statistics.mean(anchor_walls):.2f}x.",
              "", "## Raw records", ""]
    for seed in seeds:
        for method in METHODS:
            path = matrix[(seed, method)][0]
            lines.append(f"- `{path}` (SHA-256 `{sha256_file(path)}`)")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
                        choices=["development", "official"])
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix, protocol, _ = load_validated_matrix(args.input_dir, args.phase)
    report = summarize_matrix(matrix, protocol, args.phase)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
