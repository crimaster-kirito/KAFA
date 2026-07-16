import copy
import hashlib
import json
from pathlib import Path

import pytest

from kafa_script import verify_trec_server_protocol as verifier


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "kafa_script/trec_server_transfer_protocol_v1.json"


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _frozen_protocol():
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_frozen_inputs_pass_when_output_directories_do_not_exist(tmp_path):
    summary = verifier.verify(
        PROTOCOL, root=ROOT,
        development_output=tmp_path / "missing-development",
        official_output=tmp_path / "missing-official")

    assert summary["unique_records"] == 504
    assert summary["protocol_sha256"] == verifier.sha256_file(PROTOCOL)


@pytest.mark.parametrize("mutation, message", [
    (lambda protocol: protocol["task"].__setitem__(
        "task_spec_semantic_sha256", "0" * 64), "task semantic hash"),
    (lambda protocol: protocol["task"].__setitem__(
        "manifest_file_sha256", "0" * 64), "manifest raw-file hash"),
    (lambda protocol: protocol["task"].__setitem__(
        "target_train_sha256", "0" * 64), "target_train_sha256"),
    (lambda protocol: protocol["fixed_configuration"].__setitem__(
        "conflict_permutation", [2, 3, 4, 5, 0, 1]), "conflict permutation"),
])
def test_rejects_frozen_hash_or_task_semantic_drift(tmp_path, mutation, message):
    protocol = _frozen_protocol()
    mutation(protocol)
    path = tmp_path / "protocol.json"
    _write_json(path, protocol)

    with pytest.raises(verifier.VerificationError, match=message):
        verifier.verify(path, root=ROOT,
                        development_output=tmp_path / "dev",
                        official_output=tmp_path / "official")


@pytest.mark.parametrize("mutation, message", [
    (lambda protocol: protocol["official_evaluation"]["seeds"].__setitem__(
        0, protocol["development"]["seeds"][0]), "globally unique"),
    (lambda protocol: protocol["official_evaluation"].__setitem__(
        "comparators", list(reversed(verifier.EXPECTED_COMPARATORS))), "comparators"),
    (lambda protocol: protocol["fixed_configuration"].__setitem__(
        "rounds", 0), "rounds.*positive"),
])
def test_rejects_seed_comparator_or_positive_configuration_drift(
        tmp_path, mutation, message):
    protocol = _frozen_protocol()
    mutation(protocol)
    path = tmp_path / "protocol.json"
    _write_json(path, protocol)

    with pytest.raises(verifier.VerificationError, match=message):
        verifier.verify(path, root=ROOT,
                        development_output=tmp_path / "dev",
                        official_output=tmp_path / "official")


def test_rejects_manifest_overlap_even_with_updated_raw_hash(tmp_path):
    protocol = _frozen_protocol()
    manifest_path = ROOT / protocol["task"]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    replaced_label = manifest["splits"]["client_0"]["records"][0]["label"]
    overlapping = next(record for record in manifest["splits"]["anchor"]["records"]
                       if record["label"] == replaced_label)
    manifest["splits"]["client_0"]["records"][0] = copy.deepcopy(overlapping)
    ids = [record["source_id"]
           for record in manifest["splits"]["client_0"]["records"]]
    manifest["splits"]["client_0"]["source_ids_sha256"] = hashlib.sha256(
        "\n".join(ids).encode("ascii")).hexdigest()
    temp_manifest = tmp_path / "manifest.json"
    _write_json(temp_manifest, manifest)
    protocol["task"]["manifest"] = str(temp_manifest)
    protocol["task"]["manifest_file_sha256"] = verifier.sha256_file(temp_manifest)
    temp_protocol = tmp_path / "protocol.json"
    _write_json(temp_protocol, protocol)

    with pytest.raises(verifier.VerificationError, match="overlaps"):
        verifier.verify(temp_protocol, root=ROOT,
                        development_output=tmp_path / "dev",
                        official_output=tmp_path / "official")


def _official_record(protocol_hash, seed, comparator):
    return {"evidence_label": "benchmark_holdout",
            "evaluation_split": "official_test", "protocol_sha256": protocol_hash,
            "seed": seed, "comparator": comparator}


def test_output_evidence_boundary_and_complete_official_matrix(tmp_path):
    protocol = _frozen_protocol()
    protocol_hash = verifier.sha256_file(PROTOCOL)
    development = tmp_path / "development"
    official = tmp_path / "official"
    development.mkdir()
    official.mkdir()
    _write_json(development / "dev.json", {
        "evidence_label": "benchmark_dev", "evaluation_split": "development"})
    for seed in protocol["official_evaluation"]["seeds"]:
        for comparator in verifier.EXPECTED_COMPARATORS:
            _write_json(official / f"{seed}_{comparator}.json",
                        _official_record(protocol_hash, seed, comparator))

    verifier.verify(PROTOCOL, root=ROOT, development_output=development,
                    official_output=official)

    _write_json(development / "leak.json",
                _official_record(protocol_hash, 251, "public_anchor_only"))
    with pytest.raises(verifier.VerificationError,
                       match="development output contains official evidence"):
        verifier.verify(PROTOCOL, root=ROOT, development_output=development,
                        official_output=official)
    (development / "leak.json").unlink()

    missing = official / "255_routed_distillation.json"
    missing.unlink()
    with pytest.raises(verifier.VerificationError, match="exactly 10 records"):
        verifier.verify(PROTOCOL, root=ROOT, development_output=development,
                        official_output=official)


def test_official_matrix_rejects_protocol_hash_mismatch(tmp_path):
    protocol = _frozen_protocol()
    official = tmp_path / "official"
    official.mkdir()
    protocol_hash = verifier.sha256_file(PROTOCOL)
    for seed in protocol["official_evaluation"]["seeds"]:
        for comparator in verifier.EXPECTED_COMPARATORS:
            record = _official_record(protocol_hash, seed, comparator)
            if seed == 253 and comparator == "routed_distillation":
                record["protocol_sha256"] = "0" * 64
            _write_json(official / f"{seed}_{comparator}.json", record)

    with pytest.raises(verifier.VerificationError, match="protocol hash mismatch"):
        verifier.verify(PROTOCOL, root=ROOT,
                        development_output=tmp_path / "missing-dev",
                        official_output=official)
