import json
from pathlib import Path

from kafa_script import imdb_selector_component_ablation as selectors
from kafa_script import summarize_imdb_selector_component_ablation as summary
from kafa_script.verify_imdb_selector_component_ablation import SEEDS


def _component_record(seed: int, method: str, accuracy: float) -> dict:
    client = {
        "selected_kind_counts_audit_only": {"clean": 90, "conflict": 38},
    }
    return {
        "seed": seed,
        "method": method,
        "round_records": [
            {"client_records": [client, client, client]},
            {"client_records": [client, client, client]},
        ],
        "final": {
            "accuracy": accuracy,
            "macro_f1": accuracy - 0.001,
            "canonical_label_nll": 0.2,
            "ece": 0.01,
        },
        "wall_seconds": 10.0 + (seed - 321),
        "peak_allocated_gpu_bytes": 2**30,
    }


def _write_records(root: Path, methods: list[str], base_accuracy: float,
                   seed_slope: float = 0.0001) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        for offset, method in enumerate(methods):
            accuracy = base_accuracy + seed_slope * (seed - 321) - 0.001 * offset
            (root / f"seed{seed}_{method}.json").write_text(
                json.dumps(_component_record(seed, method, accuracy)),
                encoding="utf-8")


def test_selector_component_summary_joins_reference_and_component_records(
        tmp_path: Path, monkeypatch) -> None:
    component_dir = tmp_path / "component"
    matched_dir = tmp_path / "matched"
    _write_records(component_dir, sorted(selectors.METHODS), 0.90)
    _write_records(matched_dir, ["severe_routed_k"], 0.91, seed_slope=0.0002)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "official_evaluation": {
            "seeds": SEEDS,
            "comparators": sorted(selectors.METHODS),
        },
        "evidence_boundaries": ["post-hoc reused-test diagnostic"],
    }), encoding="utf-8")
    matched_protocol = tmp_path / "matched_protocol.json"
    matched_protocol.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(summary, "verify_component_matrix",
                        lambda input_dir, protocol_path: {"records": 30})
    monkeypatch.setattr(summary, "verify_matched_matrix",
                        lambda phase, input_dir, protocol_path: {
                            "protocol_sha256": "matched-protocol-sha"})

    output = tmp_path / "summary.md"
    text = summary.summarize(
        component_dir, matched_dir, output, protocol, matched_protocol)

    assert output.read_text(encoding="utf-8") == text
    assert "30** new component records plus **10** existing" in text
    assert "Full routed-K" in text
    assert "full_routed_k_minus_gradient_only_k" in text
    assert "Post-hoc selector-component Holm family" in text
    assert "post-hoc reused-test diagnostic" in text
