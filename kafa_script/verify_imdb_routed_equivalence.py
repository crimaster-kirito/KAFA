"""Verify functional equivalence of the original and matched-K routed arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_pair(original_path: Path, matched_path: Path) -> None:
    original = json.loads(original_path.read_text(encoding="utf-8"))
    matched = json.loads(matched_path.read_text(encoding="utf-8"))
    label = f"{original_path.name} vs {matched_path.name}"
    for key in ("seed", "evaluation_split", "evaluation_examples",
                "anchor_examples", "mixture_assignment_sha256", "final",
                "total_client_to_server_payload_bytes"):
        require(original[key] == matched[key], f"{label}: differs at {key}")
    require(len(original["round_records"]) == len(matched["round_records"]),
            f"{label}: round count")
    for round_index, (left_round, right_round) in enumerate(zip(
            original["round_records"], matched["round_records"])):
        for key in ("round", "teacher_weights", "mean_server_objective"):
            require(left_round[key] == right_round[key],
                    f"{label}: round {round_index} differs at {key}")
        require(len(left_round["client_records"]) ==
                len(right_round["client_records"]),
                f"{label}: round {round_index} client count")
        for client_index, (left, right) in enumerate(zip(
                left_round["client_records"], right_round["client_records"])):
            for key in ("client_index", "model", "base_target_ids_sha256",
                        "mixture_assignment_sha256", "selected_ids_sha256",
                        "mixture_kind_counts",
                        "selected_kind_counts_audit_only", "mean_local_loss",
                        "probability_payload",
                        "serialized_probability_payload", "adapter_bytes"):
                require(left[key] == right[key],
                        f"{label}: round {round_index} client {client_index} "
                        f"differs at {key}")


def verify_phase(phase: str, original_dir: Path,
                 matched_dir: Path) -> dict:
    seeds = [311, 312, 313] if phase == "development" else list(range(321, 331))
    for seed in seeds:
        verify_pair(
            original_dir / f"seed{seed}_severe_routed_distillation.json",
            matched_dir / f"seed{seed}_severe_routed_k.json")
    return {"phase": phase, "pairs": len(seeds), "seeds": seeds,
            "functional_equivalence": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["development", "official_evaluation"],
                        required=True)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--matched-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_phase(args.phase, args.original_dir,
                                  args.matched_dir), indent=2))


if __name__ == "__main__":
    main()
