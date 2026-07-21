"""Verify conditional implementation equivalence of the two routed arms.

The original arm applies the frozen routing gate: when the gate does not
trigger it retains all 256 examples. The matched-K addendum deliberately
forces the same router to select K=128. Exact output equivalence is therefore
required only along a client trajectory before an original-gate rejection.
At the first rejection, this verifier proves that the gate decision is the
only functional branch difference; after that client's state has diverged,
only immutable inputs remain comparable. This is an instrumentation check,
not an equivalence claim about the two statistical estimands.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


TOP_LEVEL_IMMUTABLE_KEYS = (
    "seed", "evaluation_split", "evaluation_examples", "anchor_examples",
    "mixture_assignment_sha256",
)
CLIENT_IMMUTABLE_KEYS = (
    "client_index", "model", "base_target_ids_sha256",
    "mixture_assignment_sha256", "mixture_kind_counts",
)
ROUTER_SCORE_KEYS = (
    "gate_ratio", "client_q25", "reference_q25",
)
EXACT_FUNCTIONAL_KEYS = (
    "selected_ids_sha256", "selected_kind_counts_audit_only",
    "mean_local_loss", "probability_payload",
    "serialized_probability_payload", "adapter_bytes",
)


def verify_pair(original_path: Path, matched_path: Path) -> dict:
    """Verify one pair under the frozen conditional/forced gate contract."""
    original = json.loads(original_path.read_text(encoding="utf-8"))
    matched = json.loads(matched_path.read_text(encoding="utf-8"))
    label = f"{original_path.name} vs {matched_path.name}"
    for key in TOP_LEVEL_IMMUTABLE_KEYS:
        require(original[key] == matched[key], f"{label}: differs at {key}")
    require(len(original["round_records"]) == len(matched["round_records"]),
            f"{label}: round count")

    diverged_clients: set[int] = set()
    first_divergences = []
    exact_pre_fork_triggered_client_rounds = 0
    post_local_fork_not_equivalence_eligible = 0
    any_divergence_before_round = False

    for round_index, (left_round, right_round) in enumerate(zip(
            original["round_records"], matched["round_records"])):
        require(left_round["round"] == right_round["round"],
                f"{label}: round {round_index} differs at round")
        require(left_round["teacher_weights"] == right_round["teacher_weights"],
                f"{label}: round {round_index} differs at teacher_weights")
        require(len(left_round["client_records"]) ==
                len(right_round["client_records"]),
                f"{label}: round {round_index} client count")

        round_introduced_divergence = False
        for client_index, (left, right) in enumerate(zip(
                left_round["client_records"], right_round["client_records"])):
            prefix = f"{label}: round {round_index} client {client_index}"
            for key in CLIENT_IMMUTABLE_KEYS:
                require(left[key] == right[key], f"{prefix} differs at {key}")
            left_selection = left["selection"]
            right_selection = right["selection"]
            require(left_selection["policy"] == "relative_balanced_routing",
                    f"{prefix}: unexpected original policy")
            require(right_selection["policy"] ==
                    "forced_relative_balanced_routing_k",
                    f"{prefix}: unexpected matched policy")
            left_ratio = float(left_selection["gate_ratio"])
            right_ratio = float(right_selection["gate_ratio"])
            require(math.isfinite(left_ratio) and math.isfinite(right_ratio),
                    f"{prefix}: non-finite gate ratio")
            require(left_selection.get("gate_threshold") == 0.8 and
                    left_selection.get("gate_triggered") is
                    (left_ratio < 0.8),
                    f"{prefix}: invalid original gate decision")
            require(right_selection.get("gate_threshold") == 1e30 and
                    right_selection.get("gate_triggered") is True and
                    right_ratio < 1e30,
                    f"{prefix}: invalid forced gate decision")
            require(right_selection.get("forced_selection") is True and
                    right_selection.get("selected") == 128 and
                    right_selection.get("quota_per_claimed_label") == 64 and
                    right_selection.get("off_task_selected") == 0 and
                    right_selection.get("known_task_membership_used") is True and
                    right_selection.get(
                        "hidden_clean_conflict_metadata_used") is False and
                    sum(right["selected_kind_counts_audit_only"].values()) == 128 and
                    right["selected_kind_counts_audit_only"].get(
                        "off_task", 0) == 0 and
                    right.get("selected_claimed_label_counts") ==
                        {"0": 64, "1": 64},
                    f"{prefix}: invalid forced-K branch")

            if left_selection["gate_triggered"]:
                require(left_selection.get("selected") == 128 and
                        left_selection.get("quota_per_claimed_label") == 64 and
                        left_selection.get("off_task_selected") == 0 and
                        sum(left["selected_kind_counts_audit_only"].values()) ==
                        128 and
                        left["selected_kind_counts_audit_only"].get(
                            "off_task", 0) == 0,
                        f"{prefix}: invalid triggered original branch")
            else:
                require(left_selection.get("selected") == 256 and
                        sum(left["selected_kind_counts_audit_only"].values()) ==
                        256 and
                        left["selected_kind_counts_audit_only"] ==
                        left["mixture_kind_counts"],
                        f"{prefix}: invalid untriggered original branch")

            identity = int(left["client_index"])
            if identity in diverged_clients:
                post_local_fork_not_equivalence_eligible += 1
                continue

            # Before state divergence both runners score the same mixture with
            # the same model state, so the router values must match exactly.
            for key in ROUTER_SCORE_KEYS:
                require(left_selection.get(key) == right_selection.get(key),
                        f"{prefix}: pre-branch router differs at {key}")

            if left_selection.get("gate_triggered") is True:
                require(left_selection.get("score_record_order_sha256") ==
                        right_selection.get("score_record_order_sha256"),
                        f"{prefix}: triggered router order differs")
                for key in EXACT_FUNCTIONAL_KEYS:
                    require(left[key] == right[key],
                            f"{prefix}: triggered branch differs at {key}")
                exact_pre_fork_triggered_client_rounds += 1
            else:
                # The sole authorized first divergence: original keeps the
                # whole mixture, while the matched addendum forces K.
                require(left["selected_ids_sha256"] != right["selected_ids_sha256"],
                        f"{prefix}: gate branches did not actually diverge")
                diverged_clients.add(identity)
                first_divergences.append({
                    "round": int(left_round["round"]),
                    "client_index": identity,
                })
                round_introduced_divergence = True

        if not any_divergence_before_round and not round_introduced_divergence:
            require(left_round["mean_server_objective"] ==
                    right_round["mean_server_objective"],
                    f"{label}: round {round_index} pre-divergence server differs")
        any_divergence_before_round |= round_introduced_divergence

    if not diverged_clients:
        require(original["total_client_to_server_payload_bytes"] ==
                matched["total_client_to_server_payload_bytes"],
                f"{label}: no gate divergence but payload total differs")
        require(original["final"] == matched["final"],
                f"{label}: no gate divergence but final differs")

    return {
        "seed": int(original["seed"]),
        "conditional_implementation_equivalence": True,
        "fully_identical_trajectory": not bool(diverged_clients),
        "total_client_rounds": sum(
            len(item["client_records"]) for item in original["round_records"]),
        "first_forced_gate_divergences": first_divergences,
        "exact_pre_fork_triggered_client_rounds":
            exact_pre_fork_triggered_client_rounds,
        "first_authorized_gate_divergence_client_rounds":
            len(first_divergences),
        "post_local_fork_not_equivalence_eligible_client_rounds":
            post_local_fork_not_equivalence_eligible,
    }


def verify_phase(phase: str, original_dir: Path,
                 matched_dir: Path) -> dict:
    seeds = [311, 312, 313] if phase == "development" else list(range(321, 331))
    receipts = []
    for seed in seeds:
        receipts.append(verify_pair(
            original_dir / f"seed{seed}_severe_routed_distillation.json",
            matched_dir / f"seed{seed}_severe_routed_k.json"))
    total_client_rounds = sum(item["total_client_rounds"] for item in receipts)
    exact = sum(item["exact_pre_fork_triggered_client_rounds"]
                for item in receipts)
    first = sum(item["first_authorized_gate_divergence_client_rounds"]
                for item in receipts)
    post = sum(item[
        "post_local_fork_not_equivalence_eligible_client_rounds"]
               for item in receipts)
    require(exact + first + post == total_client_rounds,
            "conditional-equivalence receipt does not partition client-rounds")
    return {
        "schema_version": "imdb_routed_conditional_equivalence_v2",
        "phase": phase,
        "pairs": len(seeds),
        "seeds": seeds,
        "conditional_implementation_equivalence": True,
        "total_client_rounds": total_client_rounds,
        "fully_identical_trajectory_pairs": sum(
            item["fully_identical_trajectory"] for item in receipts),
        "pairs_with_forced_gate_divergence": sum(
            not item["fully_identical_trajectory"] for item in receipts),
        "divergent_seeds": [item["seed"] for item in receipts
                            if not item["fully_identical_trajectory"]],
        "exact_pre_fork_triggered_client_rounds": exact,
        "first_authorized_gate_divergence_client_rounds": first,
        "post_local_fork_not_equivalence_eligible_client_rounds": post,
        "pair_receipts": receipts,
        "claim_boundary": (
            "Implementation equivalence is conditional on the original gate; "
            "this receipt is not statistical equivalence of the two arms."
        ),
    }


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
