"""Post-hoc selector-component controls for the IMDb matched-K audit.

This module contains the selection semantics only.  It intentionally does not
modify the frozen IMDb primary or matched-K runners.
"""

from __future__ import annotations

from collections import Counter
from collections import defaultdict

from kafa_script.diagnose_anchor_gradient_separation import (
    gradient_cosine, parameter_gradients, rank_percentiles)
from kafa_script import run_generic_classification_distillation as core
from kafa_script.run_imdb_matched_control_addendum import stable_selection_rank


METHODS = {
    "gradient_only_k": {"selection": "gradient_agreement_only_k"},
    "loss_only_k": {"selection": "training_loss_only_k"},
    "margin_only_k": {"selection": "semantic_margin_only_k"},
}


def score_records_for_component_selectors(
        model,
        tokenizer,
        mixture: list[dict],
        anchors: list[dict],
        target_spec,
        off_task_spec,
        device,
        max_length: int) -> list[dict]:
    """Compute the three frozen routed-score ingredients for target rows.

    The original full router scores every mixture row so it can decide whether
    to exclude off-task rows when the gate triggers.  These component controls
    instead share the matched-K addendum's target-eligible pool, so off-task rows
    are not scored or eligible for selection.
    """
    parameters = [parameter for parameter in model.parameters()
                  if parameter.requires_grad]
    grouped = defaultdict(list)
    for row in anchors:
        grouped[int(row[target_spec.label_field])].append(row)
    if set(grouped) != set(range(target_spec.num_classes)):
        raise ValueError("anchors do not cover every class")
    anchor_gradients = {}
    for label in range(target_spec.num_classes):
        batch = core.collate(tokenizer, [core.encode_target(
            tokenizer, row, target_spec, max_length) for row in grouped[label]])
        batch = {key: value.to(device) for key, value in batch.items()}
        model.zero_grad(set_to_none=True)
        model(**batch).loss.backward()
        anchor_gradients[label] = parameter_gradients(parameters)

    target_items = [item for item in mixture if item["kind"] != "off_task"]
    target_losses = core.candidate_loss_matrix(
        model, tokenizer,
        [{"text": item["text"], "label": item["training_label"]}
         for item in target_items],
        target_spec, device, max_length, 8).float().cpu()
    target_loss_by_id = {item["source_id"]: losses for item, losses in
                         zip(target_items, target_losses)}
    records = []
    for item in target_items:
        encoded = core.encode_mixture(tokenizer, item, target_spec,
                                      off_task_spec, max_length)
        batch = core.collate(tokenizer, [encoded])
        batch = {key: value.to(device) for key, value in batch.items()}
        model.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        claimed_label = int(item["training_label"])
        losses = target_loss_by_id[item["source_id"]]
        records.append({
            "source_id": item["source_id"],
            "claimed_label": claimed_label,
            "loss": float(loss.detach()),
            "anchor_cosine": gradient_cosine(
                parameter_gradients(parameters),
                anchor_gradients[claimed_label]),
            "semantic_margin": float(losses[claimed_label] - losses.min()),
        })
    return records


def _rank_values(records: list[dict], component: str) -> list[float]:
    if component == "gradient_only_k":
        return rank_percentiles([-float(record["anchor_cosine"])
                                 for record in records])
    if component == "loss_only_k":
        return rank_percentiles([float(record["loss"]) for record in records])
    if component == "margin_only_k":
        margins = []
        for record in records:
            margin = record.get("semantic_margin")
            if margin is None:
                raise ValueError("margin-only selector received an off-task row")
            margins.append(float(margin))
        return rank_percentiles(margins)
    raise ValueError(f"unknown selector component: {component}")


def select_component_k_from_records(
        mixture: list[dict],
        records: list[dict],
        method: str,
        k: int,
        seed: int,
        client_index: int) -> tuple[list[dict], dict]:
    """Select K target-eligible rows using exactly one routed-score component.

    ``records`` must contain one router score record per target-eligible mixture
    row.  The function preserves the matched-K addendum's 64/64 claimed-label
    quota, source-ID order in the returned mixture, and deterministic
    hash-based tie breaking.
    """
    if method not in METHODS:
        raise ValueError(f"unknown selector method: {method}")
    if k <= 0 or k % 2:
        raise ValueError("matched K must be a positive multiple of two")
    eligible = [item for item in mixture if item["kind"] != "off_task"]
    eligible_ids = [item["source_id"] for item in eligible]
    record_by_id = {record["source_id"]: record for record in records}
    if set(record_by_id) != set(eligible_ids):
        missing = sorted(set(eligible_ids) - set(record_by_id))
        extra = sorted(set(record_by_id) - set(eligible_ids))
        raise ValueError(
            f"score-record identity mismatch missing={missing} extra={extra}")
    ordered_records = [record_by_id[source_id] for source_id in eligible_ids]
    ranks = _rank_values(ordered_records, method)
    rank_by_id = {record["source_id"]: rank
                  for record, rank in zip(ordered_records, ranks)}
    quota = k // 2
    selected_ids: set[str] = set()
    for label in (0, 1):
        pool = [item for item in eligible
                if int(item["training_label"]) == label]
        if len(pool) < quota:
            raise ValueError(
                f"{method} fails closed: class {label} has {len(pool)} "
                f"eligible examples, fewer than quota {quota}")
        ranked = sorted(pool, key=lambda item: (
            rank_by_id[item["source_id"]],
            stable_selection_rank(method, seed, client_index,
                                  item["source_id"]),
            item["source_id"]))
        selected_ids.update(item["source_id"] for item in ranked[:quota])
    selected = [item for item in mixture if item["source_id"] in selected_ids]
    if len(selected) != k or len(selected_ids) != k:
        raise AssertionError("component selector did not select exactly K")
    selected_counts = Counter(int(item["training_label"]) for item in selected)
    if selected_counts != {0: quota, 1: quota}:
        raise AssertionError("component selector violated claimed-label quota")
    return selected, {
        "policy": METHODS[method]["selection"],
        "component": method.replace("_only_k", ""),
        "selected": len(selected),
        "eligible": len(eligible),
        "quota_per_claimed_label": quota,
        "off_task_selected": 0,
        "known_task_membership_used": True,
        "hidden_clean_conflict_metadata_used": False,
        "ranking_namespace": f"kafa-imdb-selector-component-v1|{method}",
    }
