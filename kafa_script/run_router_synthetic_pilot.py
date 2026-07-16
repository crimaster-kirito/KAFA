"""Controlled federated pilot for capability-conditioned gradient routing.

This is a real, deterministic computation on synthetic data.  Its results are
mechanism diagnostics, not evidence of LLM benchmark performance.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federatedscope.llm.knowledge_anchor.capability_router import (
    CapabilityGradientRouter,
)


METHODS = ("fedavg", "hard_filter", "kafa_router", "adaptive_router",
           "oracle")


def make_dataset(seed: int,
                 n_clients: int,
                 samples_per_client: int,
                 dimension: int,
                 conflict_ratio: float,
                 off_task_ratio: float):
    generator = torch.Generator().manual_seed(seed)
    target_w = F.normalize(torch.randn(dimension, generator=generator), dim=0)
    off_task_w = torch.randn(dimension, generator=generator)
    off_task_w -= torch.dot(off_task_w, target_w) * target_w
    off_task_w = F.normalize(off_task_w, dim=0)

    clients = []
    for client_id in range(n_clients):
        x = torch.randn(samples_per_client, dimension, generator=generator)
        clean_y = (x @ target_w > 0).float()
        kind = torch.zeros(samples_per_client, dtype=torch.long)
        order = torch.randperm(samples_per_client, generator=generator)
        n_conflict = int(samples_per_client * conflict_ratio)
        n_off_task = int(samples_per_client * off_task_ratio)
        conflict_idx = order[:n_conflict]
        off_task_idx = order[n_conflict:n_conflict + n_off_task]
        y = clean_y.clone()
        y[conflict_idx] = 1.0 - y[conflict_idx]
        y[off_task_idx] = (x[off_task_idx] @ off_task_w > 0).float()
        kind[conflict_idx] = 1
        kind[off_task_idx] = 2
        clients.append((x, y, kind, client_id))

    anchor_x = torch.randn(64, dimension, generator=generator)
    anchor_y = (anchor_x @ target_w > 0).float()
    test_x = torch.randn(2048, dimension, generator=generator)
    test_y = (test_x @ target_w > 0).float()
    return clients, (anchor_x, anchor_y), (test_x, test_y)


def per_example_gradients(weights: torch.Tensor, x: torch.Tensor,
                          y: torch.Tensor) -> torch.Tensor:
    residual = torch.sigmoid(x @ weights) - y
    return residual[:, None] * x


def binary_metrics(weights: torch.Tensor, x: torch.Tensor,
                   y: torch.Tensor) -> Tuple[float, float]:
    logits = x @ weights
    loss = F.binary_cross_entropy_with_logits(logits, y)
    accuracy = ((logits >= 0).float() == y).float().mean()
    return float(loss), float(accuracy)


def selection_metrics(scores: torch.Tensor, kind: torch.Tensor) -> Dict[str,
                                                                         float]:
    predicted_helpful = scores >= 0.5
    actually_helpful = kind == 0
    tp = (predicted_helpful & actually_helpful).sum().float()
    fp = (predicted_helpful & ~actually_helpful).sum().float()
    fn = (~predicted_helpful & actually_helpful).sum().float()
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    return {"routing_precision": float(precision), "routing_recall": float(recall)}


def grouped_anchor_gradients(weights: torch.Tensor, anchor,
                             n_groups: int) -> torch.Tensor:
    """Estimate stable capability directions from deterministic micro-batches."""
    gradients = per_example_gradients(weights, *anchor)
    if len(gradients) % n_groups:
        raise ValueError("anchor size must be divisible by anchor_groups")
    return gradients.reshape(n_groups, -1,
                             gradients.shape[-1]).mean(dim=1)


def train_one(method: str, clients, anchor, test, rounds: int, lr: float,
              anchor_groups: int, adaptive_protocol: Dict):
    dimension = clients[0][0].shape[1]
    weights = torch.zeros(dimension)
    router = CapabilityGradientRouter(temperature=0.12,
                                      margin=0.0,
                                      uncertainty_beta=0.5,
                                      min_weight=0.01,
                                      reject_conflicts=True)
    soft_router = CapabilityGradientRouter(temperature=0.12,
                                           margin=0.0,
                                           uncertainty_beta=0.5,
                                           min_weight=0.01,
                                           reject_conflicts=False)
    final_route_scores: List[torch.Tensor] = []
    final_kinds: List[torch.Tensor] = []

    for round_idx in range(rounds):
        anchor_gradients = grouped_anchor_gradients(weights, anchor,
                                                    anchor_groups)
        client_updates = []
        final_route_scores = []
        final_kinds = []
        for x, y, kind, _ in clients:
            gradients = per_example_gradients(weights, x, y)
            if method == "fedavg":
                update = gradients.mean(dim=0)
                scores = torch.ones(len(x))
            elif method == "hard_filter":
                anchor_direction = F.normalize(anchor_gradients.mean(0), dim=0)
                similarities = F.normalize(gradients, dim=1) @ anchor_direction
                selected = similarities >= 0
                update = gradients[selected].mean(0) if selected.any() else gradients.mean(0)
                scores = selected.float()
            elif method == "kafa_router":
                routed = router.route(gradients, anchor_gradients)
                update = routed.shared_gradient
                scores = routed.weights
            elif method == "adaptive_router":
                soft = soft_router.route(gradients, anchor_gradients)
                estimated_conflict = float(
                    (soft.mean_utility < 0).float().mean())
                if estimated_conflict <= adaptive_protocol["benign_threshold"]:
                    update = gradients.mean(dim=0)
                    scores = torch.ones(len(x))
                elif estimated_conflict <= adaptive_protocol["severe_threshold"]:
                    update = soft.shared_gradient
                    scores = soft.weights
                else:
                    safe = router.route(gradients, anchor_gradients)
                    update = safe.shared_gradient
                    scores = safe.weights
            elif method == "oracle":
                selected = kind == 0
                update = gradients[selected].mean(0)
                scores = selected.float()
            else:
                raise ValueError(method)
            client_updates.append(update)
            if round_idx == rounds - 1:
                final_route_scores.append(scores.detach())
                final_kinds.append(kind)
        weights = weights - lr * torch.stack(client_updates).mean(0)

    test_loss, test_accuracy = binary_metrics(weights, *test)
    anchor_loss, anchor_accuracy = binary_metrics(weights, *anchor)
    route = selection_metrics(torch.cat(final_route_scores),
                              torch.cat(final_kinds))
    return {
        "method": method,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "anchor_loss": anchor_loss,
        "anchor_accuracy": anchor_accuracy,
        **route,
    }


def run(args):
    rows = []
    protocol = json.loads(args.adaptive_protocol.read_text(encoding="utf-8"))
    for off_task_ratio in args.off_task_ratios:
        for conflict_ratio in args.conflict_ratios:
            if conflict_ratio + off_task_ratio >= 1:
                continue
            for seed in args.seeds:
                clients, anchor, test = make_dataset(
                    seed=seed,
                    n_clients=args.clients,
                    samples_per_client=args.samples_per_client,
                    dimension=args.dimension,
                    conflict_ratio=conflict_ratio,
                    off_task_ratio=off_task_ratio,
                )
                for method in METHODS:
                    result = train_one(method, clients, anchor, test,
                                       args.rounds, args.lr,
                                       args.anchor_groups, protocol)
                    result.update({
                        "seed": seed,
                        "conflict_ratio": conflict_ratio,
                        "off_task_ratio": off_task_ratio,
                    })
                    rows.append(result)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--conflict-ratios", nargs="+", type=float,
                        default=[0.0, 0.2, 0.4, 0.6])
    parser.add_argument("--off-task-ratios", nargs="+", type=float,
                        default=[0.2])
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--samples-per-client", type=int, default=128)
    parser.add_argument("--dimension", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.4)
    parser.add_argument("--anchor-groups", type=int, default=8)
    parser.add_argument("--adaptive-protocol", type=Path,
                        default=Path("kafa_script/router_adaptive_protocol.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("kafa_script/analysis_outputs/router_pilot_v4"))
    args = parser.parse_args()
    if any(r < 0 or r >= 1 for r in args.conflict_ratios + args.off_task_ratios):
        parser.error("ratios must be in [0, 1)")

    rows = run(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "status": "real synthetic-data mechanism pilot",
        "paper_result": False,
        "configuration": {
            **vars(args),
            "output_dir": str(args.output_dir),
            "adaptive_protocol": str(args.adaptive_protocol),
        },
        "n_runs": len(rows),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), **metadata}, indent=2))


if __name__ == "__main__":
    main()
