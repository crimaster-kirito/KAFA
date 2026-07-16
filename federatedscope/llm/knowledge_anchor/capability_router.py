"""Capability-conditioned routing for heterogeneous federated updates.

The router operates on gradient sketches rather than model parameters, so its
local decision rule is architecture agnostic.  It does not itself provide a
privacy guarantee: sketches and diagnostics must remain local unless protected
by an explicit privacy mechanism.
"""

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RoutingResult:
    """Output of capability-conditioned sample routing."""

    weights: torch.Tensor
    mean_utility: torch.Tensor
    utility_std: torch.Tensor
    lower_confidence_utility: torch.Tensor
    shared_gradient: torch.Tensor
    diagnostics: Dict[str, float]


class CapabilityGradientRouter:
    """Route local examples using agreement with capability-anchor gradients.

    Let ``g_i`` be a local example's gradient sketch and ``a_j`` the gradient
    sketch of anchor example ``j``.  The first-order change in anchor loss after
    an SGD step on ``g_i`` is proportional to ``-a_j^T g_i``.  Positive cosine
    agreement is therefore evidence that the local step helps the anchor task.

    To avoid trusting a high mean caused by a few anchors, routing uses a lower
    confidence score ``mean(cos) - beta * std(cos)``. A safety gate keeps
    confidently conflicting samples out of the shared update; a smooth sigmoid
    ranks the remaining samples. The returned shared gradient is additionally
    guarded against a negative dot product with the mean anchor gradient.
    """

    def __init__(self,
                 temperature: float = 0.1,
                 margin: float = 0.0,
                 uncertainty_beta: float = 0.5,
                 min_weight: float = 0.0,
                 reject_conflicts: bool = True,
                 eps: float = 1e-12):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if uncertainty_beta < 0:
            raise ValueError("uncertainty_beta must be non-negative")
        if not 0 <= min_weight < 1:
            raise ValueError("min_weight must be in [0, 1)")
        self.temperature = temperature
        self.margin = margin
        self.uncertainty_beta = uncertainty_beta
        self.min_weight = min_weight
        self.reject_conflicts = reject_conflicts
        self.eps = eps

    def route(self, sample_gradients: torch.Tensor,
              anchor_gradients: torch.Tensor) -> RoutingResult:
        self._validate(sample_gradients, anchor_gradients)

        sample_unit = F.normalize(sample_gradients, dim=-1, eps=self.eps)
        anchor_unit = F.normalize(anchor_gradients, dim=-1, eps=self.eps)
        utility = sample_unit @ anchor_unit.t()
        mean_utility = utility.mean(dim=1)
        # Population std is defined for a single anchor as zero.
        utility_std = utility.std(dim=1, unbiased=False)
        lower_confidence = (mean_utility -
                            self.uncertainty_beta * utility_std)
        weights = torch.sigmoid(
            (lower_confidence - self.margin) / self.temperature)
        if self.min_weight:
            weights = self.min_weight + (1.0 - self.min_weight) * weights

        # A smooth score is useful among plausible samples, but assigning even
        # a small shared weight to a confidently harmful sample accumulates
        # damage over many local steps.  The lower-confidence sign supplies a
        # conservative safety gate; private/local training may still use the
        # rejected sample outside this shared-gradient path.
        if self.reject_conflicts:
            safe = lower_confidence >= self.margin
            weights = weights * safe
            if not safe.any():
                weights = torch.zeros_like(weights)
                weights[lower_confidence.argmax()] = 1.0

        denominator = weights.sum().clamp_min(self.eps)
        shared_gradient = (weights[:, None] * sample_gradients).sum(
            dim=0) / denominator
        anchor_mean = anchor_gradients.mean(dim=0)
        shared_gradient = self._remove_anchor_conflict(shared_gradient,
                                                       anchor_mean)

        diagnostics = {
            "mean_weight": float(weights.mean().detach().cpu()),
            "accepted_fraction": float((weights >= 0.5).float().mean().detach().cpu()),
            "conflicting_fraction": float((mean_utility < 0).float().mean().detach().cpu()),
            "mean_utility": float(mean_utility.mean().detach().cpu()),
            "mean_uncertainty": float(utility_std.mean().detach().cpu()),
        }
        return RoutingResult(
            weights=weights,
            mean_utility=mean_utility,
            utility_std=utility_std,
            lower_confidence_utility=lower_confidence,
            shared_gradient=shared_gradient,
            diagnostics=diagnostics,
        )

    def _remove_anchor_conflict(self, gradient: torch.Tensor,
                                anchor_gradient: torch.Tensor) -> torch.Tensor:
        """Project away only the component opposing the anchor gradient."""
        dot = torch.dot(gradient, anchor_gradient)
        if dot >= 0:
            return gradient
        scale = dot / anchor_gradient.square().sum().clamp_min(self.eps)
        return gradient - scale * anchor_gradient

    @staticmethod
    def _validate(sample_gradients: torch.Tensor,
                  anchor_gradients: torch.Tensor) -> None:
        if sample_gradients.ndim != 2 or anchor_gradients.ndim != 2:
            raise ValueError("sample_gradients and anchor_gradients must be 2-D")
        if sample_gradients.shape[0] == 0 or anchor_gradients.shape[0] == 0:
            raise ValueError("gradient collections must be non-empty")
        if sample_gradients.shape[1] != anchor_gradients.shape[1]:
            raise ValueError("sample and anchor sketches must share a dimension")
        if not torch.isfinite(sample_gradients).all() or not torch.isfinite(
                anchor_gradients).all():
            raise ValueError("gradient sketches must be finite")
