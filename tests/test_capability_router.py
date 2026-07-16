import torch

from federatedscope.llm.knowledge_anchor.capability_router import (
    CapabilityGradientRouter,
)


def test_router_prefers_aligned_samples():
    router = CapabilityGradientRouter(temperature=0.05,
                                      uncertainty_beta=0.0)
    anchors = torch.tensor([[1.0, 0.0], [0.9, 0.1]])
    samples = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    result = router.route(samples, anchors)

    assert result.weights[0] > 0.99
    assert result.weights[1] < 0.01
    assert result.weights[0] > result.weights[2] > result.weights[1]


def test_uncertainty_penalizes_inconsistent_anchor_agreement():
    anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    consistent = torch.tensor([[1.0, 1.0]])
    inconsistent = torch.tensor([[1.0, 0.0]])
    router = CapabilityGradientRouter(temperature=0.1,
                                      uncertainty_beta=1.0)
    result = router.route(torch.cat([consistent, inconsistent]), anchors)

    assert result.mean_utility[0] > result.mean_utility[1]
    assert result.utility_std[0] < result.utility_std[1]
    assert result.weights[0] > result.weights[1]


def test_shared_gradient_is_not_first_order_anchor_conflicting():
    router = CapabilityGradientRouter(temperature=1.0,
                                      uncertainty_beta=0.0,
                                      min_weight=0.2)
    anchors = torch.tensor([[1.0, 0.0]])
    samples = torch.tensor([[-2.0, 1.0], [-1.0, -0.5]])
    result = router.route(samples, anchors)

    assert torch.dot(result.shared_gradient, anchors.mean(0)) >= -1e-6


def test_router_rejects_invalid_inputs():
    router = CapabilityGradientRouter()
    try:
        router.route(torch.randn(2, 3), torch.randn(2, 4))
    except ValueError as error:
        assert "share a dimension" in str(error)
    else:
        raise AssertionError("dimension mismatch should fail")

