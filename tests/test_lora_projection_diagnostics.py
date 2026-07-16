"""Mathematical diagnostics for KAFA's legacy LoRA projection path.

These tests document constraints that a replacement cross-model transfer
method must satisfy. They are deliberately independent of model downloads.
"""

import torch

from federatedscope.llm.misc.projection_mapper import ProjectionMapper


def _relative_error(actual, expected):
    return (actual - expected).norm() / expected.norm().clamp_min(1e-12)


def test_projection_has_expected_shapes():
    mapper = ProjectionMapper(
        small_hidden=8,
        large_hidden=16,
        rank=2,
        learnable=False,
        q_strategy="pseudo_inverse",
    )
    large = {
        "layer.lora_A.weight": torch.randn(2, 16),
        "layer.lora_B.weight": torch.randn(16, 2),
    }

    small = mapper.down_project(large)
    recovered = mapper.up_project(small)

    assert small["layer.lora_A.weight"].shape == (2, 8)
    assert small["layer.lora_B.weight"].shape == (8, 2)
    assert recovered["layer.lora_A.weight"].shape == (2, 16)
    assert recovered["layer.lora_B.weight"].shape == (16, 2)


def test_random_dimension_reduction_is_not_an_inverse():
    """An 8x16 projection necessarily discards arbitrary LoRA information."""
    torch.manual_seed(7)
    mapper = ProjectionMapper(
        small_hidden=8,
        large_hidden=16,
        rank=2,
        learnable=False,
        q_strategy="pseudo_inverse",
    )
    large = {
        "layer.lora_A.weight": torch.randn(2, 16),
        "layer.lora_B.weight": torch.randn(16, 2),
    }
    recovered = mapper.up_project(mapper.down_project(large))

    assert _relative_error(recovered["layer.lora_A.weight"], large["layer.lora_A.weight"]) > 0.1
    assert _relative_error(recovered["layer.lora_B.weight"], large["layer.lora_B.weight"]) > 0.1


def test_factorwise_fedavg_depends_on_arbitrary_lora_gauge():
    """Equivalent client updates can yield a different factor-averaged update.

    BA is invariant under B -> BR and A -> R^-1 A, but averaging A and B
    separately is not. This is the aggregation noise highlighted by FLoRA.
    """
    torch.manual_seed(11)
    a1 = torch.randn(2, 6)
    b1 = torch.randn(5, 2)
    change_of_basis = torch.tensor([[2.0, 0.4], [0.0, 0.5]])
    a2 = torch.linalg.solve(change_of_basis, a1)
    b2 = b1 @ change_of_basis

    assert torch.allclose(b1 @ a1, b2 @ a2, atol=1e-5)

    factor_average = ((b1 + b2) / 2) @ ((a1 + a2) / 2)
    intended_average = ((b1 @ a1) + (b2 @ a2)) / 2
    assert _relative_error(factor_average, intended_average) > 0.01

