#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD

from noether.core.utils.model import compute_model_norm, copy_params, update_ema


class ModelBase(nn.Module):
    """Stub for noether.core.models.base.ModelBase."""

    def __init__(self):
        super().__init__()


class SimpleModel(ModelBase):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2, bias=False)
        self.bn = nn.BatchNorm1d(10, affine=False)


@pytest.fixture
def models():
    """Returns (source, target) models with distinct weights."""
    source = SimpleModel()
    target = SimpleModel()

    with torch.no_grad():
        # Source: Weights=1.0, Buffer=0 (default)
        source.linear.weight.fill_(1.0)
        # Target: Weights=0.0
        target.linear.weight.fill_(0.0)

    return source, target


# --- Tests ---


def test_copy_params_values_and_independence(models):
    """
    Test that parameters are copied correctly AND that they are copied
    by value (not reference).
    """
    source, target = models

    # Snapshot original source weights
    og_source_weight = source.linear.weight.clone()

    # 1. Execute Copy
    copy_params(source_model=source, target_model=target)

    # 2. Verify Values match
    assert torch.equal(source.linear.weight, target.linear.weight)
    assert torch.equal(target.linear.weight, og_source_weight)

    # 3. Verify Independence (Crucial Check from Version 1)
    # Modify source using an optimizer step
    optim = SGD(source.parameters(), lr=1.0)
    loss = source.linear(torch.ones(1, 10)).mean()
    loss.backward()
    optim.step()

    # Source should have changed
    assert not torch.equal(source.linear.weight, og_source_weight)
    # Target should remain at the original copied value (proving deep copy)
    assert torch.equal(target.linear.weight, og_source_weight)


def test_copy_params_buffers(models):
    """Test copying of buffers (e.g., BatchNorm stats)."""
    source, target = models

    # Change source buffer
    source.bn(torch.ones(2, 10))
    assert source.bn.num_batches_tracked != target.bn.num_batches_tracked

    copy_params(source_model=source, target_model=target)

    assert source.bn.num_batches_tracked == target.bn.num_batches_tracked


def test_update_ema_logic(models):
    """Test EMA Math: target = target * factor + source * (1 - factor)"""
    source, target = models

    with torch.no_grad():
        source.linear.weight.fill_(1.0)
        target.linear.weight.fill_(0.5)

    update_ema(source, target, target_factor=0.9, copy_buffers=False)

    # Expected: 0.5 * 0.9 + 1.0 * (1 - 0.9) = 0.45 + 0.1 = 0.55
    # We use allclose because of float precision:
    assert torch.allclose(target.linear.weight, torch.tensor(0.55))


def test_update_ema_buffers(models):
    """Test that buffers are hard-copied (not averaged) when copy_buffers=True."""
    source, target = models

    source.bn(torch.ones(2, 10))
    assert source.bn.num_batches_tracked > target.bn.num_batches_tracked

    # Update with copy_buffers=True
    update_ema(source, target, target_factor=0.9, copy_buffers=True)

    assert source.bn.num_batches_tracked == target.bn.num_batches_tracked


def test_update_ema_ignore_buffers(models):
    """Test that buffers are ignored when copy_buffers=False."""
    source, target = models

    source.bn(torch.ones(2, 10))
    original_target_buffer = target.bn.num_batches_tracked.item()

    # Update with copy_buffers=False
    update_ema(source, target, target_factor=0.9, copy_buffers=False)

    assert target.bn.num_batches_tracked == original_target_buffer


def test_update_ema_fused_implementation_details():
    """
    Verify the fused implementation works on a list of params (nn.Sequential).
    This ensures `torch._foreach_mul_` usage in the code handles lists correctly.
    """
    source = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1))
    target = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1))

    update_ema(source, target, target_factor=0.9)
    # Assertion pass implies no runtime error occurred
    assert True


def test_compute_model_norm():
    """Test calculation of model weight norms."""
    model = nn.Linear(2, 2, bias=False)
    # Manually set weights to simple values [[1.0, 1.0], [1.0, 1.0]]:
    with torch.no_grad():
        model.weight.fill_(1.0)

    # Norm of a tensor full of 1s is sqrt(sum(1^2)).
    # Here we have 4 elements.
    # But compute_model_norm sums the norms of parameters: sum([p.norm() for p...])
    # p.norm() is the L2 norm.
    # Norm of [[1,1],[1,1]] is sqrt(1+1+1+1) = sqrt(4) = 2.0

    expected_norm = 2.0
    calculated_norm = compute_model_norm(model)

    assert torch.isclose(calculated_norm, torch.tensor(expected_norm))


def test_compute_model_norm_with_bias():
    """Test norm with multiple parameters (weight + bias)."""
    model = nn.Linear(1, 1, bias=True)
    with torch.no_grad():
        model.weight.fill_(3.0)  # norm = 3.0
        model.bias.fill_(4.0)  # norm = 4.0

    # The function sums the norms: 3.0 + 4.0 = 7.0
    expected = 7.0
    assert torch.isclose(compute_model_norm(model), torch.tensor(expected))
