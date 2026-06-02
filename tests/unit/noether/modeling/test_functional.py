#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch
from torch import nn

from noether.modeling.functional.init import (
    init_bias_to_zero,
    init_trunc_normal_zero_bias,
)
from noether.modeling.functional.logscale import from_logscale, to_logscale
from noether.modeling.functional.modulation import modulate_gate, modulate_scale_shift
from noether.modeling.functional.rope import rope


@pytest.mark.parametrize(
    "layer",
    [
        (nn.Linear, {"in_features": 3, "out_features": 64, "bias": True}),
        (nn.Conv1d, {"in_channels": 2, "out_channels": 64, "kernel_size": 3, "bias": True}),
        (nn.Conv2d, {"in_channels": 2, "out_channels": 64, "kernel_size": 3, "bias": True}),
        (nn.Conv3d, {"in_channels": 2, "out_channels": 64, "kernel_size": 3, "bias": True}),
        (nn.ConvTranspose1d, {"in_channels": 2, "out_channels": 64, "kernel_size": 3, "bias": True}),
        (nn.ConvTranspose2d, {"in_channels": 2, "out_channels": 64, "kernel_size": 3, "bias": True}),
        (nn.ConvTranspose3d, {"in_channels": 2, "out_channels": 64, "kernel_size": 3, "bias": True}),
    ],
)
def test_init_bias_to_zero(layer):
    # Create a layer with bias
    module, kwargs = layer
    layer = module(**kwargs)
    # Initialize bias to a non-zero value
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 1.0)

    assert torch.all(layer.bias == 1.0)
    # Apply the function
    init_bias_to_zero(layer)

    # Assert that the bias is now zero
    if layer.bias is not None:
        assert torch.all(layer.bias == 0.0)


@pytest.mark.parametrize(
    "layer",
    [
        (nn.Linear, {"in_features": 3, "out_features": 64, "bias": True}),
        (nn.Conv1d, {"in_channels": 2, "out_channels": 64, "kernel_size": 3, "bias": True}),
        (nn.Conv2d, {"in_channels": 2, "out_channels": 64, "kernel_size": 3, "bias": True}),
        (nn.Conv3d, {"in_channels": 2, "out_channels": 64, "kernel_size": 3, "bias": True}),
        (nn.ConvTranspose1d, {"in_channels": 2, "out_channels": 64, "kernel_size": 3, "bias": True}),
        (nn.ConvTranspose2d, {"in_channels": 2, "out_channels": 64, "kernel_size": 3, "bias": True}),
        (nn.ConvTranspose3d, {"in_channels": 2, "out_channels": 64, "kernel_size": 3, "bias": True}),
    ],
)
def test_init_trunc_normal_zero_bias(layer):
    module, kwargs = layer

    # Create a layer without bias
    layer = module(**kwargs)
    # Initialize weights to non-zero values
    nn.init.constant_(layer.weight, 1.0)

    # Apply the function
    init_trunc_normal_zero_bias(layer, std=1.0)

    # Assert that the weights are initialized using truncated normal

    # truncated normal distribution with std=1.0, values should be within -2 and 2.
    assert not torch.all(layer.weight == 1.0)
    assert torch.all(layer.weight >= -2.0)
    assert torch.all(layer.weight <= 2.0)
    # Assert that no bias exists
    if layer.bias is not None:
        assert torch.all(layer.bias == 0.0)


def test_to_logscale():
    # Test with positive values
    x = torch.tensor([1.0, 2.0, 3.0])
    expected = torch.log1p(x)
    result = to_logscale(x)
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"

    # Test with negative values
    x = torch.tensor([-1.0, -2.0, -3.0])
    expected = -torch.log1p(x.abs())
    result = to_logscale(x)
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"

    # Test with zero
    x = torch.tensor([0.0])
    expected = torch.tensor([0.0])
    result = to_logscale(x)
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"


def test_from_logscale():
    # Test with positive values
    x = torch.tensor([0.6931, 1.0986, 1.3863])  # log1p([1.0, 2.0, 3.0])
    expected = torch.tensor([1.0, 2.0, 3.0])
    result = from_logscale(x)
    assert torch.allclose(result, expected, atol=1e-4), f"Expected {expected}, got {result}"

    # Test with negative values
    x = torch.tensor([-0.6931, -1.0986, -1.3863])  # -log1p([1.0, 2.0, 3.0])
    expected = torch.tensor([-1.0, -2.0, -3.0])
    result = from_logscale(x)
    assert torch.allclose(result, expected, atol=1e-4), f"Expected {expected}, got {result}"

    # Test with zero
    x = torch.tensor([0.0])
    expected = torch.tensor([0.0])
    result = from_logscale(x)
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"


def test_to_logscale_and_from_logscale():
    # Test round-trip transformation
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    transformed = to_logscale(x)
    result = from_logscale(transformed)
    assert torch.allclose(result, x, atol=1e-4), f"Expected {x}, got {result}"


def test_scale_shift():
    x = torch.randn(4, 3)
    scale = torch.randn(4, 3)
    shift = torch.randn(4, 3)
    y = modulate_scale_shift(x, scale=scale, shift=shift)
    assert torch.equal(x * (1 + scale) + shift, y)


def test_gate():
    x = torch.randn(4, 3)
    gate = torch.randn(4, 3)
    y = modulate_gate(x, gate=gate)
    assert torch.equal(gate * x, y)


def test_invalid_shape():
    dim = 8
    num_heads = 2
    seqlen = 3
    x = torch.randn(1, num_heads, seqlen, dim)
    freqs = torch.randn(1, dim // num_heads)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    with pytest.raises(AssertionError):
        rope(x=x, freqs=freqs)


def test_invalid_shape_real():
    dim = 8
    num_heads = 2
    seqlen = 3
    x = torch.randn(1, num_heads, seqlen, dim)
    freqs = tuple(torch.randn(1, dim) for _ in range(3))
    with pytest.raises(AssertionError):
        rope(x=x, freqs=freqs)


def test_invalid_shape_x():
    dim = 8
    seqlen = 3
    num_heads = 2
    x = torch.randn(1, seqlen, dim)
    freqs = torch.randn(1, seqlen, dim // num_heads)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    with pytest.raises(AssertionError):
        rope(x=x, freqs=freqs)
