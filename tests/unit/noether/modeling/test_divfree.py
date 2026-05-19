#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.modeling.functional.divfree import (
    central_difference_positions,
    curl_from_perturbed_potential,
    divergence_from_perturbed_field,
    normalize_vector_field,
)


def test_central_difference_positions_shape_and_order():
    positions = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])

    out = central_difference_positions(positions, delta=0.5)

    expected = torch.tensor(
        [
            [
                [1.5, 2.0, 3.0],
                [4.5, 5.0, 6.0],
                [0.5, 2.0, 3.0],
                [3.5, 5.0, 6.0],
                [1.0, 2.5, 3.0],
                [4.0, 5.5, 6.0],
                [1.0, 1.5, 3.0],
                [4.0, 4.5, 6.0],
                [1.0, 2.0, 3.5],
                [4.0, 5.0, 6.5],
                [1.0, 2.0, 2.5],
                [4.0, 5.0, 5.5],
            ]
        ]
    )
    assert torch.equal(out, expected)


def test_central_difference_positions_preserves_dtype_and_device():
    positions = torch.zeros(2, 3, 3, dtype=torch.float64)

    out = central_difference_positions(positions, delta=1e-3)

    assert out.shape == (2, 18, 3)
    assert out.dtype == positions.dtype
    assert out.device == positions.device


def test_central_difference_positions_rejects_invalid_shape():
    with pytest.raises(ValueError, match="positions must have shape"):
        central_difference_positions(torch.zeros(2, 3), delta=1e-3)


def test_curl_from_perturbed_potential_zero():
    psi = torch.zeros(2, 18, 3)

    out = curl_from_perturbed_potential(psi, delta=1e-3)

    assert torch.equal(out, torch.zeros(2, 3, 3))


def test_curl_from_perturbed_potential_matches_analytic_field():
    positions = torch.tensor([[[2.0, 3.0, 5.0], [-1.0, 4.0, 0.5]]])
    perts = central_difference_positions(positions, delta=1e-3)

    psi = torch.zeros_like(perts)
    psi[..., 2] = perts[..., 0] * perts[..., 1]

    out = curl_from_perturbed_potential(psi, delta=1e-3)

    expected = torch.stack(
        [
            positions[..., 0],
            -positions[..., 1],
            torch.zeros_like(positions[..., 0]),
        ],
        dim=-1,
    )
    assert torch.allclose(out, expected, atol=5e-4)


def test_curl_from_perturbed_potential_applies_position_scale():
    positions = torch.tensor([[[2.0, 3.0, 5.0]]])
    perts = central_difference_positions(positions, delta=1e-3)

    psi = torch.zeros_like(perts)
    psi[..., 2] = perts[..., 0] * perts[..., 1]

    out = curl_from_perturbed_potential(
        psi,
        delta=1e-3,
        position_scale=[2.0, 4.0, 8.0],
    )

    assert torch.allclose(out, torch.tensor([[[0.5, -1.5, 0.0]]]), atol=5e-4)


def test_curl_from_perturbed_potential_returns_float32():
    psi = torch.zeros(1, 6, 3, dtype=torch.float64)

    out = curl_from_perturbed_potential(psi, delta=1e-3)

    assert out.dtype == torch.float32


def test_curl_from_perturbed_potential_rejects_invalid_shape():
    with pytest.raises(ValueError, match="psi must have shape"):
        curl_from_perturbed_potential(torch.zeros(1, 5, 3), delta=1e-3)


def test_divergence_from_perturbed_field_zero():
    field = torch.zeros(2, 18, 3)

    out = divergence_from_perturbed_field(field, delta=1e-3)

    assert torch.equal(out, torch.zeros(2, 3))


def test_divergence_from_perturbed_field_matches_analytic_field():
    positions = torch.tensor([[[2.0, 3.0, 5.0], [-1.0, 4.0, 0.5]]])
    perts = central_difference_positions(positions, delta=1e-3)

    field = torch.zeros_like(perts)
    field[..., 0] = perts[..., 0]
    field[..., 1] = 2.0 * perts[..., 1]
    field[..., 2] = 3.0 * perts[..., 2]

    out = divergence_from_perturbed_field(field, delta=1e-3)

    assert torch.allclose(out, torch.full((1, 2), 6.0), atol=5e-4)


def test_divergence_from_perturbed_field_applies_position_scale():
    positions = torch.tensor([[[2.0, 3.0, 5.0]]])
    perts = central_difference_positions(positions, delta=1e-3)

    field = torch.zeros_like(perts)
    field[..., 0] = perts[..., 0]

    out = divergence_from_perturbed_field(
        field,
        delta=1e-3,
        position_scale=[2.0, 4.0, 8.0],
    )

    assert torch.allclose(out, torch.tensor([[0.5]]), atol=5e-4)


def test_divergence_from_perturbed_field_returns_float32():
    field = torch.zeros(1, 6, 3, dtype=torch.float64)

    out = divergence_from_perturbed_field(field, delta=1e-3)

    assert out.dtype == torch.float32


def test_divergence_from_perturbed_field_rejects_invalid_shape():
    with pytest.raises(ValueError, match="field must have shape"):
        divergence_from_perturbed_field(torch.zeros(1, 5, 3), delta=1e-3)


def test_normalize_vector_field():
    x = torch.tensor([[[3.0, 4.0, 5.0], [-1.0, 0.0, 1.0]]])

    out = normalize_vector_field(
        x,
        mean=[1.0, 2.0, 3.0],
        std=[2.0, 4.0, 0.5],
    )

    expected = torch.tensor([[[1.0, 0.5, 4.0], [-1.0, -0.5, -4.0]]])
    assert torch.equal(out, expected)


def test_normalize_vector_field_returns_float32():
    x = torch.zeros(1, 2, 3, dtype=torch.float64)

    out = normalize_vector_field(
        x,
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
    )

    assert out.dtype == torch.float32


def test_normalize_vector_field_rejects_invalid_shape():
    with pytest.raises(ValueError, match="x must have shape"):
        normalize_vector_field(
            torch.zeros(1, 2),
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
        )
