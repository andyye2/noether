#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch


def central_difference_positions(
    positions: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Return x+, x-, y+, y-, z+, z- perturbations as (B, 6 * N, 3)."""
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape (batch_size, num_points, 3)")

    offsets = positions.new_tensor(
        [
            [delta, 0.0, 0.0],
            [-delta, 0.0, 0.0],
            [0.0, delta, 0.0],
            [0.0, -delta, 0.0],
            [0.0, 0.0, delta],
            [0.0, 0.0, -delta],
        ]
    )
    return (positions[:, None] + offsets[None, :, None]).flatten(1, 2)


def curl_from_perturbed_potential(
    psi: torch.Tensor,
    delta: float,
    position_scale: float | list[float] | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Compute curl from x+, x-, y+, y-, z+, z- vector-potential values."""
    if psi.ndim != 3 or psi.shape[-1] != 3 or psi.shape[1] % 6 != 0:
        raise ValueError("psi must have shape (batch_size, 6 * num_points, 3)")

    psi = psi.float()
    scale = torch.as_tensor(position_scale, device=psi.device, dtype=psi.dtype)
    if scale.numel() == 1:
        scale = scale.expand(3)
    if scale.shape != (3,):
        raise ValueError("position_scale must be a scalar or have shape (3,)")

    psi = psi.reshape(psi.shape[0], 6, psi.shape[1] // 6, 3)
    x_pos, x_neg, y_pos, y_neg, z_pos, z_neg = psi.unbind(dim=1)

    d_dx = (x_pos - x_neg) / (2.0 * delta * scale[0])
    d_dy = (y_pos - y_neg) / (2.0 * delta * scale[1])
    d_dz = (z_pos - z_neg) / (2.0 * delta * scale[2])

    return torch.stack(
        [
            d_dy[..., 2] - d_dz[..., 1],
            d_dz[..., 0] - d_dx[..., 2],
            d_dx[..., 1] - d_dy[..., 0],
        ],
        dim=-1,
    )

def normalize_vector_field(
    x: torch.Tensor,
    mean: list[float] | torch.Tensor,
    std: list[float] | torch.Tensor,
) -> torch.Tensor:
    """Normalize a 3D vector field feature-wise."""
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError("x must have shape (batch_size, num_points, 3)")

    x = x.float()
    mean = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
    std = torch.as_tensor(std, device=x.device, dtype=x.dtype)
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError("mean and std must have shape (3,)")

    return (x - mean) / std
