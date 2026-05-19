#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import torch


def _as_position_scale(
    position_scale: float | list[float] | torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    scale = torch.as_tensor(position_scale, device=reference.device, dtype=reference.dtype)
    if scale.numel() == 1:
        scale = scale.expand(3)
    if scale.shape != (3,):
        raise ValueError("position_scale must be a scalar or have shape (3,)")
    return scale


def central_difference_positions(
    positions: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Build central-difference query positions.

    Args:
        positions: Anchor/query positions with shape ``(batch_size, num_points, 3)``.
        delta: Perturbation step in network coordinates.

    Returns:
        Perturbed positions with shape ``(batch_size, 6 * num_points, 3)``,
        ordered as ``[x+, x-, y+, y-, z+, z-]``.

    Example:

        .. testcode::

            import torch
            from noether.modeling.functional.divfree import central_difference_positions

            positions = torch.zeros(1, 2, 3)
            perturbed = central_difference_positions(positions, delta=1e-3)
            assert perturbed.shape == (1, 12, 3)

    Raises:
        ValueError: If positions do not have shape ``(B, N, 3)``.
    """
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
    """Compute ``curl(psi)`` from central-difference vector-potential values.

    Args:
        psi: Vector potential with shape ``(batch_size, 6 * num_points, 3)``,
            ordered as ``[x+, x-, y+, y-, z+, z-]``.
        delta: Perturbation step in network coordinates.
        position_scale: ``dx_phys / dx_net``. Use a scalar for isotropic
            scaling or a length-3 value for anisotropic scaling.

    Returns:
        Physical velocity with shape ``(batch_size, num_points, 3)`` and dtype
        ``float32``.

    Example:

        .. testcode::

            import torch
            from noether.modeling.functional.divfree import curl_from_perturbed_potential

            psi = torch.zeros(1, 6, 3)
            velocity = curl_from_perturbed_potential(psi, delta=1e-3)
            assert torch.equal(velocity, torch.zeros(1, 1, 3))

    Raises:
        ValueError: If psi does not have shape ``(B, 6 * N, 3)``.
    """
    if psi.ndim != 3 or psi.shape[-1] != 3 or psi.shape[1] % 6 != 0:
        raise ValueError("psi must have shape (batch_size, 6 * num_points, 3)")

    psi = psi.float()
    scale = _as_position_scale(position_scale, reference=psi)
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


def divergence_from_perturbed_field(
    field: torch.Tensor,
    delta: float,
    position_scale: float | list[float] | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Compute numerical divergence from central-difference vector-field values.

    Args:
        field: Vector field with shape ``(batch_size, 6 * num_points, 3)``,
            ordered as ``[x+, x-, y+, y-, z+, z-]``.
        delta: Perturbation step in network coordinates.
        position_scale: ``dx_phys / dx_net``. Use a scalar for isotropic
            scaling or a length-3 value for anisotropic scaling.

    Returns:
        Physical divergence with shape ``(batch_size, num_points)`` and dtype
        ``float32``.

    Example:

        .. testcode::

            import torch
            from noether.modeling.functional.divfree import divergence_from_perturbed_field

            field = torch.zeros(1, 6, 3)
            divergence = divergence_from_perturbed_field(field, delta=1e-3)
            assert torch.equal(divergence, torch.zeros(1, 1))

    Raises:
        ValueError: If field does not have shape ``(B, 6 * N, 3)``.
    """
    if field.ndim != 3 or field.shape[-1] != 3 or field.shape[1] % 6 != 0:
        raise ValueError("field must have shape (batch_size, 6 * num_points, 3)")

    field = field.float()
    scale = _as_position_scale(position_scale, reference=field)
    field = field.reshape(field.shape[0], 6, field.shape[1] // 6, 3)
    x_pos, x_neg, y_pos, y_neg, z_pos, z_neg = field.unbind(dim=1)

    d_ux_dx = (x_pos[..., 0] - x_neg[..., 0]) / (2.0 * delta * scale[0])
    d_uy_dy = (y_pos[..., 1] - y_neg[..., 1]) / (2.0 * delta * scale[1])
    d_uz_dz = (z_pos[..., 2] - z_neg[..., 2]) / (2.0 * delta * scale[2])
    return d_ux_dx + d_uy_dy + d_uz_dz


def normalize_vector_field(
    x: torch.Tensor,
    mean: list[float] | torch.Tensor,
    std: list[float] | torch.Tensor,
) -> torch.Tensor:
    """Normalize a 3D vector field feature-wise.

    Args:
        x: Vector field with shape ``(batch_size, num_points, 3)``.
        mean: Feature mean with shape ``(3,)``.
        std: Feature standard deviation with shape ``(3,)``.

    Returns:
        Normalized vector field with shape ``(batch_size, num_points, 3)`` and
        dtype ``float32``.

    Example:

        .. testcode::

            import torch
            from noether.modeling.functional.divfree import normalize_vector_field

            x = torch.ones(1, 2, 3)
            y = normalize_vector_field(x, mean=[1.0, 1.0, 1.0], std=[1.0, 1.0, 1.0])
            assert torch.equal(y, torch.zeros(1, 2, 3))

    Raises:
        ValueError: If x, mean, or std do not have 3 vector components.
    """
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError("x must have shape (batch_size, num_points, 3)")

    x = x.float()
    mean = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
    std = torch.as_tensor(std, device=x.device, dtype=x.dtype)
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError("mean and std must have shape (3,)")

    return (x - mean) / std
