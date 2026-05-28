#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from collections.abc import Sequence

import torch


def _broadcast_position_stat(
    value: float | Sequence[float] | torch.Tensor,
    *,
    position_dim: int,
    like: torch.Tensor,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=like.dtype, device=like.device).flatten()
    if tensor.numel() == 1:
        return tensor.expand(position_dim)
    if tensor.numel() == position_dim:
        return tensor
    raise ValueError(f"{name} must contain either 1 value or {position_dim} values, got {tensor.numel()}.")


def denormalize_position(
    position: torch.Tensor,
    *,
    raw_pos_min: float | Sequence[float] | torch.Tensor,
    raw_pos_max: float | Sequence[float] | torch.Tensor,
    position_scale: float = 1000.0,
    zero_center: bool = False,
) -> torch.Tensor:
    """Invert the position normalizer used by the aero CFD datasets."""
    if position.shape[-1] <= 0:
        raise ValueError("position must have a non-empty final coordinate dimension.")

    raw_min = _broadcast_position_stat(
        raw_pos_min,
        position_dim=position.shape[-1],
        like=position,
        name="raw_pos_min",
    )
    raw_max = _broadcast_position_stat(
        raw_pos_max,
        position_dim=position.shape[-1],
        like=position,
        name="raw_pos_max",
    )

    if torch.any(raw_max <= raw_min):
        raise ValueError("raw_pos_max must be element-wise greater than raw_pos_min.")

    if zero_center:
        raw_center = (raw_max + raw_min) / 2.0
        return position * (raw_max - raw_min) / (2.0 * position_scale) + raw_center
    return position * (raw_max - raw_min) / position_scale + raw_min


def _validate_axes(axes: tuple[int, int, int], *, position_dim: int) -> tuple[int, int, int]:
    if len(axes) != 3 or len(set(axes)) != 3:
        raise ValueError(f"axes must contain three unique coordinate indices, got {axes}.")
    if min(axes) < 0 or max(axes) >= position_dim:
        raise ValueError(f"axes {axes} are invalid for position dimension {position_dim}.")
    return axes


def compute_wake_mask(
    surface_position: torch.Tensor,
    volume_position: torch.Tensor,
    *,
    raw_pos_min: float | Sequence[float] | torch.Tensor,
    raw_pos_max: float | Sequence[float] | torch.Tensor,
    box_lwh: tuple[float, float, float] = (0.47, 0.43, 0.31),
    axes: tuple[int, int, int] = (0, 1, 2),
    position_scale: float = 1000.0,
    zero_center: bool = False,
) -> torch.Tensor:
    """Return a mask for the Aultman & Duan 2024 DrivAer wake box.

    The wake-box dimensions are 0.47L x 0.43L x 0.31L in streamwise, spanwise,
    and road-normal directions. The region is computed in raw coordinates so
    dataset position normalization cannot distort the literature-defined box.
    """
    if surface_position.ndim != 2 or volume_position.ndim != 2:
        raise ValueError("surface_position and volume_position must be unbatched tensors with shape (N, D).")
    if surface_position.shape[-1] != volume_position.shape[-1]:
        raise ValueError("surface_position and volume_position must have the same coordinate dimension.")
    if surface_position.numel() == 0 or volume_position.numel() == 0:
        return torch.zeros(volume_position.shape[0], dtype=torch.bool, device=volume_position.device)
    if any(length <= 0.0 for length in box_lwh):
        raise ValueError(f"box_lwh values must be positive, got {box_lwh}.")

    stream_axis, span_axis, vertical_axis = _validate_axes(axes, position_dim=volume_position.shape[-1])

    surface_raw = denormalize_position(
        surface_position,
        raw_pos_min=raw_pos_min,
        raw_pos_max=raw_pos_max,
        position_scale=position_scale,
        zero_center=zero_center,
    )
    volume_raw = denormalize_position(
        volume_position,
        raw_pos_min=raw_pos_min,
        raw_pos_max=raw_pos_max,
        position_scale=position_scale,
        zero_center=zero_center,
    )

    body_front = surface_raw[:, stream_axis].amin()
    rear = surface_raw[:, stream_axis].amax()
    length = rear - body_front
    if length <= 0.0:
        return torch.zeros(volume_position.shape[0], dtype=torch.bool, device=volume_position.device)

    span_center = (surface_raw[:, span_axis].amin() + surface_raw[:, span_axis].amax()) / 2.0
    ground = surface_raw[:, vertical_axis].amin()

    wake_length, wake_width, wake_height = box_lwh
    stream = volume_raw[:, stream_axis]
    span = volume_raw[:, span_axis]
    vertical = volume_raw[:, vertical_axis]

    return (
        (stream >= rear)
        & (stream <= rear + wake_length * length)
        & (span >= span_center - 0.5 * wake_width * length)
        & (span <= span_center + 0.5 * wake_width * length)
        & (vertical >= ground)
        & (vertical <= ground + wake_height * length)
    )
