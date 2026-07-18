# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""DrivAerML adapter for transfer from the ShapeNet-Car coordinate frame."""

from __future__ import annotations

from typing import Literal

import torch

from noether.core.schemas.dataset import StandardDatasetConfig
from noether.data.datasets.cfd.caeml.drivaerml.dataset import DrivAerMLDataset
from noether.data.datasets.cfd.caeml.filemap import CAEML_FILEMAP


class TransferDrivAerMLDatasetConfig(StandardDatasetConfig):
    """Configuration for coordinate-aware DrivAerML transfer experiments.

    Attributes:
        coordinate_frame: ``"shapenet"`` applies the cyclic component mapping
            ``(x_D, y_D, z_D) -> (y_D, z_D, x_D)`` to every position and vector
            field. ``"native"`` preserves the stored DrivAer coordinates.
    """

    coordinate_frame: Literal["native", "shapenet"] = "shapenet"


_FRAME_DEPENDENT_FILENAMES = frozenset(
    filename
    for filename in (
        CAEML_FILEMAP.surface_position,
        CAEML_FILEMAP.surface_friction,
        CAEML_FILEMAP.surface_normals,
        CAEML_FILEMAP.volume_position,
        CAEML_FILEMAP.volume_velocity,
        CAEML_FILEMAP.volume_vorticity,
        CAEML_FILEMAP.volume_normals,
    )
    if filename is not None
)


def drivaer_to_shapenet_frame(tensor: torch.Tensor) -> torch.Tensor:
    """Map DrivAer vector components into the ShapeNet-Car frame.

    The map is the proper cyclic rotation ``(x_D, y_D, z_D) ->
    (y_D, z_D, x_D)``. It therefore applies identically to positions, polar
    vectors (velocity, wall shear, normals), and the axial vorticity vector.

    Args:
        tensor: Tensor whose final dimension contains three Cartesian
            components.

    Returns:
        Tensor with the final dimension cyclically permuted.

    Raises:
        ValueError: If the final dimension is not three.
    """
    if tensor.ndim == 0 or tensor.shape[-1] != 3:
        raise ValueError(f"Expected a Cartesian tensor with last dimension 3, got shape {tuple(tensor.shape)}")
    return tensor[..., (1, 2, 0)]


def transform_loaded_drivaerml_field(
    tensor: torch.Tensor,
    filename: str,
    *,
    coordinate_frame: Literal["native", "shapenet"],
) -> torch.Tensor:
    """Transform one loaded DrivAerML field when it depends on frame axes.

    Scalar fields such as pressure and area are returned unchanged. Keeping the
    decision at the filename boundary makes it impossible to rotate geometry
    while accidentally leaving a vector target in the native frame.

    Args:
        tensor: Tensor loaded from a run directory.
        filename: Filename used by the DrivAerML file map.
        coordinate_frame: Requested output frame.

    Returns:
        The original tensor in native mode or for scalar fields; otherwise the
        consistently component-permuted tensor.

    Raises:
        ValueError: If ``coordinate_frame`` is invalid or a frame-dependent
            tensor does not have three Cartesian components.
    """
    if coordinate_frame not in {"native", "shapenet"}:
        raise ValueError(f"Unsupported coordinate frame: {coordinate_frame!r}")
    if coordinate_frame == "shapenet" and filename in _FRAME_DEPENDENT_FILENAMES:
        return drivaer_to_shapenet_frame(tensor)
    return tensor


class TransferDrivAerMLDataset(DrivAerMLDataset):
    """DrivAerML dataset with an explicit native/ShapeNet coordinate choice.

    Rotation happens inside :meth:`_load`, before the inherited ``getitem``
    methods apply their normalizers. Thus both input geometry and vector labels
    are expressed in the same frame, and vector normalization statistics can be
    generated in that exact frame.

    Args:
        dataset_config: Root, split, normalizers, and coordinate-frame choice.
    """

    def __init__(self, dataset_config: TransferDrivAerMLDatasetConfig) -> None:
        self.coordinate_frame = dataset_config.coordinate_frame
        super().__init__(dataset_config=dataset_config)

    def _load(self, idx: int, filename: str) -> torch.Tensor:
        """Load one tensor and consistently express it in the requested frame."""
        tensor = super()._load(idx=idx, filename=filename)
        return transform_loaded_drivaerml_field(
            tensor,
            filename,
            coordinate_frame=self.coordinate_frame,
        )
