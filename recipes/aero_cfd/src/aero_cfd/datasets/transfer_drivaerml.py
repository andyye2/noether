# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""DrivAerML adapter for transfer from the ShapeNet-Car coordinate frame."""

from __future__ import annotations

from typing import Literal

import torch

from noether.core.schemas.dataset import StandardDatasetConfig
from noether.data import with_normalizers
from noether.data.datasets.cfd.caeml.drivaerml.dataset import DrivAerMLDataset
from noether.data.datasets.cfd.caeml.filemap import CAEML_FILEMAP

#: Dataset property, pipeline item, and normalizer key of the wall-distance
#: feature. The framework derives all three from the ``getitem_`` method name,
#: and ``DataKeys.as_anchor`` requires exactly one underscore.
WALL_DISTANCE_PROPERTY = "volume_distance"

#: Unsigned distance from every volume cell centre to the vehicle surface,
#: precomputed with a KD-tree and stored in metres. ``CAEML_FILEMAP`` leaves
#: its ``volume_distance_to_surface`` slot empty, so the field is named here
#: rather than in the framework: the millimetre convention below only holds for
#: this transfer dataset and must not leak into the production DrivAerML path.
WALL_DISTANCE_FILENAME = "volume_cell_surface_distance_cKDtree.pt"

#: Reference length the wall distance is expressed in. The framework applies
#: ``sign(x) * log1p(|x|)`` before the affine normalization, and ``log1p`` is
#: nearly the identity below its knee. In metres the knee sits at 1 m, which
#: leaves the 70% of cells within 20 mm of the surface compressed into 0.6% of
#: the feature's dynamic range; in millimetres it sits at the near-wall cell
#: spacing (~1.2 mm measured on this mesh) and that share becomes 30.5%. The
#: unit is therefore the physical choice of where the logarithm stops
#: resolving, not a formatting detail.
WALL_DISTANCE_REFERENCE_LENGTH_M = 1e-3


def to_wall_distance_feature(tensor: torch.Tensor) -> torch.Tensor:
    """Express a wall-distance field in units of the reference length.

    Training and statistics fitting must agree on this transform exactly, so
    both call it rather than restating the unit.

    Args:
        tensor: Distances in metres, in any shape.

    Returns:
        The same shape, non-negative, measured in
        :data:`WALL_DISTANCE_REFERENCE_LENGTH_M`.
    """
    return tensor.abs() / WALL_DISTANCE_REFERENCE_LENGTH_M


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

    @with_normalizers
    def getitem_volume_distance(self, idx: int) -> torch.Tensor:
        """Retrieve the wall distance at volume cells ``(num_volume_points, 1)``.

        The value is a scalar per cell and therefore frame-independent. It is
        an input feature, never a target: a preset that does not declare it
        must list ``volume_distance`` in ``excluded_properties`` so the file is
        not read at all.
        """
        return to_wall_distance_feature(self._load(idx=idx, filename=WALL_DISTANCE_FILENAME)).unsqueeze(1)
