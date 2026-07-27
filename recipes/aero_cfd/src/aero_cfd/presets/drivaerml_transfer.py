# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Leakage-safe DrivAerML presets for ShapeNet-Car transfer experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

import yaml

from aero_cfd.datasets.transfer_drivaerml import TransferDrivAerMLDatasetConfig
from noether.core.schemas.dataset import DatasetBaseConfig, DatasetWrappers
from noether.core.schemas.normalizers import (
    MeanStdNormalizerConfig,
    PositionNormalizerConfig,
)

from .drivaerml import DrivAerMLPreset
from .drivaerml_common import DrivAerMLCommonFieldsPreset

_FIELD_COMPONENTS = {
    "surface_pressure": 1,
    "surface_friction": 3,
    "volume_pressure": 1,
    "volume_velocity": 3,
    "volume_vorticity": 3,
}

#: RoPE/sincos maximum wavelength of the source-compatible AB-UPT architecture.
#: Normalized positions span ``[0, position_scale]``; a scale above this bound
#: would wrap the coarsest positional-encoding band, making distant points
#: share an embedding. A scale equal to the bound spans exactly one period.
_MAX_NORMALIZED_POSITION = 10000.0
_DEFAULT_POSITION_SCALE = 1000.0


def _read_mapping(path: Path) -> dict[str, Any]:
    """Read one JSON/YAML mapping without accepting ambiguous extensions."""
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"statistics artifact must be JSON or YAML, got {path}")
    if not isinstance(value, dict):
        raise TypeError(f"statistics artifact must contain a mapping, got {type(value).__name__}")
    return value


def _validate_numeric_stat(key: str, value: Any) -> list[float] | float:
    """Return one finite numeric statistic without silently dropping bad data."""
    values = value if isinstance(value, list) else [value]
    if not values:
        raise ValueError(f"statistics artifact entry {key!r} must not be empty")
    if any(isinstance(item, bool) or not isinstance(item, int | float) for item in values):
        raise ValueError(f"statistics artifact entry {key!r} must contain only numbers")
    converted = [float(item) for item in values]
    if not all(math.isfinite(item) for item in converted):
        raise ValueError(f"statistics artifact entry {key!r} must contain only finite numbers")
    return converted if isinstance(value, list) else converted[0]


class _DrivAerMLTransferStatsMixin:
    """Resolve concrete normalizers from one train-subset-only artifact.

    Args:
        statistics_artifact: Train-subset-only statistics artifact (JSON/YAML).
        coordinate_frame: ``"native"`` or ``"shapenet"``; must match the artifact.
        expected_manifest_sha256: Raw-file SHA256 of the frozen subset manifest.
        expected_train_subset_size: Exact N recorded in the artifact.
        position_scale: Upper bound of the normalized position range
            ``[0, position_scale]``. Defaults to 1000.0 (the frozen
            confirmatory value), which renders a DrivAerML car only ~39
            normalized units long against a 10,000-unit coarsest positional
            band. Exploratory runs may raise it so the car occupies a
            source-like fraction of that band; it must not exceed the
            architecture's RoPE/sincos maximum wavelength, beyond which
            distant points would alias onto the same coarse embedding.
    """

    coordinate_frame: Literal["native", "shapenet"]

    def __init__(
        self,
        *,
        statistics_artifact: Path,
        coordinate_frame: Literal["native", "shapenet"],
        expected_manifest_sha256: str,
        expected_train_subset_size: int,
        position_scale: float = _DEFAULT_POSITION_SCALE,
    ) -> None:
        artifact = _read_mapping(statistics_artifact)
        if artifact.get("train_subset_size") != expected_train_subset_size:
            raise ValueError(
                "statistics train_subset_size mismatch: "
                f"expected {expected_train_subset_size}, got {artifact.get('train_subset_size')}"
            )
        if artifact.get("coordinate_frame") != coordinate_frame:
            raise ValueError(
                "statistics coordinate_frame mismatch: "
                f"expected {coordinate_frame!r}, got {artifact.get('coordinate_frame')!r}"
            )
        provenance = artifact.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("statistics artifact has no provenance mapping")
        if provenance.get("manifest_sha256") != expected_manifest_sha256:
            raise ValueError(
                "statistics/manifest SHA256 mismatch: "
                f"expected {expected_manifest_sha256}, got {provenance.get('manifest_sha256')}"
            )
        leakage_guard = artifact.get("leakage_guard")
        if not isinstance(leakage_guard, dict) or leakage_guard.get("target_splits_read") != ["train"]:
            raise ValueError("statistics artifact does not attest train-only target access")
        stats = artifact.get("normalizer_stats")
        if not isinstance(stats, dict):
            raise ValueError("statistics artifact has no normalizer_stats mapping")

        if (
            isinstance(position_scale, bool)
            or not isinstance(position_scale, int | float)
            or not math.isfinite(position_scale)
            or position_scale <= 0.0
        ):
            raise ValueError(f"position_scale must be finite and positive, got {position_scale!r}")
        if position_scale > _MAX_NORMALIZED_POSITION:
            raise ValueError(
                f"position_scale must not exceed the RoPE/sincos maximum wavelength "
                f"{_MAX_NORMALIZED_POSITION}, got {position_scale!r}"
            )

        self.statistics_artifact = statistics_artifact.resolve()
        self.coordinate_frame = coordinate_frame
        self.expected_manifest_sha256 = expected_manifest_sha256
        self.expected_train_subset_size = expected_train_subset_size
        self.position_scale = float(position_scale)
        self._transfer_stats = {key: _validate_numeric_stat(key, value) for key, value in stats.items()}
        self.build_normalizers()

    @property
    def dataset_statistics(self) -> dict[str, list[float] | float]:
        """Return only the current frozen training-subset statistics."""
        return dict(self._transfer_stats)

    def _required_stat(self, key: str) -> list[float] | float:
        """Return one statistic or raise with the complete available-key set."""
        if key not in self._transfer_stats:
            raise ValueError(f"statistics artifact is missing {key!r}; available={sorted(self._transfer_stats)}")
        return self._transfer_stats[key]

    @staticmethod
    def _stat_values(key: str, value: list[float] | float) -> list[float]:
        """Return a list view used for exact component and range checks."""
        values = value if isinstance(value, list) else [value]
        if not values:
            raise ValueError(f"statistics artifact entry {key!r} must not be empty")
        return values

    def build_normalizers(self) -> dict[str, list[Any]]:
        """Build numeric configs so the dataset never falls back to full-train stats.

        Position-strategy fields use ``self.position_scale`` as the normalized
        upper bound instead of the declaration default, so one preset instance
        renders geometry at exactly one audited scale.
        """
        normalizers: dict[str, list[Any]] = {}
        for field, declaration in self.normalizer_spec.items():
            stat_keys = declaration.stat_keys or {}
            if declaration.strategy == "mean_std":
                mean_key = stat_keys.get("mean", f"{field}_mean")
                std_key = stat_keys.get("std", f"{field}_std")
                mean = self._required_stat(mean_key)
                std = self._required_stat(std_key)
                mean_values = self._stat_values(mean_key, mean)
                std_values = self._stat_values(std_key, std)
                if len(mean_values) != len(std_values):
                    raise ValueError(
                        f"statistics artifact entries {mean_key!r} and {std_key!r} must have equal lengths"
                    )
                expected_components = _FIELD_COMPONENTS.get(field)
                if expected_components is not None and len(mean_values) != expected_components:
                    raise ValueError(
                        f"statistics artifact field {field!r} requires {expected_components} component(s), "
                        f"got {len(mean_values)}"
                    )
                if any(value < 0.0 for value in std_values):
                    raise ValueError(f"statistics artifact entry {std_key!r} must be non-negative")
                normalizers[field] = [
                    MeanStdNormalizerConfig(
                        mean=mean,
                        std=std,
                        logscale=declaration.logscale,
                    )
                ]
            elif declaration.strategy in {"position", "min_max"}:
                min_key = stat_keys.get("min", f"{field}_min")
                max_key = stat_keys.get("max", f"{field}_max")
                minimum = self._required_stat(min_key)
                maximum = self._required_stat(max_key)
                minimum_values = self._stat_values(min_key, minimum)
                maximum_values = self._stat_values(max_key, maximum)
                if len(minimum_values) != len(maximum_values) or len(minimum_values) not in {1, 3}:
                    raise ValueError(f"position statistics {min_key!r}/{max_key!r} must have matching length 1 or 3")
                if any(upper <= lower for lower, upper in zip(minimum_values, maximum_values, strict=True)):
                    raise ValueError(f"position statistics {max_key!r} must be element-wise greater than {min_key!r}")
                normalizers[field] = [
                    PositionNormalizerConfig(
                        raw_pos_min=minimum,
                        raw_pos_max=maximum,
                        scale=self.position_scale,
                        zero_center=declaration.zero_center,
                    )
                ]
            else:
                raise ValueError(f"unsupported normalizer strategy {declaration.strategy!r}")
        return normalizers

    def build_dataset(
        self,
        *,
        split: str,
        root: str,
        model_kind: str,
        wrappers: list[DatasetWrappers] | None = None,
        **overrides: Any,
    ) -> DatasetBaseConfig:
        """Build the coordinate-aware dataset with concrete subset normalizers."""
        return TransferDrivAerMLDatasetConfig(
            kind="aero_cfd.datasets.transfer_drivaerml.TransferDrivAerMLDataset",
            root=root,
            split=split,
            coordinate_frame=self.coordinate_frame,
            pipeline=self.build_pipeline(model_kind, **overrides),
            dataset_normalizers=self.build_normalizers(),
            dataset_wrappers=wrappers,
            excluded_properties=self.excluded_properties,
        )


class DrivAerMLTransferCommonPreset(_DrivAerMLTransferStatsMixin, DrivAerMLCommonFieldsPreset):
    """Common-field preset bound to one strict train-subset statistics artifact."""


class DrivAerMLTransferFullPreset(_DrivAerMLTransferStatsMixin, DrivAerMLPreset):
    """Full-field preset bound to one strict train-subset statistics artifact."""
