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
    """Resolve concrete normalizers from one train-subset-only artifact."""

    coordinate_frame: Literal["native", "shapenet"]

    def __init__(
        self,
        *,
        statistics_artifact: Path,
        coordinate_frame: Literal["native", "shapenet"],
        expected_manifest_sha256: str,
        expected_train_subset_size: int,
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

        self.statistics_artifact = statistics_artifact.resolve()
        self.coordinate_frame = coordinate_frame
        self.expected_manifest_sha256 = expected_manifest_sha256
        self.expected_train_subset_size = expected_train_subset_size
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
        """Build numeric configs so the dataset never falls back to full-train stats."""
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
                        scale=declaration.scale,
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
