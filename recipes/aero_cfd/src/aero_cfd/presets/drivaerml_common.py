# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""DrivAerML preset for fields that are also supervised in ShapeNet-Car."""

from noether.core.schemas.dataset import DomainDataSpec, ModelDataSpecs
from noether.core.schemas.normalizers import FieldNormalizerConfig

from .drivaerml import DrivAerMLPreset


class DrivAerMLCommonFieldsPreset(DrivAerMLPreset):
    """Predict only surface pressure and volume velocity on DrivAerML.

    This target definition is the primary controlled transfer task because the
    two fields are present in both ShapeNet-Car and DrivAerML. Dataset-specific
    normalizers are still used, so no ShapeNet target statistics leak into the
    high-fidelity task.
    """

    @property
    def data_specs(self) -> ModelDataSpecs:
        """Return the two-domain, common-field model specification."""
        return ModelDataSpecs(
            position_dim=3,
            domains={
                "surface": DomainDataSpec(output_dims={"pressure": 1}),
                "volume": DomainDataSpec(output_dims={"velocity": 3}),
            },
            use_physics_features=False,
        )

    @property
    def normalizer_spec(self) -> dict[str, FieldNormalizerConfig]:
        """Return DrivAerML normalizers required by the common-field task."""
        return {
            "surface_pressure": FieldNormalizerConfig(strategy="mean_std"),
            "volume_velocity": FieldNormalizerConfig(strategy="mean_std"),
            "surface_position": FieldNormalizerConfig(
                strategy="position",
                scale=1000,
                stat_keys={"min": "raw_pos_min", "max": "raw_pos_max"},
            ),
            "volume_position": FieldNormalizerConfig(
                strategy="position",
                scale=1000,
                stat_keys={"min": "raw_pos_min", "max": "raw_pos_max"},
            ),
        }

    @property
    def excluded_properties(self) -> set[str]:
        """Avoid loading fields outside the controlled common-field task."""
        return super().excluded_properties | {
            "surface_friction",
            "volume_pressure",
            "volume_vorticity",
        }

    def target_properties(self) -> list[str]:
        """Return the two target keys consumed by the trainer."""
        return ["surface_pressure_target", "volume_velocity_target"]
