#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import Any

from .base import AeroPipelineParams
from .shapenet_car import ShapeNetCarPreset

_DIVFREE_KIND = "noether.modeling.models.divfree_aerodynamics.DivFreeAeroABUPT"
_ABUPT_KIND = "noether.modeling.models.aerodynamics.AeroABUPT"


class ShapeNetCarDivFreePreset(ShapeNetCarPreset):
    """ShapeNet-Car preset that wires DivFreeAeroABUPT."""

    pipeline_model_overrides: dict[str, AeroPipelineParams] = {
        **ShapeNetCarPreset.pipeline_model_overrides,
        _DIVFREE_KIND: dict(ShapeNetCarPreset.pipeline_model_overrides[_ABUPT_KIND]),
    }

    def build_model(
        self,
        *,
        model_kind: str,
        optimizer: Any = None,
        **model_params: Any,
    ) -> Any:
        if model_kind == _DIVFREE_KIND:
            stats = self.dataset_statistics
            model_params.setdefault("name", "divfree_ab_upt")
            model_params.setdefault("position_scale", self._isotropic_position_scale(stats))
            model_params.setdefault("volume_velocity_mean", list(stats["volume_velocity_mean"]))
            model_params.setdefault("volume_velocity_std", list(stats["volume_velocity_std"]))
        return super().build_model(model_kind=model_kind, optimizer=optimizer, **model_params)

    @staticmethod
    def _isotropic_position_scale(stats: dict[str, Any]) -> float:
        raw_min = stats["raw_pos_min"][0]
        raw_max = stats["raw_pos_max"][0]
        normalizer_scale = 1000.0
        return (raw_max - raw_min) / normalizer_scale
