# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Pattern-based learning-rate modifiers for transfer experiments."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from noether.core.optimizer.param_group_modifiers.base import ParamGroupModifierBase

if TYPE_CHECKING:
    import torch
    from torch import nn

    from noether.core.schemas.optimizers import ParamGroupModifierConfig


def _positive_scale(param_group_modifier_config: ParamGroupModifierConfig) -> float:
    """Return a finite positive LR multiplier."""
    scale = param_group_modifier_config.scale
    if isinstance(scale, bool) or not isinstance(scale, int | float) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {scale!r}")
    return float(scale)


class LrScaleByPatternModifier(ParamGroupModifierBase):
    """Multiply LR for every parameter whose name contains a pattern."""

    def __init__(self, param_group_modifier_config: ParamGroupModifierConfig) -> None:
        if not param_group_modifier_config.name:
            raise ValueError("name must contain a non-empty substring pattern")
        self.scale = _positive_scale(param_group_modifier_config)
        self.pattern = param_group_modifier_config.name
        self.match_count = 0

    def get_properties(self, model: nn.Module, name: str, param: torch.Tensor) -> dict[str, float]:
        """Return an LR multiplier for matching parameter names."""
        del model, param
        if self.pattern in name:
            self.match_count += 1
            return {"lr_scale": self.scale}
        return {}

    def __str__(self) -> str:
        return f"{type(self).__name__}(pattern={self.pattern!r},scale={self.scale})"

    def was_applied_successfully(self) -> bool:
        return self.match_count > 0


class LrScaleExceptPatternModifier(ParamGroupModifierBase):
    """Multiply LR for every parameter whose name excludes a pattern."""

    def __init__(self, param_group_modifier_config: ParamGroupModifierConfig) -> None:
        if not param_group_modifier_config.name:
            raise ValueError("name must contain a non-empty exclusion pattern")
        self.scale = _positive_scale(param_group_modifier_config)
        self.excluded_pattern = param_group_modifier_config.name
        self.match_count = 0

    def get_properties(self, model: nn.Module, name: str, param: torch.Tensor) -> dict[str, float]:
        """Return an LR multiplier unless the exclusion pattern is present."""
        del model, param
        if self.excluded_pattern not in name:
            self.match_count += 1
            return {"lr_scale": self.scale}
        return {}

    def __str__(self) -> str:
        return f"{type(self).__name__}(excluded_pattern={self.excluded_pattern!r},scale={self.scale})"

    def was_applied_successfully(self) -> bool:
        return self.match_count > 0
