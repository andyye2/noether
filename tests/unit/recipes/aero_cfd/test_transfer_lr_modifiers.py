# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Tests for transfer learning-rate pattern modifiers."""

from __future__ import annotations

import pytest
import torch

from noether.core.factory import Factory
from noether.core.schemas.optimizers import ParamGroupModifierConfig


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_modifiers_reject_invalid_lr_scales(scale: float) -> None:
    """LR multipliers must remain finite and strictly positive."""
    for kind in (
        "aero_cfd.optimizer.transfer_lr_modifiers.LrScaleByPatternModifier",
        "aero_cfd.optimizer.transfer_lr_modifiers.LrScaleExceptPatternModifier",
    ):
        with pytest.raises(ValueError):
            config = ParamGroupModifierConfig(kind=kind, name="backbone", scale=scale)
            Factory().create(config)


def test_factory_resolves_pattern_and_exclusion_modifiers() -> None:
    """Dynamic config paths produce the intended multiplicative LR masks."""
    body_config = ParamGroupModifierConfig(
        kind="aero_cfd.optimizer.transfer_lr_modifiers.LrScaleExceptPatternModifier",
        name="domain_decoder_projections",
        scale=0.1,
    )
    decoder_config = ParamGroupModifierConfig(
        kind="aero_cfd.optimizer.transfer_lr_modifiers.LrScaleByPatternModifier",
        name="domain_decoder_blocks",
        scale=3.0,
    )
    body = Factory().create(body_config)
    decoder = Factory().create(decoder_config)
    parameter = torch.nn.Parameter(torch.ones(1))

    assert body.get_properties(torch.nn.Linear(1, 1), "backbone.physics_blocks.0.weight", parameter) == {
        "lr_scale": 0.1
    }
    assert (
        body.get_properties(
            torch.nn.Linear(1, 1),
            "backbone.domain_decoder_projections.surface.linear.weight",
            parameter,
        )
        == {}
    )
    assert decoder.get_properties(
        torch.nn.Linear(1, 1),
        "backbone.domain_decoder_blocks.surface.0.weight",
        parameter,
    ) == {"lr_scale": 3.0}
    assert body.was_applied_successfully()
    assert decoder.was_applied_successfully()
