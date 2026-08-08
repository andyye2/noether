# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Tests for the named transfer reset scopes."""

from __future__ import annotations

import pytest

from aero_cfd.model.transfer_reset import (
    DEFAULT_RESET_SCOPE,
    READOUT_PATTERN,
    RESET_SCOPES,
    reset_patterns,
)
from aero_cfd.presets.drivaerml_common import DrivAerMLCommonFieldsPreset
from noether.modeling.models.aerodynamics import AeroABUPT
from recipes.aero_cfd.scripts.run_drivaerml_transfer_strict import (
    CHECKPOINT_ARCHITECTURE,
    MODEL_KIND,
)

#: Parameter counts of the frozen ShapeNet-Car architecture, per scope.
EXPECTED_RESET_PARAMETERS = {
    "readout": 1_540,
    "volume_decoder": 891_268,
    "volume_path": 965_380,
    "decoder": 1_780_996,
}
TOTAL_PARAMETERS = 7_008_004


@pytest.fixture(scope="module")
def parameter_names() -> list[str]:
    """Parameter names of one CPU model with the frozen source architecture."""
    preset = DrivAerMLCommonFieldsPreset()
    model = AeroABUPT(model_config=preset.build_model(model_kind=MODEL_KIND, **CHECKPOINT_ARCHITECTURE))
    return [name for name, _ in model.named_parameters()]


def test_default_scope_is_the_frozen_readout_only() -> None:
    """The published study identity depends on this exact pattern list."""
    assert reset_patterns(DEFAULT_RESET_SCOPE) == ["backbone.domain_decoder_projections"]


def test_every_scope_resets_the_incompatible_readout() -> None:
    """The target predicts other fields, so no scope may inherit the readout."""
    for scope in RESET_SCOPES:
        assert READOUT_PATTERN in reset_patterns(scope)


def test_unknown_scope_is_rejected() -> None:
    """A typo must fail loudly rather than silently inherit everything."""
    with pytest.raises(ValueError, match="unknown reset scope"):
        reset_patterns("volume")


def test_returned_patterns_cannot_mutate_the_scope_table() -> None:
    """Callers receive a fresh list; the module table stays authoritative."""
    patterns = reset_patterns(DEFAULT_RESET_SCOPE)
    patterns.append("backbone.physics_blocks")
    assert reset_patterns(DEFAULT_RESET_SCOPE) == ["backbone.domain_decoder_projections"]


@pytest.mark.parametrize("scope", sorted(RESET_SCOPES))
def test_every_pattern_matches_real_parameters(scope: str, parameter_names: list[str]) -> None:
    """A pattern that matches nothing would silently reset nothing at all."""
    for pattern in reset_patterns(scope):
        assert any(pattern in name for name in parameter_names), pattern


@pytest.mark.parametrize("scope", sorted(RESET_SCOPES))
def test_scope_resets_the_documented_parameter_count(scope: str, parameter_names: list[str]) -> None:
    """Guard the accounting the experiment design and the docstring rely on."""
    preset = DrivAerMLCommonFieldsPreset()
    model = AeroABUPT(model_config=preset.build_model(model_kind=MODEL_KIND, **CHECKPOINT_ARCHITECTURE))
    patterns = reset_patterns(scope)
    reset = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if any(pattern in name for pattern in patterns)
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == TOTAL_PARAMETERS
    assert reset == EXPECTED_RESET_PARAMETERS[scope]


def test_volume_scopes_leave_every_surface_parameter_inherited(parameter_names: list[str]) -> None:
    """The per-domain modules are disjoint, which bounds the intervention."""
    for scope in ("volume_decoder", "volume_path"):
        patterns = [pattern for pattern in reset_patterns(scope) if pattern != READOUT_PATTERN]
        matched = [name for name in parameter_names if any(pattern in name for pattern in patterns)]
        assert matched, scope
        assert not [name for name in matched if "surface" in name], scope
