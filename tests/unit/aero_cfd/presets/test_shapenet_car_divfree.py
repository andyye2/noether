#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys

import pytest

from noether.core.schemas.models.divfree_ab_upt import DivFreeAnchorBranchedUPTConfig

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RECIPE_SRC = _REPO_ROOT / "recipes" / "aero_cfd" / "src"
if str(_RECIPE_SRC) not in sys.path:
    sys.path.insert(0, str(_RECIPE_SRC))

_DIVFREE_KIND = "noether.modeling.models.divfree_aerodynamics.DivFreeAeroABUPT"
_ABUPT_KIND = "noether.modeling.models.aerodynamics.AeroABUPT"


def test_pipeline_override_matches_abupt() -> None:
    presets = import_module("aero_cfd.presets")

    overrides = presets.ShapeNetCarDivFreePreset.pipeline_model_overrides

    assert _DIVFREE_KIND in overrides
    assert overrides[_DIVFREE_KIND] == presets.ShapeNetCarPreset.pipeline_model_overrides[_ABUPT_KIND]
    assert overrides[_DIVFREE_KIND] is not presets.ShapeNetCarPreset.pipeline_model_overrides[_ABUPT_KIND]


def test_forward_properties_match_abupt() -> None:
    presets = import_module("aero_cfd.presets")
    preset = presets.ShapeNetCarDivFreePreset()

    assert preset.forward_properties(_DIVFREE_KIND) == preset.forward_properties(_ABUPT_KIND)


def test_build_model_injects_divfree_statistics() -> None:
    presets = import_module("aero_cfd.presets")
    preset = presets.ShapeNetCarDivFreePreset()

    config = preset.build_model(
        model_kind=_DIVFREE_KIND,
        hidden_dim=96,
        geometry_depth=2,
        physics_blocks=["perceiver", "self"],
    )

    assert isinstance(config, DivFreeAnchorBranchedUPTConfig)
    assert config.name == "divfree_ab_upt"
    assert config.position_scale == pytest.approx(0.0105)
    assert config.volume_velocity_mean == pytest.approx([0.00293915, -0.0230546, 17.546032])
    assert config.volume_velocity_std == pytest.approx([1.361689, 1.267649, 5.850353])
    assert config.delta == pytest.approx(1e-3)


def test_build_model_preserves_user_overrides() -> None:
    presets = import_module("aero_cfd.presets")
    preset = presets.ShapeNetCarDivFreePreset()

    config = preset.build_model(
        model_kind=_DIVFREE_KIND,
        name="custom",
        position_scale=0.5,
        hidden_dim=96,
        geometry_depth=2,
        physics_blocks=["perceiver", "self"],
    )

    assert config.name == "custom"
    assert config.position_scale == pytest.approx(0.5)


def test_data_specs_keep_volume_velocity_vector_output() -> None:
    presets = import_module("aero_cfd.presets")
    preset = presets.ShapeNetCarDivFreePreset()

    output_dims = preset.data_specs.domains["volume"].output_dims

    assert dict(output_dims.items()) == {"velocity": 3}


def test_evaluation_callbacks_include_divergence_monitor() -> None:
    presets = import_module("aero_cfd.presets")
    preset = presets.ShapeNetCarDivFreePreset()

    callbacks = preset.evaluation_callbacks(_DIVFREE_KIND)
    kinds = [callback.kind for callback in callbacks]

    assert kinds.count("aero_cfd.callbacks.DivFreeMetricsCallback") == 1

    divfree_callback = next(
        callback for callback in callbacks if callback.kind == "aero_cfd.callbacks.DivFreeMetricsCallback"
    )
    assert divfree_callback.dataset_key == "test"
    assert divfree_callback.delta is None
    assert divfree_callback.position_scale is None
    assert divfree_callback.num_monitor_anchors == 256
    assert divfree_callback.max_samples == 4
    assert "volume_anchor_position" in divfree_callback.forward_properties


def test_evaluation_callbacks_skip_divergence_monitor_for_baseline_model() -> None:
    presets = import_module("aero_cfd.presets")
    preset = presets.ShapeNetCarDivFreePreset()

    callbacks = preset.evaluation_callbacks(_ABUPT_KIND)
    kinds = [callback.kind for callback in callbacks]

    assert "aero_cfd.callbacks.DivFreeMetricsCallback" not in kinds
