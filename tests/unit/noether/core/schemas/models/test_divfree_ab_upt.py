#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from copy import deepcopy

import pytest
from pydantic import ValidationError

from noether.core.schemas.models import DivFreeAnchorBranchedUPTConfig
from noether.core.schemas.models.divfree_ab_upt import (
    DivFreeAnchorBranchedUPTConfig as DirectDivFreeAnchorBranchedUPTConfig,
)


def _divfree_config_dict(base_ab_upt_config_dict):
    config = deepcopy(base_ab_upt_config_dict)
    config.update(
        {
            "delta": 1e-3,
            "position_scale": 0.0105,
            "volume_velocity_mean": [0.00293915, -0.0230546, 17.546032],
            "volume_velocity_std": [1.361689, 1.267649, 5.850353],
            "zero_init_volume_projection": True,
        }
    )
    return config


def test_divfree_ab_upt_config_constructs(base_ab_upt_config_dict):
    config = DivFreeAnchorBranchedUPTConfig.model_validate(_divfree_config_dict(base_ab_upt_config_dict))

    assert config.delta == 1e-3
    assert config.position_scale == 0.0105
    assert config.volume_velocity_mean == [0.00293915, -0.0230546, 17.546032]
    assert config.volume_velocity_std == [1.361689, 1.267649, 5.850353]
    assert config.zero_init_volume_projection is True


def test_divfree_ab_upt_config_public_export():
    assert DivFreeAnchorBranchedUPTConfig is DirectDivFreeAnchorBranchedUPTConfig


def test_divfree_ab_upt_config_inherits_ab_upt_fields(base_ab_upt_config_dict):
    config = DivFreeAnchorBranchedUPTConfig.model_validate(_divfree_config_dict(base_ab_upt_config_dict))

    assert config.hidden_dim == 128
    assert config.supernode_pooling_config.hidden_dim == 128
    assert config.transformer_block_config.hidden_dim == 128
    assert config.transformer_block_config.num_heads == 4


@pytest.mark.parametrize("missing_field", ["position_scale", "volume_velocity_mean", "volume_velocity_std"])
def test_divfree_ab_upt_config_requires_divfree_fields(base_ab_upt_config_dict, missing_field):
    config = _divfree_config_dict(base_ab_upt_config_dict)
    config.pop(missing_field)

    with pytest.raises(ValidationError, match=missing_field):
        DivFreeAnchorBranchedUPTConfig.model_validate(config)


def test_divfree_ab_upt_config_accepts_per_axis_position_scale(base_ab_upt_config_dict):
    config = _divfree_config_dict(base_ab_upt_config_dict)
    config["position_scale"] = [0.0105, 0.011, 0.012]

    validated = DivFreeAnchorBranchedUPTConfig.model_validate(config)

    assert validated.position_scale == [0.0105, 0.011, 0.012]


@pytest.mark.parametrize("position_scale", [[0.0105, 0.011], [0.0105, 0.011, 0.012, 0.013], 0.0])
def test_divfree_ab_upt_config_rejects_invalid_position_scale(base_ab_upt_config_dict, position_scale):
    config = _divfree_config_dict(base_ab_upt_config_dict)
    config["position_scale"] = position_scale

    with pytest.raises(ValidationError, match="position_scale"):
        DivFreeAnchorBranchedUPTConfig.model_validate(config)


@pytest.mark.parametrize("field_name", ["volume_velocity_mean", "volume_velocity_std"])
def test_divfree_ab_upt_config_rejects_invalid_velocity_stats_shape(base_ab_upt_config_dict, field_name):
    config = _divfree_config_dict(base_ab_upt_config_dict)
    config[field_name] = [0.0, 1.0]

    with pytest.raises(ValidationError, match=field_name):
        DivFreeAnchorBranchedUPTConfig.model_validate(config)


def test_divfree_ab_upt_config_rejects_nonpositive_velocity_std(base_ab_upt_config_dict):
    config = _divfree_config_dict(base_ab_upt_config_dict)
    config["volume_velocity_std"] = [1.0, 0.0, 1.0]

    with pytest.raises(ValidationError, match="volume_velocity_std"):
        DivFreeAnchorBranchedUPTConfig.model_validate(config)
