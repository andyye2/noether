#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import pytest
import torch
from torch import nn

from noether.core.schemas.dataset import DomainDataSpec, ModelDataSpecs
from noether.core.schemas.lib import resolve_config_class
from noether.core.schemas.models import DivFreeAnchorBranchedUPTConfig, ModelBaseConfig
from noether.modeling.models import DivFreeAeroABUPT
from noether.modeling.models.divfree_aerodynamics import DivFreeAeroABUPT as DirectDivFreeAeroABUPT
from noether.modeling.functional import central_difference_positions


class FakePotentialBackbone(nn.Module):
    """Backbone stub that returns analytic vector-potential values."""

    def __init__(self) -> None:
        super().__init__()
        self.volume_query_position: torch.Tensor | None = None

    def forward(
        self,
        *,
        domain_anchor_positions: dict[str, torch.Tensor],
        domain_query_positions: dict[str, torch.Tensor],
        **_,
    ) -> tuple[dict[str, torch.Tensor], dict]:
        query_position = domain_query_positions["volume"]
        self.volume_query_position = query_position

        potential = torch.zeros_like(query_position)
        potential[..., 2] = query_position[..., 0] * query_position[..., 1]

        return (
            {
                "volume_velocity": torch.full_like(domain_anchor_positions["volume"], -123.0),
                "query_volume_velocity": potential,
            },
            {},
        )


def _expected_normalized_velocity(
    positions: torch.Tensor,
    position_scale: float | list[float] = 1.0,
) -> torch.Tensor:
    scale = torch.as_tensor(position_scale, device=positions.device, dtype=positions.dtype)
    if scale.numel() == 1:
        scale = scale.expand(3)

    velocity = torch.stack(
        [
            positions[..., 0] / scale[1],
            -positions[..., 1] / scale[0],
            torch.zeros_like(positions[..., 0]),
        ],
        dim=-1,
    )
    mean = positions.new_tensor([1.0, 2.0, 3.0])
    std = positions.new_tensor([4.0, 5.0, 6.0])
    return (velocity - mean) / std


def _config(
    *,
    data_specs: ModelDataSpecs | None = None,
    position_scale: float | list[float] = 0.0105,
    zero_init_volume_projection: bool = True,
) -> DivFreeAnchorBranchedUPTConfig:
    return DivFreeAnchorBranchedUPTConfig(
        kind="noether.modeling.models.DivFreeAeroABUPT",
        name="test_divfree_ab_upt",
        hidden_dim=12,
        geometry_depth=0,
        physics_blocks=["self"],
        num_domain_decoder_blocks={"surface": 0, "volume": 0},
        transformer_block_config={
            "hidden_dim": 12,
            "num_heads": 2,
            "mlp_expansion_factor": 2,
            "use_rope": True,
        },
        data_specs=data_specs
        or ModelDataSpecs(
            position_dim=3,
            domains={
                "surface": DomainDataSpec(output_dims={"pressure": 1}),
                "volume": DomainDataSpec(output_dims={"velocity": 3}),
            },
        ),
        delta=1e-3,
        position_scale=position_scale,
        volume_velocity_mean=[1.0, 2.0, 3.0],
        volume_velocity_std=[4.0, 5.0, 6.0],
        zero_init_volume_projection=zero_init_volume_projection,
    )


def test_divfree_aero_abupt_public_export():
    assert DivFreeAeroABUPT is DirectDivFreeAeroABUPT


@pytest.mark.parametrize(
    "kind",
    [
        "noether.modeling.models.DivFreeAeroABUPT",
        "noether.modeling.models.divfree_aerodynamics.DivFreeAeroABUPT",
    ],
)
def test_divfree_aero_abupt_resolves_config_class(kind):
    config_class = resolve_config_class(kind, ModelBaseConfig)

    assert config_class is DivFreeAnchorBranchedUPTConfig


def test_divfree_aero_abupt_registers_divfree_buffers():
    model = DivFreeAeroABUPT(model_config=_config(position_scale=[0.1, 0.2, 0.3]))

    assert torch.equal(model.position_scale, torch.tensor([0.1, 0.2, 0.3]))
    assert torch.equal(model.volume_velocity_mean, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(model.volume_velocity_std, torch.tensor([4.0, 5.0, 6.0]))


def test_divfree_aero_abupt_expands_scalar_position_scale():
    model = DivFreeAeroABUPT(model_config=_config(position_scale=0.0105))

    assert torch.equal(model.position_scale, torch.full((3,), 0.0105))


def test_divfree_aero_abupt_zero_initializes_volume_projection():
    model = DivFreeAeroABUPT(model_config=_config(zero_init_volume_projection=True))
    projection = model.backbone.domain_decoder_projections["volume"].linear.project

    assert isinstance(projection, nn.Linear)
    assert torch.equal(projection.weight, torch.zeros_like(projection.weight))
    assert projection.bias is not None
    assert torch.equal(projection.bias, torch.zeros_like(projection.bias))


def test_divfree_aero_abupt_requires_volume_velocity():
    data_specs = ModelDataSpecs(
        position_dim=3,
        domains={
            "surface": DomainDataSpec(output_dims={"pressure": 1}),
            "volume": DomainDataSpec(output_dims={"pressure": 1}),
        },
    )

    with pytest.raises(ValueError, match="velocity: 3"):
        DivFreeAeroABUPT(model_config=_config(data_specs=data_specs))


def test_divfree_aero_abupt_forward_replaces_anchor_velocity():
    model = DivFreeAeroABUPT(model_config=_config(position_scale=1.0, zero_init_volume_projection=False))
    model.backbone = FakePotentialBackbone()

    volume_anchor_position = torch.tensor([[[2.0, 3.0, 0.0], [-1.0, 4.0, 0.0]]])

    output = model(volume_anchor_position=volume_anchor_position)

    assert torch.allclose(
        output["volume_velocity"],
        _expected_normalized_velocity(volume_anchor_position),
        atol=5e-4,
    )
    assert "query_volume_velocity" not in output
    assert model.backbone.volume_query_position is not None
    assert model.backbone.volume_query_position.shape == (1, 12, 3)


def test_divfree_aero_abupt_forward_applies_position_scale():
    model = DivFreeAeroABUPT(
        model_config=_config(
            position_scale=[2.0, 4.0, 8.0],
            zero_init_volume_projection=False,
        )
    )
    model.backbone = FakePotentialBackbone()

    volume_anchor_position = torch.tensor([[[2.0, 3.0, 0.0]]])

    output = model(volume_anchor_position=volume_anchor_position)

    assert torch.allclose(
        output["volume_velocity"],
        _expected_normalized_velocity(volume_anchor_position, position_scale=[2.0, 4.0, 8.0]),
        atol=5e-4,
    )


def test_divfree_aero_abupt_forward_handles_user_volume_queries():
    model = DivFreeAeroABUPT(model_config=_config(position_scale=1.0, zero_init_volume_projection=False))
    model.backbone = FakePotentialBackbone()

    volume_anchor_position = torch.tensor([[[2.0, 3.0, 0.0]]])
    query_volume_position = torch.tensor([[[5.0, 7.0, 0.0], [-2.0, 6.0, 0.0]]])

    output = model(
        volume_anchor_position=volume_anchor_position,
        query_volume_position=query_volume_position,
    )

    assert torch.allclose(
        output["volume_velocity"],
        _expected_normalized_velocity(volume_anchor_position),
        atol=5e-4,
    )
    assert torch.allclose(
        output["query_volume_velocity"],
        _expected_normalized_velocity(query_volume_position),
        atol=5e-4,
    )

    expected_query_position = torch.cat(
        [
            central_difference_positions(volume_anchor_position, delta=1e-3),
            central_difference_positions(query_volume_position, delta=1e-3),
        ],
        dim=1,
    )
    assert model.backbone.volume_query_position is not None
    assert torch.equal(model.backbone.volume_query_position, expected_query_position)


def test_divfree_aero_abupt_forward_smoke_with_real_backbone():
    torch.manual_seed(0)
    model = DivFreeAeroABUPT(model_config=_config(position_scale=1.0, zero_init_volume_projection=False))

    surface_anchor_position = torch.randn(2, 4, 3)
    volume_anchor_position = torch.randn(2, 3, 3)
    query_volume_position = torch.randn(2, 2, 3)

    output = model(
        surface_anchor_position=surface_anchor_position,
        volume_anchor_position=volume_anchor_position,
        query_volume_position=query_volume_position,
    )

    assert output["surface_pressure"].shape == (2, 4, 1)
    assert output["volume_velocity"].shape == (2, 3, 3)
    assert output["query_volume_velocity"].shape == (2, 2, 3)
    assert torch.isfinite(output["surface_pressure"]).all()
    assert torch.isfinite(output["volume_velocity"]).all()
    assert torch.isfinite(output["query_volume_velocity"]).all()
