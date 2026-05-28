#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AERO_CFD_SRC = _REPO_ROOT / "recipes/aero_cfd/src"
if str(_AERO_CFD_SRC) not in sys.path:
    sys.path.insert(0, str(_AERO_CFD_SRC))

from aero_cfd.pipeline import AeroCFDPipelineConfig, AeroMultistagePipeline  # noqa: E402
from aero_cfd.pipeline.sample_processors import (  # noqa: E402
    AnchorPointSamplingSampleProcessor,
    WakeAwareAnchorPointSamplingSampleProcessor,
)
from aero_cfd.pipeline.multistage_pipelines.aero_multistage import (  # noqa: E402
    _split_by_underscore,
    _split_three_or_none,
)
from aero_cfd.utils.aero_regions import compute_wake_mask  # noqa: E402
from noether.core.schemas.dataset import DomainDataSpec, ModelDataSpecs  # noqa: E402
from noether.core.schemas.statistics import AeroStatsSchema  # noqa: E402


def _normalize(raw: torch.Tensor, raw_min: float = 0.0, raw_max: float = 20.0) -> torch.Tensor:
    return (raw - raw_min) * 1000.0 / (raw_max - raw_min)


def test_compute_wake_mask_uses_raw_coordinate_box() -> None:
    surface_raw = torch.tensor(
        [
            [0.0, -2.0, 0.0],
            [10.0, 2.0, 3.0],
        ]
    )
    volume_raw = torch.tensor(
        [
            [11.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
            [12.0, 3.0, 1.0],
            [12.0, 0.0, 4.0],
            [16.0, 0.0, 1.0],
            [14.6, 0.0, 3.0],
        ]
    )

    mask = compute_wake_mask(
        _normalize(surface_raw),
        _normalize(volume_raw),
        raw_pos_min=0.0,
        raw_pos_max=20.0,
    )

    assert mask.tolist() == [True, False, False, False, False, True]


def test_compute_wake_mask_supports_shapenet_axes() -> None:
    surface_raw = torch.tensor(
        [
            [-2.0, 0.0, 0.0],
            [2.0, 3.0, 10.0],
        ]
    )
    volume_raw = torch.tensor(
        [
            [0.0, 1.0, 11.0],
            [3.0, 1.0, 11.0],
        ]
    )

    mask = compute_wake_mask(
        _normalize(surface_raw),
        _normalize(volume_raw),
        raw_pos_min=0.0,
        raw_pos_max=20.0,
        axes=(2, 0, 1),
    )

    assert mask.tolist() == [True, False]


def _make_sampling_sample(num_wake: int = 2) -> dict[str, torch.Tensor | int]:
    surface = torch.tensor([[0.0, -2.0, 0.0], [10.0, 2.0, 3.0]])
    wake_points = torch.tensor([[11.0 + i * 0.1, 0.0, 1.0] for i in range(num_wake)])
    non_wake_points = torch.tensor([[float(i), 8.0, 1.0] for i in range(20 - num_wake)])
    volume = torch.cat([wake_points, non_wake_points], dim=0)
    return {
        "index": 0,
        "surface_position": surface,
        "volume_position": volume,
        "volume_velocity": torch.arange(volume.numel(), dtype=torch.float32).reshape_as(volume),
    }


def _make_wake_sampler(wake_fraction: float, num_points: int = 8) -> WakeAwareAnchorPointSamplingSampleProcessor:
    return WakeAwareAnchorPointSamplingSampleProcessor(
        items={"volume_position", "volume_velocity"},
        num_points=num_points,
        to_prefix_and_postfix=_split_by_underscore,
        to_prefix_midfix_postfix=_split_three_or_none,
        raw_pos_min=0.0,
        raw_pos_max=1000.0,
        wake_fraction=wake_fraction,
        seed=3,
    )


def test_wake_fraction_zero_matches_random_anchor_sampler() -> None:
    sample = _make_sampling_sample()
    random_sampler = AnchorPointSamplingSampleProcessor(
        items={"volume_position"},
        num_points=8,
        to_prefix_and_postfix=_split_by_underscore,
        to_prefix_midfix_postfix=_split_three_or_none,
        seed=3,
    )
    wake_sampler = WakeAwareAnchorPointSamplingSampleProcessor(
        items={"volume_position"},
        num_points=8,
        to_prefix_and_postfix=_split_by_underscore,
        to_prefix_midfix_postfix=_split_three_or_none,
        raw_pos_min=0.0,
        raw_pos_max=1000.0,
        wake_fraction=0.0,
        seed=3,
    )

    assert torch.equal(
        random_sampler(sample)["volume_anchor_position"],
        wake_sampler(sample)["volume_anchor_position"],
    )


def test_wake_sampler_allocates_requested_quota_without_duplicates() -> None:
    sample = _make_sampling_sample(num_wake=2)
    output = _make_wake_sampler(wake_fraction=0.25)(sample)
    mask = compute_wake_mask(
        sample["surface_position"],
        output["volume_anchor_position"],
        raw_pos_min=0.0,
        raw_pos_max=1000.0,
    )

    assert output["volume_anchor_position"].shape == (8, 3)
    assert int(mask.sum().item()) == 2
    assert torch.unique(output["volume_anchor_position"], dim=0).shape[0] == 8


def test_wake_sampler_falls_back_when_wake_points_are_sparse() -> None:
    sample = _make_sampling_sample(num_wake=1)
    output = _make_wake_sampler(wake_fraction=0.5)(sample)
    mask = compute_wake_mask(
        sample["surface_position"],
        output["volume_anchor_position"],
        raw_pos_min=0.0,
        raw_pos_max=1000.0,
    )

    assert output["volume_anchor_position"].shape == (8, 3)
    assert int(mask.sum().item()) == 1
    assert torch.unique(output["volume_anchor_position"], dim=0).shape[0] == 8


def test_aero_pipeline_keeps_shapes_with_default_and_enabled_wake_sampling() -> None:
    data_specs = ModelDataSpecs(
        position_dim=3,
        domains={
            "surface": DomainDataSpec(output_dims={"pressure": 1}),
            "volume": DomainDataSpec(output_dims={"velocity": 3}),
        },
    )
    sample = {
        "index": 0,
        "surface_position": torch.tensor(
            [[0.0, -2.0, 0.0], [2.0, -1.0, 0.5], [4.0, 1.0, 1.0], [10.0, 2.0, 3.0]]
        ),
        "surface_pressure": torch.ones(4, 1),
        "volume_position": torch.tensor(
            [[11.0, 0.0, 1.0], [12.0, 0.0, 1.0], [0.0, 8.0, 1.0], [1.0, 8.0, 1.0], [2.0, 8.0, 1.0]]
        ),
        "volume_velocity": torch.ones(5, 3),
    }

    for wake_fraction in (0.0, 0.25):
        pipeline = AeroMultistagePipeline(
            AeroCFDPipelineConfig(
                num_surface_points=4,
                num_volume_points=5,
                num_surface_queries=0,
                num_volume_queries=0,
                num_geometry_supernodes=2,
                num_geometry_points=4,
                num_volume_anchor_points=5,
                num_surface_anchor_points=4,
                volume_wake_fraction=wake_fraction,
                dataset_statistics=AeroStatsSchema(raw_pos_min=(0.0,), raw_pos_max=(1000.0,)),
                data_specs=data_specs,
            )
        )
        batch = pipeline([sample])

        assert batch["volume_anchor_position"].shape == (1, 5, 3)
        assert batch["surface_anchor_position"].shape == (1, 4, 3)
