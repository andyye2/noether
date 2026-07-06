#  Copyright (c) 2025 Emmi AI GmbH. All rights reserved.

import torch

from aero_cfd.pipeline.multistage_pipelines.aero_multistage import AeroCFDPipelineConfig, AeroMultistagePipeline
from aero_cfd.pipeline.sample_processors import WallNoPenetrationQuerySampleProcessor
from aero_cfd.trainers.aerodynamics_cfd import AerodynamicsCFDTrainer
from noether.core.schemas.dataset import DomainDataSpec, ModelDataSpecs


def test_wall_no_penetration_query_processor_offsets_surface_anchors():
    processor = WallNoPenetrationQuerySampleProcessor(offset=0.5)
    sample = {
        "surface_anchor_position": torch.tensor([[1.0, 2.0, 3.0]]),
        "surface_anchor_normals": torch.tensor([[0.0, 0.0, 2.0]]),
    }

    output = processor(sample)

    assert torch.allclose(output["wall_query_normals"], torch.tensor([[0.0, 0.0, 1.0]]))
    assert torch.allclose(output["query_volume_position"], torch.tensor([[1.0, 2.0, 3.5]]))


def test_anchor_pipeline_adds_wall_query_batch_items():
    data_specs = ModelDataSpecs(
        position_dim=3,
        domains={
            "surface": DomainDataSpec(output_dims={"pressure": 1}),
            "volume": DomainDataSpec(output_dims={"velocity": 3}),
        },
    )
    pipeline = AeroMultistagePipeline(
        AeroCFDPipelineConfig(
            num_surface_points=0,
            num_volume_points=0,
            num_surface_queries=0,
            num_volume_queries=0,
            num_geometry_points=4,
            num_geometry_supernodes=2,
            num_surface_anchor_points=2,
            num_volume_anchor_points=2,
            use_wall_no_penetration_queries=True,
            wall_query_offset=0.25,
            data_specs=data_specs,
        )
    )
    sample = {
        "surface_position": torch.arange(12, dtype=torch.float32).view(4, 3),
        "surface_pressure": torch.ones(4, 1),
        "surface_normals": torch.tensor([[0.0, 0.0, 1.0]]).repeat(4, 1),
        "volume_position": torch.arange(12, 24, dtype=torch.float32).view(4, 3),
        "volume_velocity": torch.ones(4, 3),
    }

    batch = pipeline([sample])

    assert batch["query_volume_position"].shape == (1, 2, 3)
    assert torch.allclose(batch["wall_query_normals"], torch.tensor([[[0.0, 0.0, 1.0]]]).repeat(1, 2, 1))
    assert torch.allclose(
        batch["query_volume_position"] - batch["surface_anchor_position"],
        0.25 * batch["wall_query_normals"],
    )


def test_wall_no_penetration_loss_uses_denormalized_velocity():
    trainer = AerodynamicsCFDTrainer.__new__(AerodynamicsCFDTrainer)
    trainer.loss_items = []
    trainer.surface_weight = 1.0
    trainer.wall_no_penetration_weight = 1.0
    trainer.wall_no_penetration_velocity_scale = 3.0
    trainer.wall_query_normal_key = "wall_query_normals"
    trainer.volume_velocity_mean = (1.0, 0.0, 0.0)
    trainer.volume_velocity_std = (2.0, 1.0, 1.0)

    losses = trainer.loss_compute(
        forward_output={"query_volume_velocity": torch.tensor([[[1.0, 0.0, 0.0]]])},
        targets={"wall_query_normals": torch.tensor([[[2.0, 0.0, 0.0]]])},
    )

    assert torch.allclose(losses["wall_no_penetration_loss"], torch.tensor(1.0))
