#!/usr/bin/env python
#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "recipes/aero_cfd/src", REPO_ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from aero_cfd.presets import ShapeNetCarPreset  # noqa: E402
from noether.core.distributed.utils import accelerator_to_device  # noqa: E402
from noether.training.runners import HydraRunner  # noqa: E402

TRAINER_KIND = "noether.training.trainers.WeightedLossTrainer"
FIELD_WEIGHTS = {"surface_pressure": 1.0, "volume_velocity": 1.0}
MODEL_KIND = "noether.modeling.models.aerodynamics.AeroABUPT"


def _set_pipeline_ablation_knobs(config, *, num_volume_anchor_points: int, wake_fraction: float) -> None:
    for dataset_config in config.datasets.values():
        pipeline = dataset_config.pipeline
        if pipeline is None:
            continue
        pipeline.num_volume_anchor_points = num_volume_anchor_points
        pipeline.volume_wake_fraction = wake_fraction


def build_config(args: argparse.Namespace):
    preset = ShapeNetCarPreset()
    config = preset.build_config(
        model_kind=MODEL_KIND,
        model_params=dict(hidden_dim=192, geometry_depth=6, physics_blocks=["perceiver"] + ["shared", "cross"] * 5),
        trainer_kind=TRAINER_KIND,
        trainer_params=dict(field_weights=FIELD_WEIGHTS),
        dataset_root=args.dataset_root,
        output_path=args.output_path,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        accelerator=args.accelerator,
        seed=args.seed,
        name=args.name,
        run_id=args.run_id,
        store_code_in_output=False,
        num_workers=args.num_workers,
    )
    _set_pipeline_ablation_knobs(
        config,
        num_volume_anchor_points=args.num_volume_anchor_points,
        wake_fraction=args.volume_wake_fraction,
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ShapeNet AB-UPT with controlled wake-anchor ablation knobs.")
    parser.add_argument("--dataset-root", default="/home/feng/Projects/data/shapenet_car")
    parser.add_argument("--output-path", default="/home/feng/Projects/ABUPT/allocated-sampling/outputs")
    parser.add_argument("--accelerator", default="gpu", choices=["cpu", "gpu", "mps"])
    parser.add_argument("--num-volume-anchor-points", type=int, default=1024)
    parser.add_argument("--volume-wake-fraction", type=float, required=True)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--name", default="shapenet-car-ab-upt-wake-ablation")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = build_config(args)
    if args.dry_run:
        for key, dataset_config in config.datasets.items():
            pipeline = dataset_config.pipeline
            if pipeline is None:
                continue
            print(
                key,
                "num_volume_anchor_points=",
                pipeline.num_volume_anchor_points,
                "volume_wake_fraction=",
                pipeline.volume_wake_fraction,
                "volume_wake_axes=",
                pipeline.volume_wake_axes,
            )
        return

    HydraRunner().main(device=accelerator_to_device(args.accelerator), config=config)


if __name__ == "__main__":
    main()
