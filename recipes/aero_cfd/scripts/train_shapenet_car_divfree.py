#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from aero_cfd.presets import ShapeNetCarDivFreePreset
from noether.core.distributed.utils import accelerator_to_device
from noether.training.runners import HydraRunner

TRAINER_KIND = "noether.training.trainers.WeightedLossTrainer"
FIELD_WEIGHTS = {"surface_pressure": 1.0, "volume_velocity": 1.0}
MODEL_KIND = "noether.modeling.models.divfree_aerodynamics.DivFreeAeroABUPT"


def train_divfree(
    *,
    dataset_root: str,
    output_path: str,
    accelerator: str = "gpu",
) -> None:
    """Trains DivFree AB-UPT model using ShapeNetCar dataset."""
    preset = ShapeNetCarDivFreePreset()
    config = preset.build_config(
        model_kind=MODEL_KIND,
        model_params=dict(hidden_dim=192, geometry_depth=6, physics_blocks=["perceiver"] + ["shared", "cross"] * 5),
        trainer_kind=TRAINER_KIND,
        trainer_params=dict(field_weights=FIELD_WEIGHTS, precision="float32"),
        dataset_root=dataset_root,
        output_path=output_path,
        max_epochs=2,
        accelerator=accelerator,
    )
    HydraRunner().main(device=accelerator_to_device(accelerator), config=config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DivFree AB-UPT on ShapeNet Car dataset.")
    parser.add_argument("--dataset-root", required=True, help="Path to the ShapeNet Car dataset.")
    parser.add_argument("--output-path", required=True, help="Path to store training outputs.")
    parser.add_argument("--accelerator", default="gpu", choices=["cpu", "gpu", "mps"], help="Accelerator to use.")
    args = parser.parse_args()

    train_divfree(dataset_root=args.dataset_root, output_path=args.output_path, accelerator=args.accelerator)
