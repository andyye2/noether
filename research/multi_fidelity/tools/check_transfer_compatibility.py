# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Compare a ShapeNet AB-UPT checkpoint with current DrivAerML targets.

This is a construction-only audit: it instantiates CPU models but does not load
datasets, allocate CFD point clouds, or start training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal, cast

import torch

from aero_cfd.presets import DrivAerMLPreset
from noether.core.schemas.dataset import DomainDataSpec, ModelDataSpecs
from noether.modeling.models.aerodynamics import AeroABUPT

MODEL_KIND = "noether.modeling.models.aerodynamics.AeroABUPT"
CHECKPOINT_ARCHITECTURE: dict[str, Any] = {
    "hidden_dim": 192,
    "geometry_depth": 1,
    "physics_blocks": ["perceiver", "self", "cross", "self", "cross", "self", "cross", "self", "cross", "self"],
    "num_domain_decoder_blocks": {"surface": 2, "volume": 2},
    "num_heads": 3,
    "mlp_expansion_factor": 4,
    "radius": 9,
}
RESET_PATTERNS = ("backbone.domain_decoder_projections",)


class DrivAerMLCommonFieldsPreset(DrivAerMLPreset):
    """DrivAerML preset restricted to fields shared with ShapeNet-Car."""

    @property
    def data_specs(self) -> ModelDataSpecs:
        """Return pressure/velocity output specs with no auxiliary inputs."""
        return ModelDataSpecs(
            position_dim=3,
            domains={
                "surface": DomainDataSpec(output_dims={"pressure": 1}),
                "volume": DomainDataSpec(output_dims={"velocity": 3}),
            },
            use_physics_features=False,
        )


def build_target_state(task: Literal["common", "full"]) -> dict[str, torch.Tensor]:
    """Construct the current-code target model and return its state dict.

    Args:
        task: ``common`` for ShapeNet-shared fields or ``full`` for all five
            DrivAerML fields.

    Returns:
        State dict of a randomly initialized CPU AB-UPT model.
    """
    preset = DrivAerMLCommonFieldsPreset() if task == "common" else DrivAerMLPreset()
    model_config = preset.build_model(model_kind=MODEL_KIND, **CHECKPOINT_ARCHITECTURE)
    model = AeroABUPT(model_config=model_config)
    return model.state_dict()


def compatibility_report(
    checkpoint_path: Path,
    task: Literal["common", "full"],
    reset_patterns: tuple[str, ...] = RESET_PATTERNS,
) -> dict[str, Any]:
    """Partition source/target tensors into compatible and incompatible sets.

    Args:
        checkpoint_path: ShapeNet Noether model checkpoint.
        task: Target task variant.
        reset_patterns: Substrings identifying modules deliberately reinitialized.

    Returns:
        JSON-serializable compatibility report.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source = checkpoint["state_dict"]
    target = build_target_state(task)

    deliberately_reset_source = sorted(key for key in source if any(pattern in key for pattern in reset_patterns))
    deliberately_reset_target = sorted(key for key in target if any(pattern in key for pattern in reset_patterns))
    source_kept = {key: value for key, value in source.items() if key not in deliberately_reset_source}
    target_kept = {key: value for key, value in target.items() if key not in deliberately_reset_target}

    common_keys = source_kept.keys() & target_kept.keys()
    shape_mismatches = {
        key: {"source": list(source_kept[key].shape), "target": list(target_kept[key].shape)}
        for key in sorted(common_keys)
        if source_kept[key].shape != target_kept[key].shape
    }
    compatible = sorted(key for key in common_keys if key not in shape_mismatches)
    source_only = sorted(source_kept.keys() - target_kept.keys())
    target_only = sorted(target_kept.keys() - source_kept.keys())

    compatible_numel = sum(target_kept[key].numel() for key in compatible)
    target_numel = sum(value.numel() for value in target.values())
    reset_numel = sum(target[key].numel() for key in deliberately_reset_target)
    nonreset_target_numel = target_numel - reset_numel
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "task": task,
        "reset_patterns": list(reset_patterns),
        "source_tensor_count": len(source),
        "target_tensor_count": len(target),
        "compatible_tensor_count": len(compatible),
        "compatible_parameter_count": compatible_numel,
        "target_parameter_count": target_numel,
        "nonreset_target_parameter_count": nonreset_target_numel,
        "compatible_fraction_of_all_target_parameters": compatible_numel / target_numel,
        "compatible_fraction_of_nonreset_target_parameters": (
            compatible_numel / nonreset_target_numel if nonreset_target_numel else 1.0
        ),
        "source_only": source_only,
        "target_only": target_only,
        "shape_mismatches": shape_mismatches,
        "deliberately_reset_source": deliberately_reset_source,
        "deliberately_reset_target": deliberately_reset_target,
    }


def main() -> None:
    """Run compatibility checks for one or both target task variants."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--task", choices=["common", "full", "both"], default="both")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tasks: list[Literal["common", "full"]] = (
        ["common", "full"] if args.task == "both" else [cast("Literal['common', 'full']", args.task)]
    )
    report = {task: compatibility_report(args.checkpoint, task) for task in tasks}
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
