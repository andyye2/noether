# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Strict-load a legacy ShapeNet AB-UPT trunk into current DrivAer targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import torch

from aero_cfd.presets.drivaerml import DrivAerMLPreset
from aero_cfd.presets.drivaerml_common import DrivAerMLCommonFieldsPreset
from noether.modeling.models.aerodynamics import AeroABUPT

MODEL_KIND = "noether.modeling.models.aerodynamics.AeroABUPT"
RESET_PATTERN = "backbone.domain_decoder_projections"
CHECKPOINT_ARCHITECTURE: dict[str, Any] = {
    "hidden_dim": 192,
    "geometry_depth": 1,
    "physics_blocks": [
        "perceiver",
        "self",
        "cross",
        "self",
        "cross",
        "self",
        "cross",
        "self",
        "cross",
        "self",
    ],
    "num_domain_decoder_blocks": {"surface": 2, "volume": 2},
    "num_heads": 3,
    "mlp_expansion_factor": 4,
    "radius": 9,
}


def _sha256(path: Path) -> str:
    """Hash one checkpoint file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_load(checkpoint_path: Path, task: Literal["common", "full"]) -> dict[str, Any]:
    """Perform the exact remove/instantiate/strict-load transfer operation."""
    preset = DrivAerMLCommonFieldsPreset() if task == "common" else DrivAerMLPreset()
    model_config = preset.build_model(model_kind=MODEL_KIND, **CHECKPOINT_ARCHITECTURE)
    model = AeroABUPT(model_config=model_config)
    target_state = model.state_dict()
    target_parameters = dict(model.named_parameters())
    source_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source_state = source_checkpoint["state_dict"]

    merged = {key: value for key, value in source_state.items() if RESET_PATTERN not in key}
    for key, value in target_state.items():
        if RESET_PATTERN in key:
            merged[key] = value.clone()
    incompatible = model.load_state_dict(merged, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(f"strict load unexpectedly returned {incompatible}")

    compatible_parameter_names = [
        name for name in target_parameters if name in source_state and RESET_PATTERN not in name
    ]
    compatible_parameters = sum(target_parameters[name].numel() for name in compatible_parameter_names)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    reset_parameters = sum(parameter.numel() for name, parameter in target_parameters.items() if RESET_PATTERN in name)
    total_state_elements = sum(value.numel() for value in target_state.values())
    compatible_state_elements = sum(
        target_state[key].numel() for key in target_state if key in source_state and RESET_PATTERN not in key
    )
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "task": task,
        "strict_load_succeeded": True,
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
        "reset_pattern": RESET_PATTERN,
        "compatible_trainable_parameters": compatible_parameters,
        "total_trainable_parameters": total_parameters,
        "reset_trainable_parameters": reset_parameters,
        "compatible_fraction_of_trainable_parameters": compatible_parameters / total_parameters,
        "compatible_state_elements": compatible_state_elements,
        "total_state_elements": total_state_elements,
        "state_buffer_elements": total_state_elements - total_parameters,
        "source_state_tensors": len(source_state),
        "target_state_tensors": len(target_state),
    }


def main() -> None:
    """Verify common, full, or both current targets and write JSON evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--task", choices=("common", "full", "both"), default="both")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tasks = ("common", "full") if args.task == "both" else (args.task,)
    report = {task: verify_load(args.checkpoint, task) for task in tasks}
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
