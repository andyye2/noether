#!/usr/bin/env python
#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "recipes/aero_cfd/src", REPO_ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from aero_cfd.utils.aero_regions import compute_wake_mask, denormalize_position  # noqa: E402
from noether.core.factory import DatasetFactory, Factory  # noqa: E402
from noether.core.schemas.dataset import StandardDatasetConfig  # noqa: E402
from noether.core.schemas.models import AnchorBranchedUPTConfig  # noqa: E402
from noether.modeling.models.aerodynamics import AeroABUPT  # noqa: E402

DEFAULT_CHECKPOINT = (
    "/home/feng/Projects/ABUPT/outputs/2026-04-25_7d0mv/train/checkpoints/"
    "ab_upt_cp=best_model.loss.test.total_model.th"
)
DEFAULT_HP_RESOLVED = "/home/feng/Projects/ABUPT/outputs/2026-04-25_7d0mv/train/hp_resolved.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results"


@dataclass
class RegionStats:
    point_count: int = 0
    sq_error_sum: float = 0.0
    abs_error_sum: float = 0.0
    delta_l2_sum: float = 0.0
    target_l2_sum: float = 0.0

    def update(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        mask = mask.bool()
        count = int(mask.sum().item())
        if count == 0:
            return

        delta = prediction[mask] - target[mask]
        self.point_count += count
        self.sq_error_sum += float((delta.square().mean(dim=-1)).sum().item())
        self.abs_error_sum += float((delta.abs().mean(dim=-1)).sum().item())
        self.delta_l2_sum += float(delta.square().sum().item())
        self.target_l2_sum += float(target[mask].square().sum().item())

    def row(self, *, total_points: int, global_mse: float) -> dict[str, float | int]:
        mse = self.sq_error_sum / self.point_count if self.point_count else math.nan
        mae = self.abs_error_sum / self.point_count if self.point_count else math.nan
        rel_l2 = math.sqrt(self.delta_l2_sum / self.target_l2_sum) if self.target_l2_sum > 0.0 else math.nan
        return {
            "point_count": self.point_count,
            "point_fraction": self.point_count / total_points if total_points else math.nan,
            "velocity_mse": mse,
            "velocity_mae": mae,
            "relative_l2": rel_l2,
            "mse_over_global": mse / global_mse if global_mse > 0.0 else math.nan,
        }


@dataclass
class RegionAccumulator:
    regions: dict[str, RegionStats] = field(default_factory=dict)

    def update(self, name: str, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        self.regions.setdefault(name, RegionStats()).update(prediction, target, mask)

    def as_rows(self) -> list[dict[str, float | int | str]]:
        total_points = self.regions["global"].point_count
        global_row = self.regions["global"].row(total_points=total_points, global_mse=math.nan)
        global_mse = float(global_row["velocity_mse"])

        ordered_names = ["global", "wake", "non_wake"]
        ordered_names.extend(name for name in self.regions if name.startswith("near_wall_sdf_"))
        rows = []
        for name in ordered_names:
            stats = self.regions.get(name, RegionStats())
            row = stats.row(total_points=total_points, global_mse=global_mse)
            row["region"] = name
            rows.append(row)
        return rows


def _parse_axes(value: str) -> tuple[int, int, int]:
    axes = tuple(int(part.strip()) for part in value.split(","))
    if len(axes) != 3:
        raise argparse.ArgumentTypeError("axes must be comma-separated as streamwise,spanwise,vertical.")
    return axes  # type: ignore[return-value]


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _load_dataset_config(
    hp_resolved: Path,
    split: str,
    *,
    eval_num_volume_anchor_points: int | None,
) -> StandardDatasetConfig:
    with hp_resolved.open() as handle:
        hp = yaml.safe_load(handle)
    dataset_dict = hp["datasets"][split]
    dataset_dict["pipeline"]["kind"] = "aero_cfd.pipeline.AeroMultistagePipeline"
    if eval_num_volume_anchor_points is not None:
        dataset_dict["pipeline"]["num_volume_anchor_points"] = eval_num_volume_anchor_points
    return StandardDatasetConfig(**dataset_dict)


def _migrate_legacy_readout_state_dict(
    state_dict: OrderedDict[str, torch.Tensor],
    model_state_dict: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    migrated: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        if ".domain_decoder_projections." in key and key.endswith(".project.weight"):
            key = key.replace(".project.weight", ".linear.project.weight")
        elif ".domain_decoder_projections." in key and key.endswith(".project.bias"):
            key = key.replace(".project.bias", ".linear.project.bias")
        migrated[key] = value

    for key, value in model_state_dict.items():
        if key in migrated:
            continue
        if ".domain_decoder_projections." in key and ".norm_final." in key:
            migrated[key] = value

    missing = set(model_state_dict) - set(migrated)
    unexpected = set(migrated) - set(model_state_dict)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint could not be migrated safely. "
            f"Missing keys: {sorted(missing)}. Unexpected keys: {sorted(unexpected)}."
        )
    return migrated


def _load_model(checkpoint_path: Path, device: torch.device) -> AeroABUPT:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_config = AnchorBranchedUPTConfig(**checkpoint["model_config"])
    model = AeroABUPT(model_config).to(device)

    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except RuntimeError:
        state_dict = _migrate_legacy_readout_state_dict(checkpoint["state_dict"], model.state_dict())
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    return model


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _matched_anchor_indices(anchor_position: torch.Tensor, source_position: torch.Tensor) -> torch.Tensor:
    index_by_position = {tuple(row.tolist()): idx for idx, row in enumerate(source_position)}
    exact_indices = [index_by_position.get(tuple(row.tolist())) for row in anchor_position]
    if all(idx is not None for idx in exact_indices):
        return torch.tensor(exact_indices, dtype=torch.long)

    distances = torch.cdist(anchor_position.float(), source_position.float())
    return distances.argmin(dim=1)


def _body_length(
    surface_position: torch.Tensor,
    *,
    raw_pos_min: torch.Tensor,
    raw_pos_max: torch.Tensor,
    axes: tuple[int, int, int],
) -> float:
    surface_raw = denormalize_position(
        surface_position,
        raw_pos_min=raw_pos_min,
        raw_pos_max=raw_pos_max,
    )
    stream_axis = axes[0]
    length = surface_raw[:, stream_axis].amax() - surface_raw[:, stream_axis].amin()
    return float(length.item())


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fieldnames = ["region", "point_count", "point_fraction", "velocity_mse", "velocity_mae", "relative_l2", "mse_over_global"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.6g}"


def _write_markdown(path: Path, rows: list[dict[str, float | int | str]], *, checkpoint: Path, split: str) -> None:
    columns = ["region", "point_count", "point_fraction", "velocity_mse", "velocity_mae", "relative_l2", "mse_over_global"]
    lines = [
        "# Baseline Region Error",
        "",
        f"- checkpoint: `{checkpoint}`",
        f"- split: `{split}`",
        "",
        "|" + "|".join(columns) + "|",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join(_fmt(row[column]) for column in columns) + "|")
    path.write_text("\n".join(lines) + "\n")


def evaluate(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(device_name)
    dataset_config = _load_dataset_config(
        args.hp_resolved,
        args.split,
        eval_num_volume_anchor_points=args.eval_num_volume_anchor_points,
    )
    dataset = DatasetFactory().create(dataset_config)
    pipeline = Factory().create(dataset_config.pipeline)
    model = _load_model(args.checkpoint, device)

    raw_pos_min = torch.as_tensor(dataset.fetch_statistics()["raw_pos_min"])
    raw_pos_max = torch.as_tensor(dataset.fetch_statistics()["raw_pos_max"])

    max_samples = args.max_samples if args.max_samples is not None else len(dataset)
    num_samples = min(max_samples, len(dataset))
    accumulator = RegionAccumulator()

    for sample_idx in range(num_samples):
        sample = dataset[sample_idx]
        batch = pipeline([sample])
        device_batch = _to_device(batch, device)
        forward_inputs = {key: device_batch[key] for key in args.forward_properties}

        with torch.no_grad():
            outputs = model(**forward_inputs)

        pred_velocity = dataset.denormalize("volume_velocity", outputs["volume_velocity"].cpu()).squeeze(0)
        target_velocity = dataset.denormalize("volume_velocity", batch["volume_velocity_target"].cpu()).squeeze(0)
        anchor_position = batch["volume_anchor_position"].squeeze(0).cpu()

        wake_mask = compute_wake_mask(
            surface_position=sample["surface_position"].cpu(),
            volume_position=anchor_position,
            raw_pos_min=raw_pos_min,
            raw_pos_max=raw_pos_max,
            box_lwh=args.wake_box_lwh,
            axes=args.wake_axes,
        )
        source_indices = _matched_anchor_indices(anchor_position, sample["volume_position"].cpu())
        sample_uri = Path(dataset.sample_info(sample_idx)["sample_uri"])
        raw_sdf = torch.load(sample_uri / "volume_sdf.pt", map_location="cpu", weights_only=True).flatten()
        anchor_sdf = raw_sdf[source_indices].abs()
        length = _body_length(
            sample["surface_position"].cpu(),
            raw_pos_min=raw_pos_min,
            raw_pos_max=raw_pos_max,
            axes=args.wake_axes,
        )

        global_mask = torch.ones(anchor_position.shape[0], dtype=torch.bool)
        accumulator.update("global", pred_velocity, target_velocity, global_mask)
        accumulator.update("wake", pred_velocity, target_velocity, wake_mask)
        accumulator.update("non_wake", pred_velocity, target_velocity, ~wake_mask)
        for fraction in args.near_wall_fractions:
            name = f"near_wall_sdf_{fraction:g}L"
            accumulator.update(name, pred_velocity, target_velocity, anchor_sdf < fraction * length)

    return accumulator.as_rows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ShapeNet AB-UPT velocity error by aerodynamic regions.")
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--hp-resolved", type=Path, default=Path(DEFAULT_HP_RESOLVED))
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--eval-num-volume-anchor-points", type=int, default=None)
    parser.add_argument("--wake-axes", type=_parse_axes, default=(2, 0, 1))
    parser.add_argument("--wake-box-lwh", type=_parse_floats, default=(0.47, 0.43, 0.31))
    parser.add_argument("--near-wall-fractions", type=_parse_floats, default=(0.005, 0.01, 0.02))
    parser.add_argument(
        "--forward-properties",
        nargs="+",
        default=[
            "geometry_position",
            "geometry_supernode_idx",
            "geometry_batch_idx",
            "surface_anchor_position",
            "volume_anchor_position",
        ],
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = evaluate(args)
    _write_csv(args.output_dir / "baseline_region_error.csv", rows)
    _write_markdown(
        args.output_dir / "baseline_region_error.md",
        rows,
        checkpoint=args.checkpoint,
        split=args.split,
    )
    print(f"Wrote {args.output_dir / 'baseline_region_error.md'}")
    print(f"Wrote {args.output_dir / 'baseline_region_error.csv'}")


if __name__ == "__main__":
    main()
