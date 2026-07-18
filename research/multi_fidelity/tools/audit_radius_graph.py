# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Audit the exact capped AB-UPT radius graph before releasing P0 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml

from noether.core.factory import DatasetFactory, Factory
from noether.core.schemas.schema import ConfigSchema
from noether.modeling.modules.encoders.supernode_pooling import SupernodePooling

DEFAULT_QUANTILES = (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_degrees(
    degrees: torch.Tensor,
    *,
    max_degree: int,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> dict[str, Any]:
    """Summarize finite nonnegative capped graph degrees."""
    flattened = degrees.detach().cpu().reshape(-1)
    if flattened.numel() == 0:
        raise ValueError("radius graph produced no query degrees")
    if flattened.dtype == torch.bool or torch.is_floating_point(flattened):
        if not torch.isfinite(flattened).all() or not torch.equal(flattened, flattened.round()):
            raise ValueError("degrees must be finite integers")
    flattened = flattened.to(torch.int64)
    if (flattened < 0).any() or (flattened > max_degree).any():
        raise ValueError(f"degrees must lie in [0, {max_degree}]")
    values = flattened.to(torch.float64)
    quantile_values = torch.quantile(values, torch.tensor(quantiles, dtype=torch.float64))
    return {
        "supernodes": int(flattened.numel()),
        "min": int(flattened.min()),
        "mean": float(values.mean()),
        "max": int(flattened.max()),
        "quantiles": {str(q): float(value) for q, value in zip(quantiles, quantile_values.tolist(), strict=True)},
        "zero_fraction": float((flattened == 0).to(torch.float64).mean()),
        "degree_cap": max_degree,
        "cap_fraction": float((flattened == max_degree).to(torch.float64).mean()),
        "histogram": torch.bincount(flattened, minlength=max_degree + 1).tolist(),
    }


def load_dry_run_config(path: Path) -> tuple[ConfigSchema, dict[str, Any]]:
    """Load the strict runner's YAML dry-run output."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("resolved_config"), dict):
        raise ValueError("input must be strict-runner dry-run YAML with resolved_config")
    audit = payload.get("audit")
    if not isinstance(audit, dict):
        raise ValueError("dry-run YAML has no audit mapping")
    config = ConfigSchema.model_validate(payload["resolved_config"])
    if set(config.datasets) != {"train", "val"}:
        raise ValueError(f"graph preflight requires train+val-only config, got {sorted(config.datasets)}")
    return config, audit


def audit_config_radius_graph(
    config: ConfigSchema,
    *,
    point_seeds: list[int],
    design_count: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run the real dataset and pipeline, then audit the model's exact capped graph."""
    if design_count < 1:
        raise ValueError("design_count must be positive")
    if not point_seeds or len(set(point_seeds)) != len(point_seeds):
        raise ValueError("point_seeds must be a non-empty unique list")
    dataset_config = config.datasets["train"]
    if dataset_config.pipeline is None:
        raise ValueError("train dataset has no pipeline")
    if config.model is None or config.model.supernode_pooling_config is None:
        raise ValueError("resolved AB-UPT config has no supernode pooling config")

    dataset = DatasetFactory().create(dataset_config)
    pool = SupernodePooling(config.model.supernode_pooling_config).to(device)
    design_count = min(design_count, len(dataset))
    all_degrees: list[torch.Tensor] = []
    cells: list[dict[str, Any]] = []
    for point_seed in point_seeds:
        pipeline_config = dataset_config.pipeline.model_copy(update={"seed": point_seed})
        pipeline = Factory().create(pipeline_config)
        for dataset_index in range(design_count):
            batch = pipeline([dataset[dataset_index]])
            positions = batch["geometry_position"].to(device)
            supernodes = batch["geometry_supernode_idx"].to(device)
            batch_indices = batch["geometry_batch_idx"].to(device)
            if not torch.isfinite(positions).all():
                raise ValueError(f"non-finite normalized geometry positions at dataset index {dataset_index}")
            _, _, local_query_indices = pool.compute_src_and_dst_indices(
                positions,
                supernodes,
                batch_indices,
            )
            degrees = torch.bincount(local_query_indices, minlength=supernodes.numel()).cpu()
            all_degrees.append(degrees)
            info = dataset.sample_info(dataset_index)
            cells.append(
                {
                    "point_seed": point_seed,
                    "dataset_index": dataset_index,
                    "design_id": info.get("design_id"),
                    **summarize_degrees(degrees, max_degree=pool.max_degree),
                }
            )
    aggregate = summarize_degrees(torch.cat(all_degrees), max_degree=pool.max_degree)
    return {
        "schema_version": 1,
        "backend_device": str(device),
        "radius": pool.radius,
        "max_degree": pool.max_degree,
        "point_seeds": point_seeds,
        "design_count_per_seed": design_count,
        "aggregate": aggregate,
        "cells": cells,
        "interpretation": (
            "Degrees are the model's capped message-graph degrees. cap_fraction measures truncation at max_degree; "
            "it is not an uncapped physical-neighbor density. Capped neighbors follow sampled source order, not kNN."
        ),
    }


def main() -> None:
    """Audit one dry-run config and fail if the preregistered zero-degree gate is exceeded."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dry_run_yaml", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--design-count", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--max-zero-fraction", type=float, default=0.001)
    args = parser.parse_args()
    if not 0.0 <= args.max_zero_fraction <= 1.0:
        parser.error("--max-zero-fraction must be in [0,1]")
    device = torch.device("cuda" if args.device == "gpu" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("--device gpu requested but CUDA is unavailable")

    try:
        config, dry_run_audit = load_dry_run_config(args.dry_run_yaml)
        graph = audit_config_radius_graph(
            config,
            point_seeds=args.point_seeds,
            design_count=args.design_count,
            device=device,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    zero_fraction = graph["aggregate"]["zero_fraction"]
    passed = zero_fraction <= args.max_zero_fraction
    result = {
        "input_dry_run_yaml": str(args.dry_run_yaml.resolve()),
        "input_dry_run_sha256": _sha256(args.dry_run_yaml),
        "dry_run_audit": dry_run_audit,
        "graph": graph,
        "gate": {
            "max_zero_fraction": args.max_zero_fraction,
            "observed_zero_fraction": zero_fraction,
            "passed": passed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result["gate"], sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
