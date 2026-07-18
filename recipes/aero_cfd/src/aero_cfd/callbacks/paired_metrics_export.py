# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Export per-design physical-field metrics for paired transfer analysis."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import torch
from pydantic import Field

from .aero_metrics import AeroMetricsCallback, AeroMetricsCallbackConfig


class PairedMetricsExportCallbackConfig(AeroMetricsCallbackConfig):
    """Configuration for one frozen, per-design result export."""

    kind: str | None = "aero_cfd.callbacks.paired_metrics_export.PairedMetricsExportCallback"
    output_csv: str
    method: str = Field(min_length=1)
    replicate: str = Field(min_length=1)
    train_sample_size: int = Field(gt=0)


class PairedMetricsExportCallback(AeroMetricsCallback):
    """Compute AeroMetrics and retain the official run ID for every test design."""

    def __init__(self, callback_config: PairedMetricsExportCallbackConfig, **kwargs: Any) -> None:
        super().__init__(callback_config=callback_config, **kwargs)
        self.output_csv = Path(callback_config.output_csv)
        self.method = callback_config.method
        self.replicate = callback_config.replicate
        self.train_sample_size = callback_config.train_sample_size

    def process_data(self, batch: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor]:
        """Attach the official DrivAer design ID to one sample's metrics."""
        metrics = super().process_data(batch, **kwargs)
        if "index" not in batch:
            raise KeyError("paired metric export requires the pipeline's index property")
        index = int(batch["index"].reshape(-1)[0].item())
        dataset = self.data_container.get_dataset(self.dataset_key)
        sample_info = dataset.sample_info(index)
        design_id = sample_info.get("design_id")
        if design_id is None:
            raise ValueError(f"dataset sample {index} has no design_id")
        return {"design_id": torch.tensor(int(design_id), device=batch["index"].device), **metrics}

    def process_results(self, results: dict[str, torch.Tensor], **kwargs: Any) -> None:
        """Atomically write the long table consumed by the analysis tool."""
        super().process_results(
            {key: value for key, value in results.items() if key != "design_id"},
            **kwargs,
        )
        design_ids = results.get("design_id")
        if design_ids is None:
            raise ValueError("collated evaluation results contain no design IDs")
        design_ids = design_ids.detach().cpu().reshape(-1)
        if len({int(value) for value in design_ids.tolist()}) != design_ids.numel():
            raise ValueError("test design IDs are not unique")

        field_metrics: dict[str, dict[str, torch.Tensor]] = {}
        for key, values in results.items():
            if key == "design_id":
                continue
            if key.endswith("_l2err"):
                field = key.removesuffix("_l2err")
                field_metrics.setdefault(field, {})["relative_l2"] = values.detach().cpu().reshape(-1)
            elif key.endswith("_mae"):
                field = key.removesuffix("_mae")
                field_metrics.setdefault(field, {})["mae"] = values.detach().cpu().reshape(-1)

        rows: list[dict[str, str | int | float]] = []
        for field, metrics in sorted(field_metrics.items()):
            relative_l2 = metrics.get("relative_l2")
            mae = metrics.get("mae")
            if relative_l2 is None or mae is None:
                raise ValueError(f"field {field!r} does not have both relative-L2 and MAE")
            if relative_l2.numel() != design_ids.numel() or mae.numel() != design_ids.numel():
                raise ValueError(f"field {field!r} metric count does not match design IDs")
            if not torch.isfinite(relative_l2).all() or not torch.isfinite(mae).all():
                raise ValueError(f"field {field!r} contains non-finite metrics")
            for index, design_id in enumerate(design_ids.tolist()):
                rows.append(
                    {
                        "method": self.method,
                        "replicate": self.replicate,
                        "n": self.train_sample_size,
                        "design_id": int(design_id),
                        "field": field,
                        "relative_l2": float(relative_l2[index]),
                        "mae": float(mae[index]),
                    }
                )

        if not rows:
            raise ValueError("no per-field metrics were produced")
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_csv.with_name(f".{self.output_csv.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["method", "replicate", "n", "design_id", "field", "relative_l2", "mae"],
            )
            writer.writeheader()
            writer.writerows(rows)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, self.output_csv)
        self.logger.info("wrote %d paired metric rows to %s", len(rows), self.output_csv)
