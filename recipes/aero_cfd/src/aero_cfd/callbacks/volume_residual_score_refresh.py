#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import os
from pathlib import Path

import torch
from pydantic import Field, model_validator

from aero_cfd.pipeline.sample_processors import make_volume_position_fingerprint
from noether.core.callbacks.periodic import PeriodicDataIteratorCallback
from noether.core.distributed.config import get_rank
from noether.core.schemas.callbacks import PeriodicDataIteratorCallbackConfig


class VolumeResidualScoreRefreshCallbackConfig(PeriodicDataIteratorCallbackConfig):
    """Configuration for refreshing per-point volume sampling scores."""

    kind: str | None = "aero_cfd.callbacks.VolumeResidualScoreRefreshCallback"

    score_dir: str
    """Directory where per-sample sampling score sidecars are written."""
    query_chunk_size: int = Field(10000, gt=0)
    """Max number of candidate volume query points per forward pass."""
    score_key: str = "volume_sampling_score"
    """Sidecar key used by the score-aware anchor sampler."""
    position_key: str = "volume_score_position"
    """Batch key containing full candidate volume positions."""
    target_key: str = "volume_score_velocity"
    """Batch key containing full candidate volume velocity targets."""
    prediction_key: str = "query_volume_velocity"
    """Model output key for query-mode volume velocity predictions."""
    batch_size: int = Field(1)

    @model_validator(mode="after")
    def validate_config(self) -> VolumeResidualScoreRefreshCallbackConfig:
        if self.batch_size != 1:
            raise ValueError("VolumeResidualScoreRefreshCallback only supports batch_size=1")
        return self


class VolumeResidualScoreRefreshCallback(PeriodicDataIteratorCallback):
    """Periodically recomputes residual-based volume sampling scores."""

    def __init__(self, callback_config: VolumeResidualScoreRefreshCallbackConfig, **kwargs):
        super().__init__(callback_config, **kwargs)
        self.score_dir = Path(callback_config.score_dir)
        self.query_chunk_size = callback_config.query_chunk_size
        self.score_key = callback_config.score_key
        self.position_key = callback_config.position_key
        self.target_key = callback_config.target_key
        self.prediction_key = callback_config.prediction_key

    def process_data(self, batch: dict[str, torch.Tensor], **_) -> dict[str, torch.Tensor]:
        position = self._required_tensor(batch, self.position_key)
        target = self._required_tensor(batch, self.target_key)
        index = self._scalar_int(self._required_tensor(batch, "index"), "index")

        if position.shape[0] != 1 or target.shape[0] != 1:
            raise ValueError("VolumeResidualScoreRefreshCallback only supports batch_size=1")
        if position.ndim != 3 or target.ndim != 3:
            raise ValueError(f"Expected 3D position and target tensors, got {position.ndim}D and {target.ndim}D")
        if position.shape[:2] != target.shape[:2]:
            raise ValueError(f"Position and target lengths do not match: {position.shape} vs {target.shape}")

        prediction = self._run_query_inference(batch=batch, position=position)
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction and target shapes do not match: {prediction.shape} vs {target.shape}")

        score = (prediction.float() - target.float()).square().mean(dim=-1).squeeze(0).detach().cpu()
        if not torch.isfinite(score).all():
            self.logger.warning(f"Skipping sampling score sidecar for index={index}: score contains NaN or Inf")
            return self._result(position, written=0, failed=1)

        self._write_sidecar(
            index=index,
            run_name=self._run_name(index),
            score=score,
            position=position.squeeze(0),
        )
        return self._result(position, written=1, failed=0, num_points=score.numel(), score_sum=float(score.sum()))

    def _run_query_inference(
        self,
        *,
        batch: dict[str, torch.Tensor],
        position: torch.Tensor,
    ) -> torch.Tensor:
        base_kwargs = {
            "geometry_position": batch["geometry_position"],
            "geometry_supernode_idx": batch["geometry_supernode_idx"],
            "geometry_batch_idx": batch["geometry_batch_idx"],
            "surface_anchor_position": batch["surface_anchor_position"],
            "volume_anchor_position": batch["volume_anchor_position"],
        }

        chunks = []
        num_points = position.shape[1]
        for start in range(0, num_points, self.query_chunk_size):
            end = min(start + self.query_chunk_size, num_points)
            chunk_kwargs = dict(base_kwargs)
            chunk_kwargs["query_volume_position"] = position[:, start:end]

            with self.trainer.autocast_context:
                output = self.model(**chunk_kwargs)

            if self.prediction_key not in output:
                raise KeyError(f"Model output does not contain {self.prediction_key!r}")
            chunks.append(output[self.prediction_key])

        return torch.cat(chunks, dim=1)

    def _write_sidecar(self, *, index: int, run_name: str | None, score: torch.Tensor, position: torch.Tensor) -> None:
        self.score_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = self.score_dir / f"score_{index}.pt"
        tmp_path = self.score_dir / f".{sidecar_path.name}.rank{get_rank()}.{os.getpid()}.tmp"

        payload = {
            "index": index,
            "run_name": run_name,
            self.score_key: score.to(dtype=torch.float32),
            "volume_position_fingerprint": make_volume_position_fingerprint(position),
        }
        try:
            torch.save(payload, tmp_path)
            tmp_path.replace(sidecar_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _run_name(self, index: int) -> str | None:
        dataset = self.data_container.get_dataset(self.dataset_key)
        if not hasattr(dataset, "sample_info"):
            return None
        info = dataset.sample_info(index)
        run_name = info.get("run_name") if isinstance(info, dict) else None
        return str(run_name) if run_name is not None else None

    @staticmethod
    def _required_tensor(batch: dict[str, torch.Tensor], key: str) -> torch.Tensor:
        value = batch.get(key)
        if not torch.is_tensor(value):
            raise KeyError(f"Batch is missing required tensor {key!r}")
        return value

    @staticmethod
    def _scalar_int(value: torch.Tensor, key: str) -> int:
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor for {key!r}, got shape {tuple(value.shape)}")
        return int(value.detach().cpu().item())

    @staticmethod
    def _result(
        position: torch.Tensor,
        *,
        written: int,
        failed: int,
        num_points: int = 0,
        score_sum: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        device = position.device
        return {
            "num_written": torch.tensor(written, device=device),
            "num_failed": torch.tensor(failed, device=device),
            "num_points": torch.tensor(num_points, device=device),
            "score_sum": torch.tensor(score_sum, device=device),
        }

    def process_results(self, results: dict[str, torch.Tensor], **_) -> None:
        num_written = results["num_written"].sum()
        num_failed = results["num_failed"].sum()
        num_points = results["num_points"].sum()
        score_sum = results["score_sum"].sum()

        self.writer.add_scalar(
            key=f"score_refresh/{self.dataset_key}/num_written",
            value=num_written,
            logger=self.logger,
            format_str=".0f",
        )
        if num_failed > 0:
            self.writer.add_scalar(
                key=f"score_refresh/{self.dataset_key}/num_failed",
                value=num_failed,
                logger=self.logger,
                format_str=".0f",
            )
        if num_points > 0:
            self.writer.add_scalar(
                key=f"score_refresh/{self.dataset_key}/mean_score",
                value=score_sum / num_points,
                logger=self.logger,
                format_str=".6f",
            )
