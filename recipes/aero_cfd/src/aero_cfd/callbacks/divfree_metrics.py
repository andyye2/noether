#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import Annotated

import torch
from pydantic import Field, field_validator, model_validator

from noether.core.callbacks.periodic import PeriodicDataIteratorCallback
from noether.core.schemas.callbacks import PeriodicDataIteratorCallbackConfig
from noether.modeling.functional import (
    central_difference_positions,
    divergence_from_perturbed_field,
)


def _scale_vector(value: float | list[float] | torch.Tensor) -> torch.Tensor:
    scale = torch.as_tensor(value, dtype=torch.float32)
    if scale.numel() == 1:
        scale = scale.expand(3).clone()
    if scale.shape != (3,):
        raise ValueError("position_scale must be a scalar or have length 3")
    return scale


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "module", model)


class DivFreeMetricsCallbackConfig(PeriodicDataIteratorCallbackConfig):
    """Configuration for physical-space divergence monitoring of a DivFree model."""

    kind: str | None = "aero_cfd.callbacks.DivFreeMetricsCallback"

    forward_properties: list[str] = Field(default_factory=list)
    delta: Annotated[float, Field(gt=0)] | None = None
    position_scale: float | list[float] | None = None
    num_monitor_anchors: Annotated[int, Field(gt=0)] | None = 256
    max_samples: Annotated[int, Field(gt=0)] | None = 4
    monitor_dataset_field: str = "volume_velocity"
    batch_size: int = Field(1)

    @field_validator("position_scale")
    @classmethod
    def _validate_position_scale(cls, value):
        if value is None:
            return value
        values = value if isinstance(value, list) else [value]
        if isinstance(value, list) and len(value) != 3:
            raise ValueError("position_scale must be a scalar or have length 3")
        if any(v <= 0 for v in values):
            raise ValueError("position_scale must be positive")
        return value

    @model_validator(mode="after")
    def _validate_batch_size(self) -> DivFreeMetricsCallbackConfig:
        if self.batch_size != 1:
            raise ValueError("DivFreeMetricsCallback only supports batch_size=1")
        return self


class DivFreeMetricsCallback(PeriodicDataIteratorCallback):
    """Monitor physical-space divergence of a DivFreeAeroABUPT-like model."""

    def __init__(self, callback_config: DivFreeMetricsCallbackConfig, **kwargs):
        super().__init__(callback_config, development=True, **kwargs)
        self._config = callback_config
        self.dataset_key = callback_config.dataset_key
        self.forward_properties = callback_config.forward_properties
        self.num_monitor_anchors = callback_config.num_monitor_anchors
        self.field = callback_config.monitor_dataset_field
        self._mismatch_warned = False

        self.sampler_config = self._sampler_config_from_key(
            key=self.dataset_key,
            max_size=callback_config.max_samples,
        )

    def _resolve_delta(self, model: torch.nn.Module) -> float:
        if self._config.delta is not None:
            return float(self._config.delta)
        return float(_unwrap(model).delta)

    def _resolve_position_scale(self, model: torch.nn.Module):
        if self._config.position_scale is not None:
            return self._config.position_scale
        return _unwrap(model).position_scale

    def _warn_position_scale_mismatch_once(self, model: torch.nn.Module) -> None:
        if self._mismatch_warned or self._config.position_scale is None:
            return
        model_ps = getattr(_unwrap(model), "position_scale", None)
        if model_ps is None:
            return
        cfg_ps = _scale_vector(self._config.position_scale)
        model_ps_cpu = model_ps.detach().cpu().float()
        if not torch.allclose(cfg_ps, model_ps_cpu):
            self.logger.warning(
                "DivFreeMetricsCallback.position_scale (%s) disagrees with "
                "model.position_scale (%s); using config value.",
                cfg_ps.tolist(),
                model_ps_cpu.tolist(),
            )
        self._mismatch_warned = True

    def process_data(
        self,
        batch: dict[str, torch.Tensor],
        *,
        trainer_model: torch.nn.Module,
        **_,
    ) -> dict[str, torch.Tensor]:
        self._warn_position_scale_mismatch_once(trainer_model)

        anchor = batch["volume_anchor_position"]
        if self.num_monitor_anchors is not None and anchor.shape[1] > self.num_monitor_anchors:
            anchor = anchor[:, : self.num_monitor_anchors]

        delta = self._resolve_delta(trainer_model)
        position_scale = self._resolve_position_scale(trainer_model)
        query_pos = central_difference_positions(anchor, delta)

        forward_kwargs = {key: batch[key] for key in self.forward_properties if key in batch}
        forward_kwargs["query_volume_position"] = query_pos

        with self.trainer.autocast_context:
            out = trainer_model(**forward_kwargs)

        if "query_volume_velocity" not in out:
            raise KeyError(
                "DivFreeMetricsCallback expected model output 'query_volume_velocity'; "
                f"got keys {sorted(out.keys())}. Is the model a DivFreeAeroABUPT subclass?"
            )

        dataset = self.data_container.get_dataset(self.dataset_key)
        u_phys = dataset.denormalize(self.field, out["query_volume_velocity"])

        div = divergence_from_perturbed_field(u_phys, delta, position_scale)
        div_abs = div.abs()
        return {
            "divergence_abs_mean": div_abs.mean(),
            "divergence_abs_max": div_abs.max(),
            "divergence_square_mean": div.square().mean(),
        }

    def process_results(self, results: dict[str, torch.Tensor], *, interval_type, update_counter, **_) -> None:
        if not results:
            self.logger.warning("DivFreeMetricsCallback: empty results for dataset '%s'", self.dataset_key)
            return

        final = {
            "divergence_abs_mean": results["divergence_abs_mean"].mean(),
            "divergence_abs_max": results["divergence_abs_max"].max(),
            "divergence_rms": results["divergence_square_mean"].mean().sqrt(),
        }
        for name, value in final.items():
            self.writer.add_scalar(
                key=f"divergence/{self.dataset_key}/{name}",
                value=value,
                logger=self.logger,
                format_str=".6e",
            )
