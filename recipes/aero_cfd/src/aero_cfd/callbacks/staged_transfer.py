# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Update-based staged unfreezing for transfer-learning diagnostics."""

from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, Field, model_validator

from noether.core.callbacks.periodic import IntervalType, PeriodicCallback
from noether.core.schemas.callbacks import CallBackBaseConfig
from noether.core.utils.training.counter import UpdateCounter


class TransferStageConfig(BaseModel):
    """Trainability mask that starts at an absolute optimizer update."""

    start_update: int = Field(ge=0)
    trainable_patterns: list[str] = Field(default_factory=list)
    train_all: bool = False

    @model_validator(mode="after")
    def validate_mask(self) -> TransferStageConfig:
        """Require either an explicit pattern list or the all-parameter flag."""
        if self.train_all == bool(self.trainable_patterns):
            raise ValueError("exactly one of train_all or trainable_patterns must be specified")
        return self


class StagedTransferCallbackConfig(CallBackBaseConfig):
    """Configuration for deterministic update-based staged unfreezing."""

    kind: str | None = "aero_cfd.callbacks.staged_transfer.StagedTransferCallback"
    every_n_updates: int = 1
    stages: list[TransferStageConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_stages(self) -> StagedTransferCallbackConfig:
        """Require a stage at update zero and strictly increasing boundaries."""
        starts = [stage.start_update for stage in self.stages]
        if starts[0] != 0:
            raise ValueError("the first transfer stage must start at update 0")
        if starts != sorted(set(starts)):
            raise ValueError(f"stage start_update values must be strictly increasing, got {starts}")
        if any(stage.train_all for stage in self.stages[:-1]):
            raise ValueError("train_all is only valid for the final stage")
        return self


class StagedTransferCallback(PeriodicCallback):
    """Apply parameter-name masks and progressively unfreeze by update count.

    The optimizer already owns all model parameters, so changing
    ``requires_grad`` is sufficient.  Distributed trainers must enable
    ``find_unused_params`` while a subset of parameters is frozen.
    """

    def __init__(
        self,
        callback_config: StagedTransferCallbackConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(callback_config=callback_config, **kwargs)
        self.stages = tuple(callback_config.stages)
        self.stage_index = 0

    def _stage_for_update(self, update: int) -> int:
        """Return the highest stage whose start boundary has been reached."""
        index = 0
        for candidate, stage in enumerate(self.stages):
            if update < stage.start_update:
                break
            index = candidate
        return index

    def _apply_stage(self, stage_index: int) -> tuple[int, int]:
        """Apply one mask and return trainable/frozen parameter counts."""
        stage = self.stages[stage_index]
        matched_names: list[str] = []
        trainable = 0
        frozen = 0
        for name, parameter in self.model.named_parameters():
            should_train = stage.train_all or any(pattern in name for pattern in stage.trainable_patterns)
            parameter.requires_grad_(should_train)
            if should_train:
                matched_names.append(name)
                trainable += parameter.numel()
            else:
                frozen += parameter.numel()
        if not matched_names:
            raise ValueError(f"stage {stage_index} matched no parameters: patterns={stage.trainable_patterns}")
        self.stage_index = stage_index
        self.logger.info(
            "applied transfer stage=%d start_update=%d trainable=%d frozen=%d patterns=%s train_all=%s",
            stage_index,
            stage.start_update,
            trainable,
            frozen,
            stage.trainable_patterns,
            stage.train_all,
        )
        return trainable, frozen

    def before_training(self, *, update_counter: UpdateCounter) -> None:
        """Restore the appropriate stage before the first forward pass."""
        current_update = update_counter.cur_iteration.update or 0
        expected_stage = self._stage_for_update(current_update)
        if self.stage_index > expected_stage:
            expected_stage = self.stage_index
        self._apply_stage(expected_stage)

    def periodic_callback(
        self,
        *,
        interval_type: IntervalType,
        update_counter: UpdateCounter,
        **kwargs: Any,
    ) -> None:
        """Advance the mask when an optimizer-update boundary is reached."""
        del kwargs
        if interval_type != "update":
            return
        current_update = update_counter.cur_iteration.update or 0
        expected_stage = self._stage_for_update(current_update)
        if expected_stage != self.stage_index:
            self._apply_stage(expected_stage)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Persist the active stage for exact resumption."""
        return {"stage_index": torch.tensor(self.stage_index, dtype=torch.int64)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the active stage index from a trainer checkpoint."""
        value = state_dict["stage_index"]
        stage_index = int(value.item()) if isinstance(value, torch.Tensor) else int(value)
        if not 0 <= stage_index < len(self.stages):
            raise ValueError(f"invalid restored stage_index={stage_index}")
        self.stage_index = stage_index
