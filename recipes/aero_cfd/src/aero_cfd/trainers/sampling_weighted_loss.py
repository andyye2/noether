#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import torch
from pydantic import Field

from noether.core.schemas.trainers import WeightedLossTrainerConfig
from noether.training.trainers import WeightedLossTrainer


class SamplingWeightedLossTrainerConfig(WeightedLossTrainerConfig):
    """Config for a weighted-loss trainer with optional per-point sampling weights."""

    kind: str = "aero_cfd.trainers.SamplingWeightedLossTrainer"

    sample_weight_keys: dict[str, str] = Field(default_factory=dict)
    """Mapping from output field name to the batch key holding its per-point sampling weight."""


class SamplingWeightedLossTrainer(WeightedLossTrainer):
    """Weighted-loss trainer that applies per-point sampling weights.

    Behaves exactly like :class:`~noether.training.trainers.weighted_loss.WeightedLossTrainer`
    unless ``sample_weight_keys`` is set. For each configured field, the per-point loss is
    multiplied by the matching batch weight from the score-aware anchor sampler.
    """

    def __init__(self, trainer_config: SamplingWeightedLossTrainerConfig, **kwargs):
        """
        Args:
            trainer_config: Configuration for the trainer.
            **kwargs: Additional keyword arguments for the base trainer.

        Raises:
            ValueError: If ``sample_weight_keys`` references a field without a configured loss.
        """
        super().__init__(trainer_config, **kwargs)
        self.sample_weight_keys = dict(trainer_config.sample_weight_keys)

        loss_fields = {field_name for field_name, _ in self.loss_items}
        unknown_fields = sorted(set(self.sample_weight_keys) - loss_fields)
        if unknown_fields:
            raise ValueError(
                f"sample_weight_keys reference fields without a configured loss: {unknown_fields}. "
                f"Available: {sorted(loss_fields)}"
            )

        self.batch_keys = self.batch_keys | set(self.sample_weight_keys.values())

    def _split_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Splits the batch like the base trainer, then routes sampling weights into the targets."""
        forward_batch, targets_batch = super()._split_batch(batch)
        for weight_key in self.sample_weight_keys.values():
            if weight_key in batch:
                targets_batch[weight_key] = batch[weight_key]
        return forward_batch, targets_batch

    def loss_compute(
        self, forward_output: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Computes the per-field loss, applying per-point sampling weights where configured.

        Args:
            forward_output: Output of the model after the forward pass.
            targets: Target tensors plus any routed sampling-weight tensors.

        Returns:
            A dictionary with the weighted per-field losses to log.
        """
        losses: dict[str, torch.Tensor] = {}
        for field_name, weight in self.loss_items:
            if weight <= 0 or field_name not in forward_output:
                continue
            target_key = f"{field_name}_target"
            if target_key not in targets:
                raise ValueError(f"Target '{target_key}' not found in targets. Available: {list(targets.keys())}")

            sample_weight_key = self.sample_weight_keys.get(field_name)
            if sample_weight_key is None:
                field_loss = self._loss_fn(targets[target_key], forward_output[field_name])
            else:
                if sample_weight_key not in targets:
                    raise ValueError(
                        f"Sample weight '{sample_weight_key}' not found in targets. Available: {list(targets.keys())}"
                    )
                field_loss = self._weighted_loss(
                    target=targets[target_key],
                    prediction=forward_output[field_name],
                    sample_weight=targets[sample_weight_key],
                    sample_weight_key=sample_weight_key,
                )
            losses[f"{field_name}_loss"] = field_loss * weight
        if not losses:
            raise ValueError(
                "No losses computed. Check that 'field_weights' keys match model output keys and 'target_properties'."
            )
        return losses

    def _weighted_loss(
        self,
        *,
        target: torch.Tensor,
        prediction: torch.Tensor,
        sample_weight: torch.Tensor,
        sample_weight_key: str,
    ) -> torch.Tensor:
        """Reduces the per-element loss with broadcast per-point sampling weights."""
        try:
            point_loss = self._loss_fn(target, prediction, reduction="none")
        except TypeError as exc:
            raise ValueError("Sample weighting requires a loss_fn that supports reduction='none'.") from exc

        sample_weight = sample_weight.to(device=point_loss.device, dtype=point_loss.dtype)
        if not torch.isfinite(sample_weight).all():
            raise ValueError(f"Sample weight '{sample_weight_key}' contains NaN or Inf.")
        if (sample_weight < 0).any():
            raise ValueError(f"Sample weight '{sample_weight_key}' contains negative values.")
        if sample_weight.ndim > point_loss.ndim:
            raise ValueError(
                f"Sample weight '{sample_weight_key}' shape {tuple(sample_weight.shape)} "
                f"is not broadcastable to loss shape {tuple(point_loss.shape)}."
            )
        while sample_weight.ndim < point_loss.ndim:
            sample_weight = sample_weight.unsqueeze(-1)

        try:
            return (point_loss * sample_weight).mean()
        except RuntimeError as exc:
            raise ValueError(
                f"Sample weight '{sample_weight_key}' shape {tuple(sample_weight.shape)} "
                f"is not broadcastable to loss shape {tuple(point_loss.shape)}."
            ) from exc
