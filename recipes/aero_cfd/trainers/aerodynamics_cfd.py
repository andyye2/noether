#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import torch
import torch.nn.functional as F

from noether.core.schemas.trainers import BaseTrainerConfig
from noether.training.trainers import BaseTrainer


class AerodynamicsCfdTrainerConfig(BaseTrainerConfig):
    surface_weight: float = 1.0
    """ Weight of the predicted values on the surface mesh. Defaults to 1.0.."""
    volume_weight: float = 1.0
    """Weight of the predicted values in the volume. Defaults to 1.0."""
    surface_pressure_weight: float = 1.0
    """Weight of the predicted values for the surface pressure. Defaults to 1.0."""
    surface_friction_weight: float = 0.0
    """Weight of the predicted values for the surface wall shear stress. Defaults to 0.0."""
    volume_velocity_weight: float = 1.0
    """Weight of the predicted values for the volume velocity. Defaults to 1.0."""
    volume_pressure_weight: float = 0.0
    """Weight of the predicted values for the volume total pressure coefficient. Defaults to 0.0."""
    volume_vorticity_weight: float = 0.0
    """Weight of the predicted values for the volume vorticity. Defaults to 0.0."""
    use_physics_features: bool = False
    wall_no_penetration_weight: float = 0.0
    """Weight for the wall no-penetration residual on near-wall volume queries."""
    wall_no_penetration_velocity_scale: float = 20.0
    """Velocity scale used to non-dimensionalize the wall-normal velocity residual."""
    wall_query_normal_key: str = "wall_query_normals"
    """Target key containing normals for wall no-penetration queries."""
    volume_velocity_mean: tuple[float, float, float] | None = None
    """Mean used to denormalize predicted volume velocity before wall residuals."""
    volume_velocity_std: tuple[float, float, float] | None = None
    """Standard deviation used to denormalize predicted volume velocity before wall residuals."""


class AerodynamicsCFDTrainer(BaseTrainer):
    """Trainer class for to train automative aerodynaimcs CFD for the: AhmedML, DrivaerML and Shapenet-Car Car dataset."""

    def __init__(self, trainer_config: AerodynamicsCfdTrainerConfig, **kwargs):
        """Trainer class for to train automative aerodynaimcs CFD for the: AhmedML, DrivaerML and Shapenet-Car Car dataset.

        Args:
            trainer_config: Configuration for the trainer.
            **kwargs: Additional keyword arguments for the SgdTrainer.

        Raises:
            ValueError: When an output mode is not defined in the loss items.
        """
        super().__init__(
            config=trainer_config,
            **kwargs,
        )

        self.surface_pressure_weight = trainer_config.surface_pressure_weight
        self.surface_friction_weight = trainer_config.surface_friction_weight
        self.volume_velocity_weight = trainer_config.volume_velocity_weight
        self.volume_pressure_weight = trainer_config.volume_pressure_weight
        self.volume_vorticity_weight = trainer_config.volume_vorticity_weight

        self.surface_weight = trainer_config.surface_weight
        self.volume_weight = trainer_config.volume_weight

        self.wall_no_penetration_weight = trainer_config.wall_no_penetration_weight
        self.wall_no_penetration_velocity_scale = trainer_config.wall_no_penetration_velocity_scale
        self.wall_query_normal_key = trainer_config.wall_query_normal_key
        self.volume_velocity_mean = trainer_config.volume_velocity_mean
        self.volume_velocity_std = trainer_config.volume_velocity_std
        if self.wall_no_penetration_weight > 0:
            if self.wall_no_penetration_velocity_scale <= 0:
                raise ValueError("wall_no_penetration_velocity_scale must be positive.")
            if self.volume_velocity_mean is None or self.volume_velocity_std is None:
                raise ValueError("volume_velocity_mean and volume_velocity_std are required for wall no-penetration loss.")

        loss_items = {
            "surface_pressure": (self.surface_pressure_weight, self.surface_weight),
            "surface_friction": (
                self.surface_friction_weight,
                self.surface_weight,
            ),  # not used for ShapeNet-Car
            "volume_velocity": (self.volume_velocity_weight, self.volume_weight),
            "volume_pressure": (self.volume_pressure_weight, self.volume_weight),  # not used for ShapeNet-Car
            "volume_vorticity": (self.volume_vorticity_weight, self.volume_weight),  # not used for ShapeNet-Car
        }

        auxiliary_target_properties = {self.wall_query_normal_key}
        self.loss_items = []
        for target_property in self.target_properties:
            if target_property in auxiliary_target_properties:
                continue
            if not target_property.endswith("_target"):
                raise ValueError(f"Target property '{target_property}' must end with '_target'.")
            item = target_property[: -len("_target")]
            if item not in loss_items:
                raise ValueError(f"Output mode '{target_property}' is not defined in loss items.")
            self.loss_items.append((item, loss_items[item][0], loss_items[item][1]))

    def _wall_no_penetration_loss(
        self, forward_output: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if "query_volume_velocity" not in forward_output:
            raise ValueError("query_volume_velocity is required for wall no-penetration loss.")
        if self.wall_query_normal_key not in targets:
            raise ValueError(f"{self.wall_query_normal_key} is required for wall no-penetration loss.")

        velocity = forward_output["query_volume_velocity"].float()
        normals = targets[self.wall_query_normal_key].to(device=velocity.device).float()
        normals = F.normalize(normals, dim=-1)

        stat_shape = (1,) * (velocity.ndim - 1) + (velocity.shape[-1],)
        mean = torch.as_tensor(self.volume_velocity_mean, device=velocity.device, dtype=velocity.dtype).view(stat_shape)
        std = torch.as_tensor(self.volume_velocity_std, device=velocity.device, dtype=velocity.dtype).view(stat_shape)
        velocity = velocity * std + mean

        normal_velocity = (velocity * normals).sum(dim=-1)
        residual = normal_velocity / self.wall_no_penetration_velocity_scale
        return residual.square().mean()

    def loss_compute(
        self, forward_output: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Given the output of the model and the targets, compute the losses.
        Args:
            forward_output The output of the model, containing the predictions for each output mode.
            targets: Dict containing all target values to compute the loss.

        Returns:
            A dictionary containing the computed losses for each output mode.
        """
        losses: dict[str, torch.Tensor] = {}
        for item, weight, group_weight in self.loss_items:
            if weight > 0 and group_weight > 0 and item in forward_output:
                if f"{item}_target" not in targets:
                    raise ValueError(
                        f"Target for '{item}' not found in targets. Ensure the targets contain the correct keys."
                    )
                losses[f"{item}_loss"] = (
                    F.mse_loss(targets[f"{item}_target"], forward_output[item]) * weight * group_weight
                )
        if self.wall_no_penetration_weight > 0:
            losses["wall_no_penetration_loss"] = (
                self._wall_no_penetration_loss(forward_output, targets)
                * self.wall_no_penetration_weight
                * self.surface_weight
            )
        if len(losses) == 0:
            raise ValueError("No losses computed, check your output keys and loss function.")
        return losses
