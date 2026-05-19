#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import torch
from torch import nn

from noether.core.schemas.lib import ConfiguredBy
from noether.core.schemas.models import DivFreeAnchorBranchedUPTConfig
from noether.modeling.functional import (
    central_difference_positions,
    curl_from_perturbed_potential,
    normalize_vector_field,
)
from noether.modeling.models.aerodynamics import AeroABUPT


@ConfiguredBy(DivFreeAnchorBranchedUPTConfig)
class DivFreeAeroABUPT(AeroABUPT):
    """Aerodynamic AB-UPT wrapper with divergence-free volume velocity.

    The volume velocity decoder is interpreted internally as a vector
    potential. The forward pass later converts that potential to velocity via
    a finite-difference curl while keeping the public output key as
    ``volume_velocity``.

    Args:
        model_config: Div-free AB-UPT configuration.

    Example:

        .. testcode::

            from noether.modeling.models.divfree_aerodynamics import DivFreeAeroABUPT

            assert DivFreeAeroABUPT._config_class is not None
    """

    def __init__(self, model_config: DivFreeAnchorBranchedUPTConfig, **kwargs) -> None:
        self._validate_volume_velocity_spec(model_config)
        super().__init__(model_config=model_config, **kwargs)
        self.delta = model_config.delta

        self.register_buffer("position_scale", self._as_vector(model_config.position_scale))
        self.register_buffer("volume_velocity_mean", self._as_vector(model_config.volume_velocity_mean))
        self.register_buffer("volume_velocity_std", self._as_vector(model_config.volume_velocity_std))

        if model_config.zero_init_volume_projection:
            self._zero_init_volume_projection()

    @staticmethod
    def _as_vector(value: float | list[float]) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.numel() == 1:
            tensor = tensor.expand(3).clone()
        if tensor.shape != (3,):
            raise ValueError("value must be a scalar or have shape (3,)")
        return tensor

    @staticmethod
    def _validate_volume_velocity_spec(model_config: DivFreeAnchorBranchedUPTConfig) -> None:
        volume_spec = model_config.data_specs.domains.get("volume")
        if (
            volume_spec is None
            or "velocity" not in volume_spec.output_dims.keys()
            or volume_spec.output_dims["velocity"] != 3
        ):
            raise ValueError("DivFreeAeroABUPT requires volume output_dims to contain velocity: 3")

    def _zero_init_volume_projection(self) -> None:
        readout = self.backbone.domain_decoder_projections["volume"]
        projection = readout.linear.project

        if not isinstance(projection, nn.Linear):
            raise TypeError("volume decoder projection must be an nn.Linear")

        nn.init.zeros_(projection.weight)
        if projection.bias is not None:
            nn.init.zeros_(projection.bias)

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Predict normalized divergence-free volume velocity.

        The base AB-UPT volume decoder is evaluated once at central-difference
        perturbations around volume anchors and optional query points. The
        perturbation predictions are interpreted as vector potential values
        and converted to velocity with a curl.

        Args:
            **kwargs: Flat aerodynamics inputs accepted by :class:`AeroABUPT`.
                ``volume_anchor_position`` is required. ``query_volume_position``
                is optional.

        Returns:
            Prediction dictionary with ``volume_velocity`` replaced by the
            normalized curl velocity. When user volume queries are provided,
            ``query_volume_velocity`` contains their normalized curl velocity.

        Raises:
            ValueError: If volume positions are missing or malformed.
        """
        volume_anchor_position = kwargs.get("volume_anchor_position")
        if volume_anchor_position is None:
            raise ValueError("DivFreeAeroABUPT requires volume_anchor_position")
        if volume_anchor_position.ndim != 3 or volume_anchor_position.shape[-1] != 3:
            raise ValueError("volume_anchor_position must have shape (batch_size, num_points, 3)")
        if "volume_query_features" in kwargs:
            raise ValueError("DivFreeAeroABUPT does not support volume_query_features")

        query_volume_position = kwargs.get("query_volume_position")
        if query_volume_position is not None and (
            query_volume_position.ndim != 3 or query_volume_position.shape[-1] != 3
        ):
            raise ValueError("query_volume_position must have shape (batch_size, num_points, 3)")

        kwargs = dict(kwargs)
        anchor_size = volume_anchor_position.shape[1]
        query_size = 0
        anchor_perturbed_position = central_difference_positions(volume_anchor_position, self.delta)

        if query_volume_position is None:
            kwargs["query_volume_position"] = anchor_perturbed_position
        else:
            query_size = query_volume_position.shape[1]
            query_perturbed_position = central_difference_positions(query_volume_position, self.delta)
            kwargs["query_volume_position"] = torch.cat(
                [anchor_perturbed_position, query_perturbed_position],
                dim=1,
            )

        output = super().forward(**kwargs)
        volume_query_potential = output.pop("query_volume_velocity")

        anchor_stop = 6 * anchor_size
        anchor_potential = volume_query_potential[:, :anchor_stop]
        output["volume_velocity"] = self._curl_and_normalize(anchor_potential)

        if query_volume_position is not None:
            query_stop = anchor_stop + 6 * query_size
            query_potential = volume_query_potential[:, anchor_stop:query_stop]
            output["query_volume_velocity"] = self._curl_and_normalize(query_potential)

        return output

    def _curl_and_normalize(self, potential: torch.Tensor) -> torch.Tensor:
        velocity = curl_from_perturbed_potential(
            potential,
            delta=self.delta,
            position_scale=self.position_scale,
        )
        return normalize_vector_field(
            velocity,
            mean=self.volume_velocity_mean,
            std=self.volume_velocity_std,
        )
