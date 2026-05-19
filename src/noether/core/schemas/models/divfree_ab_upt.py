#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from pydantic import Field, field_validator

from noether.core.schemas.models.ab_upt import AnchorBranchedUPTConfig


class DivFreeAnchorBranchedUPTConfig(AnchorBranchedUPTConfig):
    """Configuration for divergence-free AB-UPT velocity prediction.

    This extends :class:`AnchorBranchedUPTConfig` with the parameters needed to
    interpret the volume velocity head as a vector potential, compute its curl
    in physical coordinates, and normalize the resulting velocity for the
    existing aerodynamics losses.

    Args:
        delta: Central-difference perturbation step in network coordinates.
        position_scale: ``dx_phys / dx_net``. Use a scalar for isotropic scaling
            or a length-3 value for anisotropic scaling.
        volume_velocity_mean: Mean used to normalize physical volume velocity.
        volume_velocity_std: Standard deviation used to normalize physical
            volume velocity.
        zero_init_volume_projection: Whether to zero-initialize the volume
            decoder projection so the initial curl velocity is close to zero.

    Example:

        .. testcode::

            from noether.core.schemas.models.divfree_ab_upt import DivFreeAnchorBranchedUPTConfig

            assert DivFreeAnchorBranchedUPTConfig.model_fields["delta"].default == 1e-3
    """

    kind: str | None = "noether.core.schemas.models.DivFreeAnchorBranchedUPTConfig"

    delta: float = Field(1e-3, gt=0)
    """Central-difference perturbation step in network coordinates."""

    position_scale: float | list[float] = Field(...)
    """Physical-to-network coordinate scale, ``dx_phys / dx_net``."""

    volume_velocity_mean: list[float] = Field(...)
    """Per-component mean of physical volume velocity."""

    volume_velocity_std: list[float] = Field(...)
    """Per-component standard deviation of physical volume velocity."""

    zero_init_volume_projection: bool = True
    """Whether to zero-initialize the volume decoder projection."""

    @field_validator("position_scale")
    @classmethod
    def validate_position_scale(cls, value: float | list[float]) -> float | list[float]:
        """Validate isotropic or per-axis physical position scale."""
        values = value if isinstance(value, list) else [value]
        if isinstance(value, list) and len(value) != 3:
            raise ValueError("position_scale must be a scalar or have length 3")
        if any(v <= 0 for v in values):
            raise ValueError("position_scale must be positive")
        return value

    @field_validator("volume_velocity_mean")
    @classmethod
    def validate_volume_velocity_mean(cls, value: list[float]) -> list[float]:
        """Validate velocity mean shape."""
        if len(value) != 3:
            raise ValueError("volume_velocity_mean must have length 3")
        return value

    @field_validator("volume_velocity_std")
    @classmethod
    def validate_volume_velocity_std(cls, value: list[float]) -> list[float]:
        """Validate velocity standard deviation shape."""
        if len(value) != 3:
            raise ValueError("volume_velocity_std must have length 3")
        if any(v <= 0 for v in value):
            raise ValueError("volume_velocity_std must be positive")
        return value
