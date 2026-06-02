#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections.abc import Sequence

from pydantic import BaseModel


class AeroStatsSchema(BaseModel):
    """Unified statistics dataclass for aerodynamics datasets such as AhmedML, and DrivAerML, DrivAerNet++,
    ShapeNet-Car, and Wing."""

    # Surface statistics
    surface_domain_min: tuple[float, float, float] | None = None
    surface_domain_max: tuple[float, float, float] | None = None
    surface_pos_mean: tuple[float, float, float] | None = None
    surface_pos_std: tuple[float, float, float] | None = None
    surface_pressure_mean: tuple[float] | None = None
    surface_pressure_std: tuple[float] | None = None
    surface_friction_mean: tuple[float, float, float] | None = None
    surface_friction_std: tuple[float, float, float] | None = None

    # Volume statistics
    volume_pos_mean: tuple[float, float, float] | None = None
    volume_pos_std: tuple[float, float, float] | None = None
    volume_pressure_mean: tuple[float] | None = None
    volume_pressure_std: tuple[float] | None = None
    volume_velocity_mean: tuple[float, float, float] | None = None
    volume_velocity_std: tuple[float, float, float] | None = None
    volume_vorticity_mean: tuple[float, float, float] | None = None
    volume_vorticity_std: tuple[float, float, float] | None = None
    volume_vorticity_logscale_mean: tuple[float, float, float] | None = None
    volume_vorticity_logscale_std: tuple[float, float, float] | None = None
    volume_vorticity_magnitude_mean: float | None = None
    volume_vorticity_magnitude_std: float | None = None
    volume_domain_min: tuple[float, float, float] | None = None
    volume_domain_max: tuple[float, float, float] | None = None
    volume_sdf_mean: tuple[float] | None = None
    volume_sdf_std: tuple[float] | None = None

    # Inflow design parameter statistics
    inflow_design_parameters_min: Sequence[float] | None = None
    inflow_design_parameters_max: Sequence[float] | None = None
    inflow_design_parameters_mean: Sequence[float] | None = None
    inflow_design_parameters_std: Sequence[float] | None = None

    # Geometry design parameter statistics
    geometry_design_parameters_min: Sequence[float] | None = None
    geometry_design_parameters_max: Sequence[float] | None = None
    geometry_design_parameters_mean: Sequence[float] | None = None
    geometry_design_parameters_std: Sequence[float] | None = None

    # raw position statistics
    raw_pos_min: tuple[float] | None = None
    raw_pos_max: tuple[float] | None = None
