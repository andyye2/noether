#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from pydantic import BaseModel, Field


class RopeFrequencyConfig(BaseModel):
    """Configuration for RoPE frequency settings."""

    hidden_dim: int = Field(...)
    """Dimensionality of frequencies (in transformers this should be the head dimension)."""
    input_dim: int = Field(...)
    """Dimensionality of the coordinates (e.g., 2 for 2D coordinates, 3 for 3D coordinates)."""
    max_wavelength: int = Field(10000)
    """ Theta parameter for the transformer sine/cosine embedding. Default: 10000.0"""
    implementation: Literal["real", "complex"] = Field("real")
    """
    "real" -> basic implementation using real coordinates (this is slow and only here for backward compatibility).
    "complex" -> fast implementation of rotation via complex multiplication. Default: "real".
    """
