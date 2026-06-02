#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from pydantic import BaseModel, Field


class LayerScaleConfig(BaseModel):
    """Configuration for Layer Scale module."""

    hidden_dim: int = Field(...)
    """ Number of dimensions of the input tensor to be scaled."""
    init_values: float | None = Field(1e-5)
    """ Initial gamme scale value. Defaults to 1e-5."""
