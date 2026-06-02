#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Self

from pydantic import BaseModel, Field, model_validator

from noether.core.types import InitWeightsMode


class LinearProjectionConfig(BaseModel):
    """Configuration for a LinearProjection layer."""

    input_dim: int = Field(...)
    """Input dimension of the linear projection."""
    output_dim: int = Field(...)
    """Output dimension of the linear projection."""
    ndim: None | int = Field(None)
    """Number of dimensions of the input domain. Either None (Linear projection),  1D (sequence), 2D, or 3D. Defaults to None."""
    bias: bool = Field(True)
    """If true, use bias term in the linear projection. Defaults to True."""
    optional: bool = Field(False)
    """If true and input_dim==output_dim (i.e., there is no up/down projection), then the identity mapping is used. Defaults to False."""
    init_weights: InitWeightsMode = Field("torch")
    """Initialization method of the weights of the MLP. Options are 'torch' (i.e., similar to the module) or 'truncnormal002', or 'zero'. Defaults to 'torch'."""

    @model_validator(mode="after")
    def validate_ndim(self) -> Self:
        """Validate the ndim field to ensure it is either None, 1, 2, or 3."""
        if self.ndim is not None and self.ndim not in [1, 2, 3]:
            raise ValueError("ndim must be either None, 1, 2, or 3.")
        return self
