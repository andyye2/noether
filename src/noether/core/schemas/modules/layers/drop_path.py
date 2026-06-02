#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from pydantic import BaseModel, Field


class UnquantizedDropPathConfig(BaseModel):
    """Configuration for the UnquantizedDropPath layer."""

    drop_prob: float = Field(0.0, ge=0.0, le=1.0)
    """Probability of dropping a path during training."""

    scale_by_keep: bool = Field(True)
    """ Up-scales activations during training by 1 - drop_prob to avoid train-test mismatch. Defaults to True."""
