#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import nn


class ParamGroupModifierBase:
    """Generic implementation to change properties of optimizer parameter groups."""

    @abstractmethod
    def get_properties(self, model: nn.Module, name: str, param: torch.Tensor) -> dict[str, float]:
        """Returns the modified properties for a given model parameter. This method is called with all items of
        `model.named_parameters()` to compose the parameter groups for the whole model.

        Args:
            model: Model from which the parameter originates from. Used to extract properties (e.g., number of layers
                for a layerwise learning rate decay).
            name: Name of the parameter as stored inside the model.
            param: The parameter tensor.
        """
        raise NotImplementedError

    def __repr__(self):
        return str(self)

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def was_applied_successfully(self) -> bool:
        """Checks if the parameter group modifier was applied successfully."""
        raise NotImplementedError
