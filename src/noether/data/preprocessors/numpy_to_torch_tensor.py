#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import numpy as np
import torch

from noether.data.preprocessors import PreProcessor


class NumpyToTorchTensorPreProcessor(PreProcessor):
    """Convert numpy arrays to PyTorch tensors."""

    def __init__(self, normalization_key: str):
        super().__init__(normalization_key=normalization_key)

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        """Convert a numpy array to a PyTorch tensor."""
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        return torch.from_numpy(x)

    def denormalize(self, x: torch.Tensor) -> np.ndarray:
        """Convert a PyTorch tensor back to a numpy array."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        return x.detach().cpu().numpy()
