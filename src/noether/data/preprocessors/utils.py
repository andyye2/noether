#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections.abc import Sequence

import torch

from noether.data.preprocessors.types import ScalarOrSequence


def to_tensor(data: ScalarOrSequence) -> torch.Tensor:
    """
    Helper function to convert input data to a PyTorch tensor if it is not already one.

    Args:
        data: The input data to convert. Can be a sequence of floats, a torch.Tensor, or None.

    Returns: The input data as a torch.Tensor if it was a sequence, the original tensor if it was already a torch.Tensor, or None if the input was None.
    Raises:
        TypeError: If the input data is of an unsupported type.
    """
    if isinstance(data, torch.Tensor):
        return data

    if isinstance(data, Sequence | float | int):
        try:
            return torch.tensor(data)
        except ValueError as e:
            raise ValueError(
                f"Failed to convert {data} to a tensor: {e} since dimensions are not compatible or data is wrong."
            ) from e
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Data must be a Sequence, torch.Tensor, or None.")
