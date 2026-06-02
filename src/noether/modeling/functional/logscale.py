#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch


def to_logscale(x: torch.Tensor) -> torch.Tensor:
    """Turns a tensor into log scale. Log is the natural logarithm of x + 1 .

    Args:
        x: Tensor to be transformed

    Returns:
        Tensor in log scale
    """
    return torch.sign(x) * torch.log1p(x.abs())


def from_logscale(x: torch.Tensor) -> torch.Tensor:
    """Turns a tensor from log scale into orginal scale.
        x = from_logscale(to_logscale(x))

    Args:
        x: Tensor to be de-transformed from log scale. Expected to be in natural logarithm + 1.

    Returns:
        Tensor in orginal scale
    """
    return torch.sign(x) * (x.abs().exp() - 1)
