#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import einops
import torch
import torch.nn.functional as F
from torch import nn


class TransformerBatchNorm(nn.Module):
    """Wrapper around `torch.nn.BatchNorm1d` that considers all tokens of a single sample as the full batch.
    Additionally remaps `affine` to `elementwise_affine` and supports disabling bias to comply with the
    `torch.nn.LayerNorm` interface. Does not use any nn.BatchNorm1d modules to avoid errors with nn.SyncBatchnorm.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight: nn.Parameter | None = nn.Parameter(torch.ones(num_features))
        else:
            self.weight = None
        if bias:
            assert elementwise_affine
            self.bias: nn.Parameter | None = nn.Parameter(torch.zeros(num_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """BatchNorm1d where all tokens of a single sample correspond to a full batch.

        Args:
            x: Tensor of shape (batch_size, seqlen, dim).

        Returns:
            Normalized x of shape (batch_size, seqlen, dim).
        """
        if len(x) == 1:
            # fast implementation via kernel
            assert x.ndim == 3
            x = einops.rearrange(x, "bs seqlen dim -> bs dim seqlen")
            x = F.batch_norm(
                x,
                weight=self.weight,
                bias=self.bias,
                eps=self.eps,
                running_mean=None,
                running_var=None,
                training=True,
            )
            x = einops.rearrange(x, "bs dim seqlen -> bs seqlen dim")
        else:
            # slow native implementation
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, keepdim=True, unbiased=False) + self.eps
            x = (x - mean) / var.sqrt()
            if self.weight is not None:
                x = x * einops.rearrange(self.weight, "dim -> 1 1 dim")
            if self.bias is not None:
                x = x + einops.rearrange(self.bias, "dim -> 1 1 dim")
        return x
