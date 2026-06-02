#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import einops
import torch
from torch import nn

from noether.modeling.modules.layers import TransformerBatchNorm


def test_equal_to_bn1d_for_bs1():
    dim = 8
    bn1 = nn.BatchNorm1d(dim)
    bn2 = nn.BatchNorm1d(dim)
    tbn = TransformerBatchNorm(dim)
    x_bn1 = torch.randn(1, dim, 5)
    x_bn2 = einops.rearrange(x_bn1, "bs dim seqlen -> (bs seqlen) dim")
    x_tbn = einops.rearrange(x_bn1, "bs dim seqlen -> bs seqlen dim")
    y_bn1 = bn1(x_bn1)
    y_bn2 = bn2(x_bn2)
    y_bn2 = einops.rearrange(y_bn2, "(bs seqlen) dim -> bs dim seqlen", bs=len(x_bn1))
    y_tbn = tbn(x_tbn)
    y_tbn = einops.rearrange(y_tbn, "bs seqlen dim -> bs dim seqlen")
    assert torch.equal(y_bn1, y_tbn)
    assert torch.allclose(y_bn2, y_tbn, atol=1e-7)


def test_equal_to_bn1d_for_bs2():
    dim = 8
    bn = nn.BatchNorm1d(dim)
    tbn = TransformerBatchNorm(dim)
    x_bn = torch.randn(2, dim, 5)
    x_tbn = einops.rearrange(x_bn, "bs dim seqlen -> bs seqlen dim")
    y_bn = torch.concat([bn(x_bn[i].unsqueeze(0)) for i in range(len(x_bn))])
    y_tbn = tbn(x_tbn)
    y_tbn = einops.rearrange(y_tbn, "bs seqlen dim -> bs dim seqlen")
    assert torch.allclose(y_bn, y_tbn, atol=1e-7)
