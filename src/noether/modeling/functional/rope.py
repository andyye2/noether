#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import einops
import torch


def _rope_polar(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """RoPE via polar coordinate rotations."""
    # adapted from https://github.com/meta-llama/llama3/blob/main/llama/model.py#L65
    assert torch.is_tensor(freqs) and torch.is_complex(freqs)
    assert x.ndim == 4, "x.shape should be (batch_size, num_heads, seqlen, head_dim)"
    assert freqs.ndim == 3, "freqs.shape should be (batch_size, seqlen, head_dim // 2)"
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # add dim for num_heads
    freqs = einops.rearrange(freqs, "batch_size seqlen head_dim -> batch_size 1 seqlen head_dim")
    x_out = torch.view_as_real(x_ * freqs).flatten(start_dim=3)
    return x_out.type_as(x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    assert x.size(-1) % 2 == 0
    x1, x2 = x.chunk(chunks=2, dim=-1)
    return torch.concat([-x2, x1], dim=-1)


def _rope_real(x: torch.Tensor, freqs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Applies RoPE by iterating over all dimension and rotating them with sine/cosine waves. This version is
    inefficient due to loops/splits/concats/pads and is only here for backward compatibility. Use complex version with
    polar coordinates via RopeFrequency(implementation="complex").
    """
    assert x.ndim == 4
    assert isinstance(freqs, tuple)
    head_dim = x.size(-1)
    freqs_dim_sum = sum(freqs[i].size(-1) for i in range(len(freqs)))
    assert freqs_dim_sum <= head_dim, (
        f"dimension of frequencies ({freqs_dim_sum}) > head dimension ({head_dim}) "
        "-> make sure you create frequencies with head_dim instead of dim"
    )
    n_freqs = len(freqs)
    split_size = freqs[0].size(-1)
    splits = x.split(split_size, dim=-1)
    pad = splits[n_freqs:]
    splits = splits[:n_freqs]
    # unsqueeze(1) adds dimension for heads
    rotated = tuple(
        (splits[i] * freqs[i].unsqueeze(1).cos()) + (_rotate_half(splits[i]) * freqs[i].unsqueeze(1).sin())
        for i in range(n_freqs)
    )
    x = torch.concat(rotated + pad, dim=-1)
    return x


def rope(x: torch.Tensor, freqs: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Applies Rotary Position Embeddings (RoPE)

    Args:
        x: Vector to rotate (e.g., queries or keys of a transformer). Shape=(batch_size, num_heads, seqlen, head_dim).
        freqs (torch.Tensor): Complex tensor of frequencies for rotating x.
        freqs (tuple): Sine/cosine frequencies for rotating x. For 1D, freqs is a tuple with length 1 with shape
        (batch_size, num_heads, num_dim_to_rotate) where num_dim_to_rotate is the number of dimensions to rotate.
        If positions are higher dimensional (e.g., 2D or 3D), freqs has multiple items (i.e., 2 or 3) where each
        corresponds to frequencies of the nth axis for rotation.

    Returns:
        Rotated x.
    """
    if isinstance(freqs, tuple):
        # LEGACY: simple implementation with real rotations. Kept for backward compatibility as reshaping is sometimes
        # necessary which requires iterating over the tuple which would break if it is a complex tensor
        return _rope_real(x=x, freqs=freqs)
    return _rope_polar(x=x, freqs=freqs)
