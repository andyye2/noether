"""Sketch for the first noether adaptive-anchor sampler.

This is intentionally not imported by the project yet. It documents the smallest
change that should replace uniform volume-anchor sampling with score-weighted
sampling while keeping the original pipeline contract.
"""

import torch


def sample_score_weighted_indices(
    *,
    num_points: int,
    score: torch.Tensor | None,
    uniform_fraction: float = 0.3,
    gamma: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return point indices using uniform fallback or mixed score sampling."""
    if score is None:
        return torch.randperm(num_points)[:num_points]

    score = score.flatten().to(dtype=torch.float32)
    if score.numel() != num_points or not torch.isfinite(score).all():
        return torch.randperm(num_points)[:num_points]

    adaptive = torch.clamp(score, min=0.0).pow(gamma)
    adaptive = adaptive + eps
    adaptive = adaptive / adaptive.sum()

    uniform = torch.full_like(adaptive, 1.0 / adaptive.numel())
    prob = uniform_fraction * uniform + (1.0 - uniform_fraction) * adaptive

    return torch.multinomial(prob, num_points, replacement=False)


def apply_to_noether_sample(
    sample: dict,
    *,
    input_prefix: str = "volume",
    output_prefix: str = "volume_anchor",
    num_anchor_points: int = 256,
    score_key: str = "volume_sampling_score",
    uniform_fraction: float = 0.3,
    gamma: float = 1.0,
) -> dict:
    """Mirror AnchorPointSamplingSampleProcessor with score-weighted ids."""
    position_key = f"{input_prefix}_position"
    num_candidates = sample[position_key].shape[0]
    num_selected = min(num_anchor_points, num_candidates)

    score = sample.get(score_key)
    ids = sample_score_weighted_indices(
        num_points=num_candidates,
        score=score,
        uniform_fraction=uniform_fraction,
        gamma=gamma,
    )[:num_selected]

    for key, value in list(sample.items()):
        if key.startswith(f"{input_prefix}_") and value.shape[0] == num_candidates:
            suffix = key.removeprefix(f"{input_prefix}_")
            sample[f"{output_prefix}_{suffix}"] = value[ids]

    return sample

