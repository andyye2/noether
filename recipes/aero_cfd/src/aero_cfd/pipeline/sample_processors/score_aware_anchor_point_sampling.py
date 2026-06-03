#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections.abc import Callable
from typing import Any

import torch

from noether.data import SampleProcessor


class ScoreAwareAnchorPointSamplingSampleProcessor(SampleProcessor):
    """Subsamples anchor points from a pointcloud using per-point sampling scores."""

    def __init__(
        self,
        items: set[str],
        num_points: int,
        to_prefix_and_postfix: Callable[[str], tuple[str, str]],
        to_prefix_midfix_postfix: Callable[[str], tuple[str, str, str]],
        keep_queries: bool = False,
        seed: int | None = None,
        score_key: str = "volume_sampling_score",
        uniform_fraction: float = 0.3,
        gamma: float = 1.0,
        eps: float = 1e-8,
        prob_key: str = "volume_anchor_sampling_prob",
        weight_key: str = "volume_anchor_sampling_weight",
    ):
        if not num_points >= 0:
            raise ValueError("Number of points to sample must be non-negative.")
        if not 0.0 <= uniform_fraction <= 1.0:
            raise ValueError("Uniform fraction must be between 0 and 1.")
        if not gamma > 0.0:
            raise ValueError("Gamma must be positive.")
        if not eps > 0.0:
            raise ValueError("Eps must be positive.")

        self.items = items
        self.num_points = num_points
        self.keep_queries = keep_queries
        self.to_prefix_and_postfix = to_prefix_and_postfix
        self.to_prefix_midfix_postfix = to_prefix_midfix_postfix
        self.seed = seed
        self.score_key = score_key
        self.uniform_fraction = uniform_fraction
        self.gamma = gamma
        self.eps = eps
        self.prob_key = prob_key
        self.weight_key = weight_key

    def __call__(self, input_sample: dict[str, Any]) -> dict[str, Any]:
        """Subsamples pointcloud items with identical indices and writes anchor outputs."""

        output_sample = self.save_copy(input_sample)

        any_item = next(iter(self.items))
        first_item_tensor = output_sample[any_item]
        assert torch.is_tensor(first_item_tensor)

        generator = self._make_generator(output_sample)
        perm, discarded_perm, anchor_prob, anchor_weight = self._sample_indices(
            output_sample, first_item_tensor, generator
        )

        for item in self.items:
            tensor = output_sample[item]
            prefix, postfix = self.to_prefix_and_postfix(item)
            output_sample[f"{prefix}_anchor_{postfix}"] = tensor[perm]
            if discarded_perm is not None:
                output_sample[f"{prefix}_query_{postfix}"] = tensor[discarded_perm]

        output_sample[self.prob_key] = anchor_prob
        output_sample[self.weight_key] = anchor_weight

        return output_sample

    def _make_generator(self, output_sample: dict[str, Any]) -> torch.Generator | None:
        if self.seed is None:
            return None
        if "index" not in output_sample:
            raise ValueError("Sample index is required for deterministic point sampling with a seed.")
        seed = int(output_sample["index"]) + self.seed
        return torch.Generator().manual_seed(seed)

    def _sample_indices(
        self,
        output_sample: dict[str, Any],
        first_item_tensor: torch.Tensor,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        num_total_points = len(first_item_tensor)
        prob = self._get_sampling_prob(output_sample, num_total_points)

        if prob is None or self.num_points >= num_total_points:
            return self._random_indices(num_total_points, generator)

        perm = torch.multinomial(prob, self.num_points, replacement=False, generator=generator)

        if self.keep_queries:
            keep = torch.ones(num_total_points, dtype=torch.bool)
            keep[perm] = False
            discarded_perm = torch.arange(num_total_points)[keep]
        else:
            discarded_perm = None

        anchor_prob = prob[perm]
        anchor_weight = 1.0 / (num_total_points * anchor_prob.clamp_min(self.eps))

        return perm, discarded_perm, anchor_prob, anchor_weight

    def _random_indices(
        self,
        num_total_points: int,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        perm = torch.randperm(num_total_points, generator=generator)

        if self.keep_queries:
            if num_total_points <= self.num_points:
                discarded_perm = None
            else:
                discarded_perm = perm[self.num_points :]
            perm = perm[: self.num_points]
        else:
            perm = perm[: self.num_points]
            discarded_perm = None

        if num_total_points == 0:
            anchor_prob = torch.ones(len(perm))
        else:
            anchor_prob = torch.full((len(perm),), 1.0 / num_total_points)
        anchor_weight = torch.ones(len(perm))

        return perm, discarded_perm, anchor_prob, anchor_weight

    def _get_sampling_prob(self, output_sample: dict[str, Any], num_total_points: int) -> torch.Tensor | None:
        score = output_sample.get(self.score_key)
        if (
            not torch.is_tensor(score)
            or score.ndim == 0
            or len(score) != num_total_points
            or score.numel() != num_total_points
        ):
            return None

        score = score.detach().reshape(-1).to(dtype=torch.float32)
        if not torch.isfinite(score).all():
            return None

        score = torch.clamp(score, min=0.0)
        if not torch.any(score > 0.0):
            return None

        adaptive = score.pow(self.gamma) + self.eps
        adaptive_sum = adaptive.sum()
        if not torch.isfinite(adaptive_sum) or adaptive_sum <= 0.0:
            return None

        adaptive = adaptive / adaptive_sum
        uniform = torch.full_like(adaptive, 1.0 / num_total_points)
        prob = self.uniform_fraction * uniform + (1.0 - self.uniform_fraction) * adaptive

        prob_sum = prob.sum()
        if not torch.isfinite(prob_sum) or prob_sum <= 0.0:
            return None

        return prob / prob_sum
