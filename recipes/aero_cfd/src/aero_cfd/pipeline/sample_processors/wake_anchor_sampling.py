#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from aero_cfd.utils.aero_regions import compute_wake_mask
from noether.data import SampleProcessor


class WakeAwareAnchorPointSamplingSampleProcessor(SampleProcessor):
    """Subsample anchor points with a fixed wake quota before random fill-in."""

    def __init__(
        self,
        items: set[str],
        num_points: int,
        to_prefix_and_postfix: Callable[[str], tuple[str, str]],
        to_prefix_midfix_postfix: Callable[[str], tuple[str, str, str]],
        raw_pos_min: float | Sequence[float] | torch.Tensor,
        raw_pos_max: float | Sequence[float] | torch.Tensor,
        wake_fraction: float,
        wake_box_lwh: tuple[float, float, float] = (0.47, 0.43, 0.31),
        wake_axes: tuple[int, int, int] = (0, 1, 2),
        reference_item: str = "surface_position",
        position_item: str = "volume_position",
        position_scale: float = 1000.0,
        zero_center: bool = False,
        keep_queries: bool = False,
        seed: int | None = None,
    ):
        """
        Args:
            items: Pointcloud items sampled with the same indices.
            num_points: Number of anchor points to sample.
            raw_pos_min: Raw-coordinate minimum used by the position normalizer.
            raw_pos_max: Raw-coordinate maximum used by the position normalizer.
            wake_fraction: Requested fraction of anchors drawn from the wake box. Values <= 0 use random sampling.
            wake_box_lwh: Wake-box dimensions normalized by body length, from Aultman & Duan 2024.
            wake_axes: Coordinate axes in streamwise, spanwise, and road-normal order.
            reference_item: Surface position key used to infer the per-sample body bounding box.
            position_item: Volume position key sampled by this processor.
            seed: Random seed for deterministic sampling for evaluation. Default None.
        """
        if not num_points >= 0:
            raise ValueError("Number of points to sample must be non-negative.")
        if wake_fraction > 1.0:
            raise ValueError("wake_fraction must be <= 1.0.")

        self.items = items
        self.num_points = num_points
        self.keep_queries = keep_queries
        self.to_prefix_and_postfix = to_prefix_and_postfix
        self.to_prefix_midfix_postfix = to_prefix_midfix_postfix
        self.raw_pos_min = raw_pos_min
        self.raw_pos_max = raw_pos_max
        self.wake_fraction = wake_fraction
        self.wake_box_lwh = wake_box_lwh
        self.wake_axes = wake_axes
        self.reference_item = reference_item
        self.position_item = position_item
        self.position_scale = position_scale
        self.zero_center = zero_center
        self.seed = seed

    def _get_generator(self, output_sample: dict[str, Any]) -> torch.Generator | None:
        if self.seed is None:
            return None
        if "index" not in output_sample:
            raise ValueError("Sample index is required for deterministic point sampling with a seed.")
        seed = output_sample["index"] + self.seed
        return torch.Generator().manual_seed(seed)

    def _random_permutation(
        self,
        num_candidates: int,
        *,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.keep_queries:
            perm = torch.randperm(num_candidates, generator=generator)
            discarded_perm = None if num_candidates <= self.num_points else perm[self.num_points :]
            return perm[: self.num_points], discarded_perm

        return torch.randperm(num_candidates, generator=generator)[: self.num_points], None

    def _wake_aware_permutation(
        self,
        output_sample: dict[str, Any],
        *,
        num_candidates: int,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.wake_fraction <= 0.0:
            return self._random_permutation(num_candidates, generator=generator)
        if self.reference_item not in output_sample:
            raise KeyError(f"Missing wake reference item '{self.reference_item}'.")
        if self.position_item not in output_sample:
            raise KeyError(f"Missing wake position item '{self.position_item}'.")

        global_order = torch.randperm(num_candidates, generator=generator)
        if num_candidates <= self.num_points:
            discarded_perm = None if self.keep_queries else None
            return global_order[: self.num_points], discarded_perm

        wake_mask = compute_wake_mask(
            surface_position=output_sample[self.reference_item],
            volume_position=output_sample[self.position_item],
            raw_pos_min=self.raw_pos_min,
            raw_pos_max=self.raw_pos_max,
            box_lwh=self.wake_box_lwh,
            axes=self.wake_axes,
            position_scale=self.position_scale,
            zero_center=self.zero_center,
        )
        if wake_mask.shape[0] != num_candidates:
            raise ValueError(
                f"Wake mask length {wake_mask.shape[0]} does not match sampled tensor length {num_candidates}."
            )

        requested_wake = round(self.num_points * self.wake_fraction)
        wake_order = global_order[wake_mask[global_order]]
        wake_perm = wake_order[: min(requested_wake, len(wake_order))]

        selected = torch.zeros(num_candidates, dtype=torch.bool)
        selected[wake_perm] = True
        fill_order = global_order[~selected[global_order]]
        perm = torch.cat([wake_perm, fill_order[: self.num_points - len(wake_perm)]])

        discarded_perm = None
        if self.keep_queries:
            discarded_perm = fill_order[self.num_points - len(wake_perm) :]

        return perm, discarded_perm

    def __call__(self, input_sample: dict[str, Any]) -> dict[str, Any]:
        """Subsample the configured pointcloud items with a shared permutation."""
        output_sample = self.save_copy(input_sample)

        any_item = next(iter(self.items))
        first_item_tensor = output_sample[any_item]
        assert torch.is_tensor(first_item_tensor)

        generator = self._get_generator(output_sample)
        perm, discarded_perm = self._wake_aware_permutation(
            output_sample,
            num_candidates=len(first_item_tensor),
            generator=generator,
        )

        for item in self.items:
            tensor = output_sample[item]
            prefix, postfix = self.to_prefix_and_postfix(item)
            output_sample[f"{prefix}_anchor_{postfix}"] = tensor[perm]
            if discarded_perm is not None:
                output_sample[f"{prefix}_query_{postfix}"] = tensor[discarded_perm]

        return output_sample
