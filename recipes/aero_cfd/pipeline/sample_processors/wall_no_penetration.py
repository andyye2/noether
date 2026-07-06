#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

import torch

from noether.data import SampleProcessor


class WallNoPenetrationQuerySampleProcessor(SampleProcessor):
    """Create near-wall volume queries from sampled surface anchors."""

    def __init__(
        self,
        position_key: str = "surface_anchor_position",
        normal_key: str = "surface_anchor_normals",
        query_position_key: str = "query_volume_position",
        query_normal_key: str = "wall_query_normals",
        offset: float = 0.0,
    ):
        self.position_key = position_key
        self.normal_key = normal_key
        self.query_position_key = query_position_key
        self.query_normal_key = query_normal_key
        self.offset = offset

    def __call__(self, input_sample: dict[str, Any]) -> dict[str, Any]:
        output_sample = self.save_copy(input_sample)
        positions = output_sample[self.position_key]
        normals = output_sample[self.normal_key]
        assert torch.is_tensor(positions)
        assert torch.is_tensor(normals)

        normals = torch.nn.functional.normalize(normals, dim=-1)
        output_sample[self.query_position_key] = positions + self.offset * normals
        output_sample[self.query_normal_key] = normals
        return output_sample
