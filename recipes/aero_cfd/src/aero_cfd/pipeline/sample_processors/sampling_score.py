#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import math
import pickle
from pathlib import Path
from typing import Any

import torch

from noether.data import SampleProcessor


def make_volume_position_fingerprint(position: torch.Tensor, num_samples: int = 32) -> dict[str, Any]:
    position = position.detach().cpu()
    position64 = position.to(dtype=torch.float64)
    num_points = len(position) if position.ndim > 0 else 0

    if num_points == 0:
        sample_indices = torch.empty(0, dtype=torch.long)
    else:
        num_samples = min(num_samples, num_points)
        sample_indices = torch.linspace(0, num_points - 1, steps=num_samples).round().to(dtype=torch.long)

    if num_points == 0:
        sample_values = []
    else:
        sample_values = position64.index_select(0, sample_indices).reshape(-1).tolist()

    return {
        "dtype": str(position.dtype),
        "shape": list(position.shape),
        "num_points": num_points,
        "sample_indices": sample_indices.tolist(),
        "sample_values": sample_values,
        "sum": float(position64.sum().item()),
        "sum_of_squares": float(position64.square().sum().item()),
    }


class LoadSamplingScoreSampleProcessor(SampleProcessor):
    """Loads per-point sampling scores from sidecar files if they match the current sample."""

    def __init__(
        self,
        score_dir: str | Path,
        score_key: str = "volume_sampling_score",
        position_key: str = "volume_position",
        index_key: str = "index",
        fingerprint_key: str = "volume_position_fingerprint",
    ):
        self.score_dir = Path(score_dir)
        self.score_key = score_key
        self.position_key = position_key
        self.index_key = index_key
        self.fingerprint_key = fingerprint_key

    def __call__(self, input_sample: dict[str, Any]) -> dict[str, Any]:
        output_sample = self.save_copy(input_sample)

        index = self._to_int(output_sample.get(self.index_key))
        position = output_sample.get(self.position_key)
        if index is None or not torch.is_tensor(position) or position.ndim == 0:
            return output_sample

        sidecar_path = self.score_dir / f"score_{index}.pt"
        if not sidecar_path.exists():
            return output_sample

        try:
            sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=True)
        except (EOFError, OSError, RuntimeError, ValueError, pickle.UnpicklingError):
            return output_sample

        if not isinstance(sidecar, dict):
            return output_sample
        if self._to_int(sidecar.get(self.index_key)) != index:
            return output_sample

        score = sidecar.get(self.score_key)
        if (
            not torch.is_tensor(score)
            or score.ndim == 0
            or len(score) != len(position)
            or score.numel() != len(position)
        ):
            return output_sample

        expected_fingerprint = sidecar.get(self.fingerprint_key)
        current_fingerprint = make_volume_position_fingerprint(position)
        if not self._fingerprints_match(expected_fingerprint, current_fingerprint):
            return output_sample

        output_sample[self.score_key] = score.detach().reshape(-1).to(dtype=torch.float32)

        return output_sample

    @staticmethod
    def _to_int(value: Any) -> int | None:
        if torch.is_tensor(value):
            if value.numel() != 1:
                return None
            value = value.item()
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _fingerprints_match(expected: Any, current: dict[str, Any]) -> bool:
        if not isinstance(expected, dict):
            return False

        for key in ("dtype", "shape", "num_points", "sample_indices"):
            if expected.get(key) != current.get(key):
                return False

        if not LoadSamplingScoreSampleProcessor._float_lists_close(
            expected.get("sample_values"), current.get("sample_values")
        ):
            return False

        return LoadSamplingScoreSampleProcessor._float_close(
            expected.get("sum"), current.get("sum")
        ) and LoadSamplingScoreSampleProcessor._float_close(
            expected.get("sum_of_squares"), current.get("sum_of_squares")
        )

    @staticmethod
    def _float_close(left: Any, right: Any) -> bool:
        try:
            return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-8)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _float_lists_close(left: Any, right: Any) -> bool:
        if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
            return False
        return all(LoadSamplingScoreSampleProcessor._float_close(a, b) for a, b in zip(left, right, strict=True))
