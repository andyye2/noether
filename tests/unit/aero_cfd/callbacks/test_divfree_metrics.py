#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from contextlib import nullcontext
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest
import torch
from pydantic import ValidationError

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RECIPE_SRC = _REPO_ROOT / "recipes" / "aero_cfd" / "src"
if str(_RECIPE_SRC) not in sys.path:
    sys.path.insert(0, str(_RECIPE_SRC))

from aero_cfd.callbacks.divfree_metrics import (  # noqa: E402
    DivFreeMetricsCallback,
    DivFreeMetricsCallbackConfig,
    _scale_vector,
)


class _IdentityDataset:
    pipeline = None

    def __len__(self) -> int:
        return 1

    def denormalize(self, field: str, tensor: torch.Tensor) -> torch.Tensor:
        assert field == "volume_velocity"
        return tensor


class _DataContainer:
    def __init__(self) -> None:
        self.dataset = _IdentityDataset()

    def get_dataset(self, key=None, properties=None, max_size=None):
        return self.dataset


class _FieldModel(torch.nn.Module):
    def __init__(
        self,
        *,
        mode: str = "linear_x",
        delta: float = 1e-3,
        position_scale: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        super().__init__()
        self.mode = mode
        self.delta = delta
        self.register_buffer("position_scale", torch.tensor(position_scale, dtype=torch.float32))
        self.query_shape = None

    def forward(self, **kwargs):
        query = kwargs["query_volume_position"]
        self.query_shape = query.shape

        if self.mode == "linear_x":
            zeros = torch.zeros_like(query[..., 0])
            velocity = torch.stack([query[..., 0], zeros, zeros], dim=-1)
        elif self.mode == "zero_div":
            zeros = torch.zeros_like(query[..., 0])
            velocity = torch.stack([zeros, torch.sin(query[..., 2]), zeros], dim=-1)
        elif self.mode == "missing":
            return {}
        else:
            raise ValueError(self.mode)

        return {"query_volume_velocity": velocity}


def _batch(num_anchors: int = 16) -> dict[str, torch.Tensor]:
    anchor = torch.linspace(-0.5, 0.5, steps=num_anchors * 3, dtype=torch.float32)
    return {"volume_anchor_position": anchor.reshape(1, num_anchors, 3)}


def _callback(config: DivFreeMetricsCallbackConfig | None = None) -> DivFreeMetricsCallback:
    if config is None:
        config = DivFreeMetricsCallbackConfig(
            dataset_key="test",
            every_n_epochs=1,
            forward_properties=["volume_anchor_position"],
        )

    callback = object.__new__(DivFreeMetricsCallback)
    callback._config = config
    callback.dataset_key = config.dataset_key
    callback.forward_properties = config.forward_properties
    callback.num_monitor_anchors = config.num_monitor_anchors
    callback.field = config.monitor_dataset_field
    callback._mismatch_warned = False
    callback.trainer = SimpleNamespace(autocast_context=nullcontext())
    callback.data_container = _DataContainer()
    callback._logger = Mock()
    callback.writer = Mock()
    return callback


def test_config_defaults_are_valid() -> None:
    config = DivFreeMetricsCallbackConfig(dataset_key="test", every_n_epochs=1)

    assert config.delta is None
    assert config.position_scale is None
    assert config.num_monitor_anchors == 256
    assert config.max_samples == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 2},
        {"delta": 0.0},
        {"num_monitor_anchors": 0},
        {"max_samples": 0},
        {"position_scale": [0.1, 0.2]},
        {"position_scale": [0.1, 0.0, 0.3]},
        {"position_scale": -0.5},
    ],
)
def test_config_validation(kwargs) -> None:
    with pytest.raises(ValidationError):
        DivFreeMetricsCallbackConfig(dataset_key="test", every_n_epochs=1, **kwargs)


def test_scale_vector_accepts_scalar_and_vector() -> None:
    assert _scale_vector(0.5).shape == (3,)
    assert _scale_vector([0.5]).shape == (3,)
    assert _scale_vector([0.1, 0.2, 0.3]).shape == (3,)
    assert _scale_vector(torch.tensor([0.1, 0.2, 0.3])).dtype == torch.float32


def test_linear_field_matches_expected_divergence() -> None:
    callback = _callback()
    model = _FieldModel(position_scale=(0.5, 0.5, 0.5))

    result = callback.process_data(_batch(), trainer_model=model)

    assert result["divergence_abs_mean"].item() == pytest.approx(2.0, abs=1e-4)
    assert result["divergence_abs_max"].item() == pytest.approx(2.0, abs=1e-4)
    assert result["divergence_square_mean"].item() == pytest.approx(4.0, abs=1e-4)


def test_zero_divergence_field_is_near_zero() -> None:
    callback = _callback()
    model = _FieldModel(mode="zero_div")

    result = callback.process_data(_batch(), trainer_model=model)

    assert result["divergence_abs_mean"].item() < 1e-3
    assert result["divergence_abs_max"].item() < 1e-3


def test_anchor_subset_controls_query_shape() -> None:
    config = DivFreeMetricsCallbackConfig(
        dataset_key="test",
        every_n_epochs=1,
        forward_properties=["volume_anchor_position"],
        num_monitor_anchors=10,
    )
    callback = _callback(config)
    model = _FieldModel()

    callback.process_data(_batch(num_anchors=100), trainer_model=model)

    assert model.query_shape == (1, 60, 3)


def test_delta_defaults_to_model(monkeypatch) -> None:
    module = import_module("aero_cfd.callbacks.divfree_metrics")
    original = module.central_difference_positions
    seen = {}

    def spy(position, delta):
        seen["delta"] = delta
        return original(position, delta)

    monkeypatch.setattr(module, "central_difference_positions", spy)

    callback = _callback()
    model = _FieldModel(delta=5e-4)

    callback.process_data(_batch(), trainer_model=model)

    assert seen["delta"] == pytest.approx(5e-4)


def test_position_scale_defaults_to_model(monkeypatch) -> None:
    module = import_module("aero_cfd.callbacks.divfree_metrics")
    seen = {}

    def spy(field, delta, position_scale):
        seen["same_object"] = position_scale is model.position_scale
        return torch.zeros(field.shape[0], field.shape[1] // 6, device=field.device)

    monkeypatch.setattr(module, "divergence_from_perturbed_field", spy)

    callback = _callback()
    model = _FieldModel()

    callback.process_data(_batch(), trainer_model=model)

    assert seen["same_object"]


def test_position_scale_override_warns_once() -> None:
    config = DivFreeMetricsCallbackConfig(
        dataset_key="test",
        every_n_epochs=1,
        forward_properties=["volume_anchor_position"],
        position_scale=0.25,
    )
    callback = _callback(config)
    model = _FieldModel(position_scale=(0.5, 0.5, 0.5))

    callback.process_data(_batch(), trainer_model=model)
    callback.process_data(_batch(), trainer_model=model)

    callback.logger.warning.assert_called_once()


def test_missing_query_volume_velocity_raises() -> None:
    callback = _callback()
    model = _FieldModel(mode="missing")

    with pytest.raises(KeyError):
        callback.process_data(_batch(), trainer_model=model)


def test_process_results_logs_public_metrics() -> None:
    callback = _callback()
    results = {
        "divergence_abs_mean": torch.tensor([1.0, 3.0]),
        "divergence_abs_max": torch.tensor([2.0, 5.0]),
        "divergence_square_mean": torch.tensor([4.0, 16.0]),
    }

    callback.process_results(results, interval_type=None, update_counter=None)

    logged = {call.kwargs["key"]: call.kwargs["value"] for call in callback.writer.add_scalar.call_args_list}

    assert logged["divergence/test/divergence_abs_mean"].item() == pytest.approx(2.0)
    assert logged["divergence/test/divergence_abs_max"].item() == pytest.approx(5.0)
    assert logged["divergence/test/divergence_rms"].item() == pytest.approx(10.0**0.5)
