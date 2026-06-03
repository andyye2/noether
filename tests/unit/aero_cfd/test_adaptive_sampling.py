#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from aero_cfd.callbacks import VolumeResidualScoreRefreshCallback, VolumeResidualScoreRefreshCallbackConfig
from aero_cfd.pipeline.multistage_pipelines.aero_multistage import AeroCFDPipelineConfig, AeroMultistagePipeline
from aero_cfd.pipeline.sample_processors import (
    AnchorPointSamplingSampleProcessor,
    LoadSamplingScoreSampleProcessor,
    ScoreAwareAnchorPointSamplingSampleProcessor,
    make_volume_position_fingerprint,
)
from aero_cfd.presets import ShapeNetCarPreset
from aero_cfd.trainers import SamplingWeightedLossTrainer, SamplingWeightedLossTrainerConfig
from noether.core.providers import MetricPropertyProvider, PathProvider
from noether.core.schemas.dataset import DomainDataSpec, FieldDimSpec, ModelDataSpecs
from noether.core.schemas.trainers import WeightedLossTrainerConfig
from noether.core.trackers import NoopTracker
from noether.training.trainers import WeightedLossTrainer

_ABUPT_MODEL_KIND = "noether.modeling.models.aerodynamics.AeroABUPT"
_TRANSFORMER_MODEL_KIND = "noether.modeling.models.aerodynamics.AeroTransformer"
_TRAINER_KIND = "noether.training.trainers.WeightedLossTrainer"
_FIELD_WEIGHTS = {"surface_pressure": 1.0, "volume_velocity": 1.0}


def _split_by_underscore(item: str) -> list[str]:
    return item.split("_")


def _split_three_or_none(item: str) -> list[str | None]:
    parts = item.split("_")
    return parts if len(parts) == 3 else [None] * 3


def _sample(num_points: int = 8) -> dict[str, torch.Tensor | int]:
    position = torch.arange(num_points * 3, dtype=torch.float32).reshape(num_points, 3)
    return {
        "index": 4,
        "volume_position": position,
        "volume_velocity": position + 1000.0,
    }


def _anchor_indices(processed_sample: dict[str, torch.Tensor]) -> torch.Tensor:
    return (processed_sample["volume_anchor_position"][:, 0] / 3).to(dtype=torch.long)


def _make_score_sampler(**kwargs) -> ScoreAwareAnchorPointSamplingSampleProcessor:
    params = dict(
        items={"volume_position", "volume_velocity"},
        num_points=3,
        to_prefix_and_postfix=_split_by_underscore,
        to_prefix_midfix_postfix=_split_three_or_none,
        seed=11,
    )
    params.update(kwargs)
    return ScoreAwareAnchorPointSamplingSampleProcessor(**params)


def _make_random_sampler(**kwargs) -> AnchorPointSamplingSampleProcessor:
    params = dict(
        items={"volume_position", "volume_velocity"},
        num_points=3,
        to_prefix_and_postfix=_split_by_underscore,
        to_prefix_midfix_postfix=_split_three_or_none,
        seed=11,
    )
    params.update(kwargs)
    return AnchorPointSamplingSampleProcessor(**params)


def test_score_aware_anchor_sampling_records_true_probabilities_and_ipw_weights() -> None:
    sample = _sample(num_points=8)
    score = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    sample["volume_sampling_score"] = score

    processor = _make_score_sampler(num_points=4, uniform_fraction=0.25, gamma=1.0, eps=1e-6)
    processed = processor(sample)
    indices = _anchor_indices(processed)

    adaptive = score.clamp_min(0.0).pow(1.0) + 1e-6
    adaptive = adaptive / adaptive.sum()
    uniform = torch.full_like(adaptive, 1.0 / len(score))
    expected_prob = 0.25 * uniform + 0.75 * adaptive
    expected_prob = expected_prob / expected_prob.sum()

    assert torch.allclose(processed["volume_anchor_sampling_prob"], expected_prob[indices])
    assert torch.allclose(processed["volume_anchor_sampling_weight"], 1.0 / (len(score) * expected_prob[indices]))
    assert torch.all(processed["volume_anchor_velocity"] - processed["volume_anchor_position"] == 1000.0)


def test_score_aware_anchor_sampling_prefers_high_score_points() -> None:
    sample = _sample(num_points=8)
    sample["volume_sampling_score"] = torch.tensor([0.0, 0.0, 0.0, 1.0e20, 0.0, 0.0, 0.0, 0.0])

    processed = _make_score_sampler(num_points=1, uniform_fraction=0.0)(sample)

    assert _anchor_indices(processed).tolist() == [3]


def test_score_aware_anchor_sampling_fallback_matches_random_sampler_without_scores() -> None:
    sample = _sample(num_points=8)

    score_processed = _make_score_sampler()(sample)
    random_processed = _make_random_sampler()(sample)

    assert torch.equal(score_processed["volume_anchor_position"], random_processed["volume_anchor_position"])
    assert torch.equal(score_processed["volume_anchor_velocity"], random_processed["volume_anchor_velocity"])
    assert torch.allclose(score_processed["volume_anchor_sampling_prob"], torch.full((3,), 1.0 / 8.0))
    assert torch.equal(score_processed["volume_anchor_sampling_weight"], torch.ones(3))


@pytest.mark.parametrize(
    "score",
    [
        torch.ones(7),
        torch.tensor([0.0, 1.0, float("nan"), 1.0, 1.0, 1.0, 1.0, 1.0]),
        torch.zeros(8),
    ],
)
def test_score_aware_anchor_sampling_invalid_scores_fallback_to_random(score: torch.Tensor) -> None:
    sample = _sample(num_points=8)
    sample["volume_sampling_score"] = score

    score_processed = _make_score_sampler()(sample)
    random_processed = _make_random_sampler()(sample)

    assert torch.equal(score_processed["volume_anchor_position"], random_processed["volume_anchor_position"])
    assert torch.equal(score_processed["volume_anchor_velocity"], random_processed["volume_anchor_velocity"])


def test_score_aware_anchor_sampling_keep_queries_partitions_points() -> None:
    sample = _sample(num_points=8)
    sample["volume_sampling_score"] = torch.arange(1, 9, dtype=torch.float32)

    processed = _make_score_sampler(num_points=3, keep_queries=True)(sample)

    anchor_indices = _anchor_indices(processed)
    query_indices = (processed["volume_query_position"][:, 0] / 3).to(dtype=torch.long)

    assert set(anchor_indices.tolist()).isdisjoint(set(query_indices.tolist()))
    assert sorted(torch.cat([anchor_indices, query_indices]).tolist()) == list(range(8))


def test_load_sampling_score_loads_matching_sidecar(tmp_path: Path) -> None:
    sample = _sample(num_points=5)
    score = torch.linspace(0.0, 1.0, steps=5)
    torch.save(
        {
            "index": sample["index"],
            "volume_sampling_score": score,
            "volume_position_fingerprint": make_volume_position_fingerprint(sample["volume_position"]),
        },
        tmp_path / "score_4.pt",
    )

    processed = LoadSamplingScoreSampleProcessor(score_dir=tmp_path)(sample)

    assert torch.equal(processed["volume_sampling_score"], score)


def test_load_sampling_score_ignores_mismatched_fingerprint(tmp_path: Path) -> None:
    sample = _sample(num_points=5)
    torch.save(
        {
            "index": sample["index"],
            "volume_sampling_score": torch.ones(5),
            "volume_position_fingerprint": make_volume_position_fingerprint(sample["volume_position"] + 1.0),
        },
        tmp_path / "score_4.pt",
    )

    processed = LoadSamplingScoreSampleProcessor(score_dir=tmp_path)(sample)

    assert "volume_sampling_score" not in processed


def test_load_sampling_score_ignores_missing_sidecar(tmp_path: Path) -> None:
    processed = LoadSamplingScoreSampleProcessor(score_dir=tmp_path)(_sample(num_points=5))

    assert "volume_sampling_score" not in processed


def test_load_sampling_score_ignores_sidecar_without_score_key(tmp_path: Path) -> None:
    sample = _sample(num_points=5)
    torch.save(
        {
            "index": sample["index"],
            "volume_position_fingerprint": make_volume_position_fingerprint(sample["volume_position"]),
        },
        tmp_path / "score_4.pt",
    )

    processed = LoadSamplingScoreSampleProcessor(score_dir=tmp_path)(sample)

    assert "volume_sampling_score" not in processed


def _make_trainer_for_loss() -> SamplingWeightedLossTrainer:
    trainer = object.__new__(SamplingWeightedLossTrainer)
    trainer._loss_fn = F.mse_loss
    trainer.loss_items = [("volume_velocity", 1.0)]
    trainer.sample_weight_keys = {"volume_velocity": "volume_anchor_sampling_weight"}
    return trainer


def test_sampling_weighted_loss_applies_per_point_weights() -> None:
    trainer = _make_trainer_for_loss()
    forward_output = {"volume_velocity": torch.tensor([[[1.0], [0.0]]])}
    targets = {
        "volume_velocity_target": torch.tensor([[[0.0], [2.0]]]),
        "volume_anchor_sampling_weight": torch.tensor([[0.5, 2.0]]),
    }

    losses = trainer.loss_compute(forward_output, targets)

    assert torch.allclose(losses["volume_velocity_loss"], torch.tensor(4.25))


def test_sampling_weighted_loss_requires_configured_weight() -> None:
    trainer = _make_trainer_for_loss()
    forward_output = {"volume_velocity": torch.zeros(1, 2, 1)}
    targets = {"volume_velocity_target": torch.zeros(1, 2, 1)}

    with pytest.raises(ValueError, match="Sample weight"):
        trainer.loss_compute(forward_output, targets)


class _TinyDataset:
    def __len__(self) -> int:
        return 4

    def sample_info(self, index: int) -> dict[str, str]:
        return {"run_name": f"run_{index}"}


class _TinyDataContainer:
    def __init__(self):
        self.dataset = _TinyDataset()

    def get_dataset(self, *_args, **_kwargs) -> _TinyDataset:
        return self.dataset


def _trainer_kwargs(tmp_path: Path) -> dict:
    metric_property_provider = MetricPropertyProvider()
    path_provider = PathProvider(output_root_path=tmp_path, run_id="unit")
    tracker = NoopTracker(metric_property_provider=metric_property_provider, path_provider=path_provider)
    return {
        "data_container": _TinyDataContainer(),
        "device": "cpu",
        "tracker": tracker,
        "path_provider": path_provider,
        "metric_property_provider": metric_property_provider,
    }


def _weighted_loss_config(config_cls, **kwargs):
    params = dict(
        max_epochs=1,
        effective_batch_size=1,
        callbacks=[],
        add_default_callbacks=False,
        add_trainer_callbacks=False,
        forward_properties=["volume_anchor_position"],
        target_properties=["volume_velocity_target"],
        field_weights={"volume_velocity": 1.0},
    )
    params.update(kwargs)
    return config_cls(**params)


def test_sampling_weighted_loss_empty_weight_mapping_matches_base_trainer(tmp_path: Path) -> None:
    kwargs = _trainer_kwargs(tmp_path)
    base_trainer = WeightedLossTrainer(
        _weighted_loss_config(
            WeightedLossTrainerConfig,
            kind="noether.training.trainers.WeightedLossTrainer",
        ),
        **kwargs,
    )
    sampling_trainer = SamplingWeightedLossTrainer(
        _weighted_loss_config(SamplingWeightedLossTrainerConfig),
        **kwargs,
    )
    forward_output = {"volume_velocity": torch.tensor([[[1.0], [0.0]]])}
    targets = {"volume_velocity_target": torch.tensor([[[0.0], [2.0]]])}

    assert torch.allclose(
        sampling_trainer.loss_compute(forward_output, targets)["volume_velocity_loss"],
        base_trainer.loss_compute(forward_output, targets)["volume_velocity_loss"],
    )


def test_sampling_weighted_loss_unit_weights_match_mean_mse(tmp_path: Path) -> None:
    trainer = SamplingWeightedLossTrainer(
        _weighted_loss_config(
            SamplingWeightedLossTrainerConfig,
            sample_weight_keys={"volume_velocity": "volume_anchor_sampling_weight"},
        ),
        **_trainer_kwargs(tmp_path),
    )
    forward_output = {"volume_velocity": torch.tensor([[[1.0], [0.0]]])}
    targets = {
        "volume_velocity_target": torch.tensor([[[0.0], [2.0]]]),
        "volume_anchor_sampling_weight": torch.ones(1, 2),
    }

    loss = trainer.loss_compute(forward_output, targets)["volume_velocity_loss"]

    assert torch.allclose(loss, F.mse_loss(targets["volume_velocity_target"], forward_output["volume_velocity"]))


def test_sampling_weighted_loss_split_batch_routes_weight_to_targets(tmp_path: Path) -> None:
    trainer = SamplingWeightedLossTrainer(
        _weighted_loss_config(
            SamplingWeightedLossTrainerConfig,
            sample_weight_keys={"volume_velocity": "volume_anchor_sampling_weight"},
        ),
        **_trainer_kwargs(tmp_path),
    )
    batch = {
        "volume_anchor_position": torch.zeros(1, 2, 3),
        "volume_velocity_target": torch.zeros(1, 2, 1),
        "volume_anchor_sampling_weight": torch.ones(1, 2),
    }

    forward_batch, targets_batch = trainer._split_batch(batch)

    assert set(forward_batch) == {"volume_anchor_position"}
    assert set(targets_batch) == {"volume_velocity_target", "volume_anchor_sampling_weight"}
    assert "volume_anchor_sampling_weight" in trainer.batch_keys


def test_sampling_weighted_loss_rejects_unknown_weight_field(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="without a configured loss"):
        SamplingWeightedLossTrainer(
            _weighted_loss_config(
                SamplingWeightedLossTrainerConfig,
                sample_weight_keys={"surface_pressure": "surface_anchor_sampling_weight"},
            ),
            **_trainer_kwargs(tmp_path),
        )


def _data_specs() -> ModelDataSpecs:
    return ModelDataSpecs(
        position_dim=3,
        use_physics_features=False,
        domains={
            "surface": DomainDataSpec(output_dims=FieldDimSpec(root=OrderedDict({"pressure": 1}))),
            "volume": DomainDataSpec(output_dims=FieldDimSpec(root=OrderedDict({"velocity": 3}))),
        },
    )


def _make_pipeline(**kwargs) -> AeroMultistagePipeline:
    params = dict(
        num_surface_points=0,
        num_volume_points=0,
        num_surface_queries=0,
        num_volume_queries=0,
        use_physics_features=False,
        sample_query_points=True,
        num_supernodes=0,
        num_geometry_supernodes=2,
        num_geometry_points=4,
        num_volume_anchor_points=3,
        num_surface_anchor_points=2,
        seed=5,
        data_specs=_data_specs(),
    )
    params.update(kwargs)
    return AeroMultistagePipeline(AeroCFDPipelineConfig(**params))


def _default_collator_items(pipeline: AeroMultistagePipeline) -> list[str]:
    return pipeline.collators[0].items


def test_aero_pipeline_uses_random_anchor_sampling_by_default() -> None:
    pipeline = _make_pipeline()

    assert not any(isinstance(processor, ScoreAwareAnchorPointSamplingSampleProcessor) for processor in pipeline.sample_processors)
    assert sum(isinstance(processor, AnchorPointSamplingSampleProcessor) for processor in pipeline.sample_processors) == 2
    assert "volume_anchor_sampling_prob" not in _default_collator_items(pipeline)
    assert "volume_anchor_sampling_weight" not in _default_collator_items(pipeline)


def test_aero_pipeline_replaces_only_volume_anchor_sampling_when_enabled(tmp_path: Path) -> None:
    pipeline = _make_pipeline(use_volume_score_sampling=True, volume_score_dir=str(tmp_path))

    score_processors = [
        processor
        for processor in pipeline.sample_processors
        if isinstance(processor, ScoreAwareAnchorPointSamplingSampleProcessor)
    ]
    random_anchor_processors = [
        processor for processor in pipeline.sample_processors if isinstance(processor, AnchorPointSamplingSampleProcessor)
    ]

    assert len(score_processors) == 1
    assert score_processors[0].items == {"volume_position", "volume_velocity"}
    assert len(random_anchor_processors) == 1
    assert random_anchor_processors[0].items == {"surface_position", "surface_pressure"}
    assert any(isinstance(processor, LoadSamplingScoreSampleProcessor) for processor in pipeline.sample_processors)
    assert "volume_anchor_sampling_prob" in _default_collator_items(pipeline)
    assert "volume_anchor_sampling_weight" in _default_collator_items(pipeline)


def test_aero_pipeline_emits_score_refresh_candidates_without_score_sampling(tmp_path: Path) -> None:
    pipeline = _make_pipeline(emit_volume_score_candidates=True, volume_score_dir=str(tmp_path))

    assert not any(isinstance(processor, ScoreAwareAnchorPointSamplingSampleProcessor) for processor in pipeline.sample_processors)
    assert "volume_score_position" in _default_collator_items(pipeline)
    assert "volume_score_velocity" in _default_collator_items(pipeline)


def _build_shapenet_config(tmp_path: Path, **kwargs):
    params = dict(
        model_kind=_ABUPT_MODEL_KIND,
        model_params={
            "hidden_dim": 96,
            "geometry_depth": 1,
            "physics_blocks": ["perceiver"],
            "num_domain_decoder_blocks": {"surface": 1, "volume": 1},
        },
        trainer_kind=_TRAINER_KIND,
        trainer_params={"field_weights": _FIELD_WEIGHTS},
        dataset_root=str(tmp_path / "dataset"),
        output_path=str(tmp_path / "out"),
        accelerator="cpu",
        max_epochs=1,
        batch_size=1,
        datasets=["train", "test"],
        include_evaluation=False,
    )
    params.update(kwargs)
    return ShapeNetCarPreset().build_config(**params)


def test_shapenet_preset_default_keeps_existing_trainer_and_uniform_sampling(tmp_path: Path) -> None:
    config = _build_shapenet_config(tmp_path)

    assert config.trainer.kind == _TRAINER_KIND
    assert config.datasets["train"].pipeline.use_volume_score_sampling is False
    assert config.datasets["test"].pipeline.use_volume_score_sampling is False


def test_shapenet_preset_enables_score_sampling_only_for_train(tmp_path: Path) -> None:
    config = _build_shapenet_config(
        tmp_path,
        use_volume_score_sampling=True,
        volume_score_dir=str(tmp_path / "scores"),
        volume_score_refresh_every_n_epochs=2,
    )

    assert config.trainer.kind == "aero_cfd.trainers.SamplingWeightedLossTrainer"
    assert config.trainer.sample_weight_keys == {"volume_velocity": "volume_anchor_sampling_weight"}
    assert config.datasets["train"].pipeline.use_volume_score_sampling is True
    assert config.datasets["test"].pipeline.use_volume_score_sampling is False
    assert config.datasets["score_refresh_train"].pipeline.emit_volume_score_candidates is True
    assert config.datasets["score_refresh_train"].pipeline.use_volume_score_sampling is False


def test_shapenet_preset_rejects_score_sampling_for_non_anchor_pipeline(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="anchor-point AB-UPT"):
        _build_shapenet_config(
            tmp_path,
            model_kind=_TRANSFORMER_MODEL_KIND,
            model_params={"hidden_dim": 96, "depth": 2},
            use_volume_score_sampling=True,
            volume_score_dir=str(tmp_path / "scores"),
        )


class _FakeTrainer:
    autocast_context = nullcontext()


class _QueryEchoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_lengths: list[int] = []
        self.anchor_contexts: list[torch.Tensor] = []

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        query_position = kwargs["query_volume_position"]
        self.query_lengths.append(query_position.shape[1])
        self.anchor_contexts.append(kwargs["volume_anchor_position"].detach().clone())
        return {"query_volume_velocity": query_position}


def _score_refresh_callback(tmp_path: Path, model: nn.Module, query_chunk_size: int = 2):
    metric_property_provider = MetricPropertyProvider()
    return VolumeResidualScoreRefreshCallback(
        callback_config=VolumeResidualScoreRefreshCallbackConfig(
            score_dir=str(tmp_path),
            dataset_key="score_refresh_train",
            every_n_epochs=1,
            batch_size=1,
            query_chunk_size=query_chunk_size,
        ),
        trainer=_FakeTrainer(),
        model=model,
        data_container=_TinyDataContainer(),
        tracker=object(),
        log_writer=object(),
        checkpoint_writer=object(),
        metric_property_provider=metric_property_provider,
        development=True,
    )


def _score_refresh_batch(num_points: int = 5) -> dict[str, torch.Tensor]:
    position = torch.arange(num_points * 3, dtype=torch.float32).reshape(1, num_points, 3) / 10.0
    point_offsets = torch.arange(num_points, dtype=torch.float32).reshape(1, num_points, 1)
    return {
        "index": torch.tensor([7]),
        "geometry_position": torch.zeros(1, 4, 3),
        "geometry_supernode_idx": torch.zeros(1, 4, dtype=torch.long),
        "geometry_batch_idx": torch.zeros(1, 4, dtype=torch.long),
        "surface_anchor_position": torch.zeros(1, 2, 3),
        "volume_anchor_position": torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
        "volume_score_position": position,
        "volume_score_velocity": position + point_offsets,
    }


def test_volume_residual_score_refresh_writes_chunked_scores_in_order(tmp_path: Path) -> None:
    model = _QueryEchoModel()
    callback = _score_refresh_callback(tmp_path, model=model, query_chunk_size=2)

    result = callback.process_data(_score_refresh_batch(num_points=5))

    sidecar = torch.load(tmp_path / "score_7.pt", map_location="cpu", weights_only=True)
    expected_score = torch.arange(5, dtype=torch.float32).square()
    assert model.query_lengths == [2, 2, 1]
    assert torch.equal(sidecar["volume_sampling_score"], expected_score)
    assert sidecar["run_name"] == "run_7"
    assert torch.equal(result["num_written"], torch.tensor(1))
    assert torch.equal(result["num_points"], torch.tensor(5))


def test_volume_residual_score_refresh_keeps_anchor_context_across_chunks(tmp_path: Path) -> None:
    model = _QueryEchoModel()
    callback = _score_refresh_callback(tmp_path, model=model, query_chunk_size=2)
    batch = _score_refresh_batch(num_points=5)

    callback.process_data(batch)

    assert len(model.anchor_contexts) == 3
    for anchor_context in model.anchor_contexts:
        assert torch.equal(anchor_context, batch["volume_anchor_position"])


def test_volume_residual_score_refresh_overwrites_existing_sidecar(tmp_path: Path) -> None:
    torch.save({"index": 7, "volume_sampling_score": torch.full((5,), -1.0)}, tmp_path / "score_7.pt")
    model = _QueryEchoModel()
    callback = _score_refresh_callback(tmp_path, model=model, query_chunk_size=10)

    callback.process_data(_score_refresh_batch(num_points=5))

    sidecar = torch.load(tmp_path / "score_7.pt", map_location="cpu", weights_only=True)
    assert torch.equal(sidecar["volume_sampling_score"], torch.arange(5, dtype=torch.float32).square())
