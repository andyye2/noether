# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Tests for the volume wall-distance input feature.

The feature adds an input the ShapeNet-Car source checkpoint never saw, so the
tests here pin the three things that make such an arm interpretable: the
evaluation metric is defined over exactly the same points as before, the model
gains exactly the tensors the transfer loader knows to instantiate, and the
feature's unit is fixed rather than implied.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from aero_cfd.datasets.transfer_drivaerml import (
    WALL_DISTANCE_FILENAME,
    WALL_DISTANCE_PROPERTY,
    WALL_DISTANCE_REFERENCE_LENGTH_M,
    to_wall_distance_feature,
)
from aero_cfd.multi_fidelity.evaluation import EvaluationRequest, TargetModel, build_evaluation
from aero_cfd.multi_fidelity.experiment import (
    CHECKPOINT_ARCHITECTURE,
    FEATURE_PROJECTION_PATTERN,
    MODEL_KIND,
    RESET_PATTERN,
    GeometryRendering,
    SourceCheckpoint,
    TrainingRequest,
    build_experiment,
)
from aero_cfd.multi_fidelity.integrity import GitState, canonical_json_bytes as canonical_bytes, sha256_file
from aero_cfd.multi_fidelity.manifest import build_manifest, load_manifest_cell
from aero_cfd.multi_fidelity.protocol import ProtocolBinding
from aero_cfd.multi_fidelity.provenance import write_training_provenance
from aero_cfd.multi_fidelity.statistics import load_statistics_binding
from aero_cfd.pipeline.multistage_pipelines.aero_multistage import (
    AeroMultistagePipeline,
    _split_by_underscore,
    _split_three_or_none,
)
from aero_cfd.pipeline.sample_processors import AnchorPointSamplingSampleProcessor
from aero_cfd.presets.drivaerml_transfer import DrivAerMLTransferCommonPreset
from noether.modeling.models.aerodynamics import AeroABUPT
from research.multi_fidelity.tools.compute_subset_statistics import (
    FIELD_SPECS,
    build_statistics_artifact,
)

from .conftest import FIXTURE_GIT_STATE, write_statistics_artifact

SOURCE_ROOT = Path("/scratch/andyye2/ABUPT/outputs")

#: Moments of the log-scaled feature, in the units the dataset produces.
FEATURE_STATISTICS: dict[str, Any] = {
    f"{WALL_DISTANCE_PROPERTY}_logscale_mean": [4.0],
    f"{WALL_DISTANCE_PROPERTY}_logscale_std": [2.0],
}


def _preset(
    tmp_path: Path,
    protocol: ProtocolBinding,
    manifest_path: Path,
    *,
    wall_distance_feature: bool,
) -> DrivAerMLTransferCommonPreset:
    """Build the common-field transfer preset with or without the feature."""
    statistics = write_statistics_artifact(
        tmp_path / f"statistics-wd{int(wall_distance_feature)}.json",
        protocol=protocol,
        manifest_raw_sha256=sha256_file(manifest_path),
        train_subset_size=25,
        overrides=FEATURE_STATISTICS,
    )
    return DrivAerMLTransferCommonPreset(
        statistics_artifact=statistics,
        coordinate_frame="shapenet",
        expected_manifest_sha256=sha256_file(manifest_path),
        expected_train_subset_size=25,
        wall_distance_feature=wall_distance_feature,
    )


def _backbone_keys(preset: DrivAerMLTransferCommonPreset) -> set[str]:
    """Return the state-dict keys of the model this preset configures."""
    model_config = preset.build_model(model_kind=MODEL_KIND, **CHECKPOINT_ARCHITECTURE)
    return set(AeroABUPT(model_config=model_config).state_dict())


def test_the_feature_adds_exactly_the_projection_the_loader_instantiates(
    tmp_path: Path,
    protocol: ProtocolBinding,
    manifest_path: Path,
) -> None:
    """Enabling the feature must change the model in one confined way.

    If any other tensor appeared, moved, or changed shape, the transfer load
    would stop being strict and the arm would silently differ from its pair by
    more than the feature.
    """
    without = _backbone_keys(_preset(tmp_path, protocol, manifest_path, wall_distance_feature=False))
    with_feature = _backbone_keys(_preset(tmp_path, protocol, manifest_path, wall_distance_feature=True))

    added = with_feature - without
    assert without - with_feature == set()
    assert added == {
        f"backbone.domain_feature_projs.volume.mlp.{index}.{parameter}"
        for index in (0, 2)
        for parameter in ("weight", "bias")
    }
    assert all(key.startswith(FEATURE_PROJECTION_PATTERN) for key in added)
    # No surface projection: a surface point's distance to the surface is zero.
    assert not any("domain_feature_projs.surface" in key for key in with_feature)


def test_the_prediction_actually_depends_on_the_feature(
    tmp_path: Path,
    protocol: ProtocolBinding,
    manifest_path: Path,
) -> None:
    """A feature that is collated but never read would be invisible otherwise.

    Nothing else in the pipeline fails if ``volume_anchor_features`` is dropped
    on the way to the model: the run trains, the metrics export, and the arm
    just silently equals its no-feature pair. So the check is behavioural --
    change only the feature and require the volume prediction to move.
    """
    preset = _preset(tmp_path, protocol, manifest_path, wall_distance_feature=True)
    model = AeroABUPT(model_config=preset.build_model(model_kind=MODEL_KIND, **CHECKPOINT_ARCHITECTURE))
    model.eval()

    generator = torch.Generator().manual_seed(11)
    anchors = 32
    inputs: dict[str, torch.Tensor] = {
        "geometry_position": torch.rand(128, 3, generator=generator) * 40.0,
        "geometry_supernode_idx": torch.arange(16),
        "geometry_batch_idx": torch.zeros(128, dtype=torch.long),
        "surface_anchor_position": torch.rand(1, anchors, 3, generator=generator) * 40.0,
        "volume_anchor_position": torch.rand(1, anchors, 3, generator=generator) * 40.0,
    }
    near_wall = torch.full((1, anchors, 1), -2.0)
    far_field = torch.full((1, anchors, 1), 2.0)

    with torch.no_grad():
        near = model(**inputs, volume_anchor_features=near_wall)["volume_velocity"]
        far = model(**inputs, volume_anchor_features=far_field)["volume_velocity"]

    assert near.shape == (1, anchors, 3)
    assert not torch.allclose(near, far)


def test_the_transfer_initializer_instantiates_only_what_is_missing() -> None:
    """The readout is removed and rebuilt; the projection is only rebuilt.

    Removing a pattern the checkpoint does not contain would be a no-op that
    reads as if the source had been trained with the feature.
    """
    source = SourceCheckpoint(output_path=SOURCE_ROOT)

    baseline = source.initializer()
    assert baseline.patterns_to_remove == [RESET_PATTERN]
    assert baseline.patterns_to_instantiate == [RESET_PATTERN]

    with_feature = source.initializer(instantiate_feature_projections=True)
    assert with_feature.patterns_to_remove == [RESET_PATTERN]
    assert with_feature.patterns_to_instantiate == [RESET_PATTERN, FEATURE_PROJECTION_PATTERN]


def test_the_extra_sampled_item_does_not_move_the_evaluation_points() -> None:
    """The metric must stay defined over exactly the points it was before.

    ``AnchorPointSamplingSampleProcessor`` seeds its permutation from the
    sample index and draws it from whichever item the set happens to yield
    first, so adding an item is only safe because every item is a per-point
    array of the same length. This asserts that rather than assuming it.
    """
    generator = torch.Generator().manual_seed(0)
    sample = {
        "index": 3,
        "volume_position": torch.rand(64, 3, generator=generator),
        "volume_velocity": torch.rand(64, 3, generator=generator),
        WALL_DISTANCE_PROPERTY: torch.rand(64, 1, generator=generator),
    }
    arguments: dict[str, Any] = {
        "num_points": 16,
        "keep_queries": False,
        "to_prefix_and_postfix": _split_by_underscore,
        "to_prefix_midfix_postfix": _split_three_or_none,
        "seed": 4242,
    }
    sampled = {"volume_position", "volume_velocity"}

    without = AnchorPointSamplingSampleProcessor(items=set(sampled), **arguments)(sample)
    with_feature = AnchorPointSamplingSampleProcessor(items=sampled | {WALL_DISTANCE_PROPERTY}, **arguments)(sample)

    assert torch.equal(without["volume_anchor_position"], with_feature["volume_anchor_position"])
    assert torch.equal(without["volume_anchor_velocity"], with_feature["volume_anchor_velocity"])
    assert with_feature["volume_anchor_distance"].shape == (16, 1)


def test_a_single_domain_feature_builds_a_pipeline(
    tmp_path: Path,
    protocol: ProtocolBinding,
    manifest_path: Path,
) -> None:
    """Declaring features on the volume alone must not build an empty concatenation.

    ``torch.cat`` rejects an empty list, so an unconditional per-domain
    processor would abort the first batch of every feature run.
    """
    preset = _preset(tmp_path, protocol, manifest_path, wall_distance_feature=True)
    pipeline = AeroMultistagePipeline(pipeline_config=preset.build_pipeline(MODEL_KIND))

    assert "volume_anchor_features" in pipeline.default_collator_items
    assert "surface_anchor_features" not in pipeline.default_collator_items
    concatenations = {
        processor.target_key: list(processor.items)
        for processor in pipeline.sample_processors
        if hasattr(processor, "target_key")
    }
    assert concatenations["volume_anchor_features"] == ["volume_anchor_distance"]
    assert "surface_anchor_features" not in concatenations


def test_the_feature_reaches_the_model_through_forward_properties(
    tmp_path: Path,
    protocol: ProtocolBinding,
    manifest_path: Path,
) -> None:
    """A collated feature that is not a forward property never reaches the model."""
    with_feature = _preset(tmp_path, protocol, manifest_path, wall_distance_feature=True)
    without = _preset(tmp_path, protocol, manifest_path, wall_distance_feature=False)

    assert "volume_anchor_features" in with_feature.forward_properties(MODEL_KIND)
    assert "surface_anchor_features" not in with_feature.forward_properties(MODEL_KIND)
    assert "volume_anchor_features" not in without.forward_properties(MODEL_KIND)


def test_the_feature_is_normalized_in_reference_lengths_on_a_log_scale(
    tmp_path: Path,
    protocol: ProtocolBinding,
    manifest_path: Path,
) -> None:
    """The unit and the log scale are the whole point of the feature.

    ``log1p`` is nearly the identity below its knee, so in metres the near-wall
    band -- where the velocity gradient is -- would be compressed into a
    fraction of a percent of the feature's range and the projection would have
    nothing to read.
    """
    preset = _preset(tmp_path, protocol, manifest_path, wall_distance_feature=True)
    normalizer = preset.build_normalizers()[WALL_DISTANCE_PROPERTY][0]

    assert normalizer.logscale is True
    assert normalizer.mean.tolist() == FEATURE_STATISTICS[f"{WALL_DISTANCE_PROPERTY}_logscale_mean"]
    assert normalizer.std.tolist() == FEATURE_STATISTICS[f"{WALL_DISTANCE_PROPERTY}_logscale_std"]

    metres = torch.tensor([0.0, 1e-4, 2e-2, 80.0])
    assert torch.allclose(to_wall_distance_feature(metres), metres / WALL_DISTANCE_REFERENCE_LENGTH_M)
    # 70% of DrivAerML cells lie within 20 mm. In metres they would occupy
    # under 1% of the log-scaled range; in reference lengths, about a third.
    scaled = torch.log1p(to_wall_distance_feature(metres))
    assert float(scaled[2] / scaled[3]) > 0.25
    assert float(torch.log1p(metres)[2] / torch.log1p(metres)[3]) < 0.01


def test_disabling_the_feature_reproduces_the_previous_arm(
    tmp_path: Path,
    protocol: ProtocolBinding,
    manifest_path: Path,
) -> None:
    """An arm without the feature must be byte-identical to the audited one.

    Otherwise the existing results would stop being reproducible by this code.
    """
    preset = _preset(tmp_path, protocol, manifest_path, wall_distance_feature=False)

    assert WALL_DISTANCE_PROPERTY in preset.excluded_properties
    assert WALL_DISTANCE_PROPERTY not in preset.normalizer_spec
    assert preset.data_specs.use_physics_features is False
    assert preset.data_specs.all_features == set()
    assert preset.pipeline_params(MODEL_KIND)["use_physics_features"] is False


def test_the_feature_marks_the_run_id(
    tmp_path: Path,
    protocol_path: Path,
    manifest_path: Path,
    statistics_path: Path,
) -> None:
    """Two arms that read different inputs cannot share a run identifier."""
    arguments: dict[str, Any] = {
        "dataset_root": Path("/scratch/andyye2/data/drivaerml_subsampled_10x"),
        "protocol_path": protocol_path,
        "manifest_path": manifest_path,
        "statistics_path": statistics_path,
        "output_path": tmp_path / "outputs",
        "task": "common",
        "strategy": "scratch",
        "sample_size": 25,
        "replicate": 0,
        "model_seed": 7103,
        "source": SourceCheckpoint(output_path=SOURCE_ROOT),
    }
    cell = load_manifest_cell(manifest_path, 25)

    without = TrainingRequest(**arguments).derive_run_id(cell)
    with_feature = TrainingRequest(**arguments, wall_distance_feature=True).derive_run_id(cell)

    assert not without.endswith("-wd")
    assert with_feature == f"{without}-wd"


def test_the_evaluator_must_render_the_feature_the_model_was_trained_with(
    tmp_path: Path,
    protocol: ProtocolBinding,
    protocol_path: Path,
    manifest_path: Path,
) -> None:
    """A feature-trained checkpoint cannot be scored without the feature.

    The restored model carries a feature projection. Scoring it through a
    pipeline that never produces the input would silently evaluate a different
    model than the one that was trained, and the resulting row would be filed
    under the same arm.
    """
    statistics_path = write_statistics_artifact(
        tmp_path / "statistics.json",
        protocol=protocol,
        manifest_raw_sha256=sha256_file(manifest_path),
        train_subset_size=25,
        overrides=FEATURE_STATISTICS,
    )
    request = TrainingRequest(
        dataset_root=Path("/scratch/andyye2/data/drivaerml_subsampled_10x"),
        protocol_path=protocol_path,
        manifest_path=manifest_path,
        statistics_path=statistics_path,
        output_path=tmp_path / "outputs",
        task="common",
        strategy="scratch",
        sample_size=25,
        replicate=0,
        model_seed=7103,
        source=SourceCheckpoint(output_path=SOURCE_ROOT),
        geometry=GeometryRendering(supernode_radius=0.1),
        wall_distance_feature=True,
    )
    experiment = build_experiment(request, load_manifest_cell(manifest_path, 25), protocol)
    checkpoints = request.output_path / experiment.audit["run_id"] / "train/checkpoints"
    checkpoints.mkdir(parents=True)
    latest = checkpoints / "ab_upt_cp=latest_model.th"
    latest.write_bytes(b"construction-only-checkpoint")
    write_training_provenance(
        request,
        experiment.audit,
        experiment.config.model_dump(mode="json", exclude_computed_fields=True),
        FIXTURE_GIT_STATE,
    )

    def _evaluation(wall_distance_feature: bool) -> EvaluationRequest:
        return EvaluationRequest(
            dataset_root=request.dataset_root,
            protocol_path=protocol_path,
            manifest_path=manifest_path,
            statistics_path=statistics_path,
            target=TargetModel(
                output_path=request.output_path,
                run_id=str(experiment.audit["run_id"]),
                expected_sha256=sha256_file(latest),
            ),
            task="common",
            method="S",
            replicate=0,
            sample_size=25,
            budget="compute_matched",
            eval_output_path=tmp_path / "eval",
            output_csv=tmp_path / "metrics.csv",
            geometry=GeometryRendering(supernode_radius=0.1),
            wall_distance_feature=wall_distance_feature,
        )

    def _build(evaluation: EvaluationRequest) -> tuple[Any, dict[str, Any]]:
        return build_evaluation(
            evaluation,
            protocol=protocol,
            manifest_cell=load_manifest_cell(manifest_path, 25),
            statistics=load_statistics_binding(statistics_path, protocol=protocol),
            evaluator_git_state=GitState(commit="e" * 40, dirty=False, status_sha256="f" * 64),
        )

    _, audit = _build(_evaluation(True))
    assert audit["wall_distance_feature"] is True
    assert audit["input_feature"]["reference_length_m"] == WALL_DISTANCE_REFERENCE_LENGTH_M

    with pytest.raises(ValueError, match="training provenance mismatch for wall_distance_feature"):
        _build(_evaluation(False))


def _write_run(root: Path, run_id: int, offset: float) -> None:
    """Write one synthetic run directory with a target and the feature file."""
    run_dir = root / f"run_{run_id}"
    run_dir.mkdir(parents=True)
    torch.save(torch.tensor([[1.0 + offset, 2.0 + offset, 3.0 + offset]]), run_dir / "volume_cell_velocity.pt")
    torch.save(torch.tensor([1e-3 + offset, 2e-3 + offset]), run_dir / WALL_DISTANCE_FILENAME)


def _write_manifest(path: Path, sizes: tuple[int, ...]) -> list[int]:
    """Write a minimal hashed manifest and return its largest train prefix."""
    manifest = build_manifest(seed=9, sizes=sizes)
    manifest.pop("manifest_sha256")
    manifest["study_id"] = "unit-test-study"
    manifest["protocol_sha256"] = "a" * 64
    manifest["implementation_git_commit"] = "b" * 40
    manifest["implementation_git_dirty"] = True
    manifest["val_run_ids"] = list(manifest["official_val_run_ids"])
    manifest["test_run_ids"] = list(manifest["official_test_run_ids"])
    manifest["nested_train_run_ids"] = {str(size): list(manifest["subsets"][str(size)]["run_ids"]) for size in sizes}
    manifest["manifest_sha256"] = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    path.write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return list(manifest["nested_train_run_ids"][str(sizes[-1])])


def test_fitting_the_feature_leaves_the_target_statistics_untouched(tmp_path: Path) -> None:
    """One artifact must serve both arms of a comparison.

    The moments are accumulated per field, so adding the feature cannot move a
    target statistic. Asserting it is what allows the feature and no-feature
    arms to share a statistics artifact, which is what makes them paired.
    """
    manifest_path = tmp_path / "manifest.json"
    run_ids = _write_manifest(manifest_path, (2,))
    data_root = tmp_path / "data"
    _write_run(data_root, run_ids[0], 0.0)
    _write_run(data_root, run_ids[1], 3.0)

    arguments: dict[str, Any] = {
        "root": data_root,
        "manifest_path": manifest_path,
        "n": 2,
        "coordinate_frame": "shapenet",
        "chunk_rows": 1,
        "position_min": -40.0,
        "position_max": 80.0,
    }
    targets_only, targets_flat = build_statistics_artifact(fields=["volume_velocity"], **arguments)
    both, both_flat = build_statistics_artifact(fields=["volume_velocity", WALL_DISTANCE_PROPERTY], **arguments)

    assert both_flat["volume_velocity_mean"] == targets_flat["volume_velocity_mean"]
    assert both_flat["volume_velocity_std"] == targets_flat["volume_velocity_std"]
    assert both["field_statistics"]["volume_velocity"] == targets_only["field_statistics"]["volume_velocity"]
    assert both["leakage_guard"]["target_splits_read"] == ["train"]

    # The recorded transform must state the unit, and the moments must be the
    # moments of exactly that transform of exactly those values.
    details = both["field_statistics"][WALL_DISTANCE_PROPERTY]
    assert details["filename"] == WALL_DISTANCE_FILENAME
    assert details["transform"] == "abs(x) / 0.001 m, then sign(x)*log1p(abs(x))"
    values = torch.log1p(
        to_wall_distance_feature(torch.tensor([1e-3, 2e-3, 3.001, 3.002], dtype=torch.float64)),
    )
    assert both_flat[f"{WALL_DISTANCE_PROPERTY}_logscale_mean"] == pytest.approx([float(values.mean())])


def test_the_statistics_tool_reuses_the_dataset_transform() -> None:
    """The fitted moments and the training values must not drift apart.

    Two independent copies of the unit conversion would be invisible until the
    normalization was silently wrong, so the tool is required to call the
    dataset's own function.
    """
    assert FIELD_SPECS[WALL_DISTANCE_PROPERTY].rescale is to_wall_distance_feature
    assert FIELD_SPECS[WALL_DISTANCE_PROPERTY].log_scale is True
    assert FIELD_SPECS[WALL_DISTANCE_PROPERTY].filename == WALL_DISTANCE_FILENAME
