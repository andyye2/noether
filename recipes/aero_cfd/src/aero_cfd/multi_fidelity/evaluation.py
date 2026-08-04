# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Construction of one frozen single-split DrivAerML evaluation.

The evaluator never re-derives a scientific label.  Everything that describes
the trained model is read from its training provenance sidecar and must agree
with what the caller claims, so a metric row cannot be attributed to the wrong
cell, checkpoint, or geometry rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from aero_cfd.callbacks.paired_metrics_export import PairedMetricsExportCallbackConfig
from aero_cfd.presets.drivaerml_transfer import DrivAerMLTransferCommonPreset, DrivAerMLTransferFullPreset
from noether.core.schemas.initializers import PreviousRunInitializerConfig
from noether.data.datasets.cfd.caeml.drivaerml.split import DrivAerMLDefaultSplitIDs

from .experiment import (
    CHECKPOINT_ARCHITECTURE,
    FIELD_WEIGHTS,
    MODEL_KIND,
    MODEL_NAME,
    STAGE_NAME,
    TRAINER_KIND,
    Accelerator,
    CoordinateFrame,
    GeometryRendering,
    Precision,
    Task,
)
from .integrity import GitState, sha256_file
from .manifest import ManifestCell
from .protocol import ProtocolBinding
from .provenance import PROVENANCE_FILENAME, TrainingProvenance, load_training_provenance
from .statistics import StatisticsBinding

EvaluationSplit = Literal["val", "test"]

#: Anchor points per design at evaluation time.  The preregistration fixes the
#: dense output policy; training uses far fewer anchors per step.
DEFAULT_EVALUATION_POINTS = 16384
DEFAULT_GEOMETRY_POINTS = 16384
DEFAULT_GEOMETRY_SUPERNODES = 1024


@dataclass(frozen=True)
class TargetModel:
    """Location and expected identity of the trained model under evaluation.

    Attributes:
        output_path: Root of the training outputs.
        run_id: Training run identifier.
        stage_name: Training stage name.
        model_name: Model name used in checkpoint filenames.
        checkpoint_tag: Checkpoint tag to evaluate.
        expected_sha256: SHA256 the selected checkpoint file must have.
    """

    output_path: Path
    run_id: str
    expected_sha256: str
    stage_name: str = STAGE_NAME
    model_name: str = MODEL_NAME
    checkpoint_tag: str = "latest"

    @property
    def stage_path(self) -> Path:
        """Return the directory that holds checkpoints and the sidecar."""
        return self.output_path / self.run_id / self.stage_name

    @property
    def checkpoint_path(self) -> Path:
        """Return the path of the selected checkpoint file."""
        return self.stage_path / "checkpoints" / f"{self.model_name}_cp={self.checkpoint_tag}_model.th"

    @property
    def sidecar_path(self) -> Path:
        """Return the path of the training provenance sidecar."""
        return self.stage_path / PROVENANCE_FILENAME

    def initializer(self) -> PreviousRunInitializerConfig:
        """Build the initializer that restores the evaluated weights."""
        return PreviousRunInitializerConfig(
            output_path=self.output_path,
            run_id=self.run_id,
            stage_name=self.stage_name,
            model_name=self.model_name,
            checkpoint_tag=self.checkpoint_tag,
        )


@dataclass(frozen=True)
class EvaluationRequest:
    """One frozen evaluation of one trained checkpoint on one official split.

    Attributes:
        dataset_root: Root of the DrivAerML dataset.
        protocol_path: Path of the frozen preregistration.
        manifest_path: Path of the manifest the model was trained on.
        statistics_path: Path of the statistics artifact used at training time.
        target: The trained model under evaluation.
        task: Preregistered field set.
        method: Method code the caller claims, verified against the sidecar.
        replicate: Replicate label, verified against the sidecar.
        sample_size: Training-subset size, verified against the sidecar.
        budget: Budget name, verified against the sidecar.
        coordinate_frame: Frame, verified against the sidecar.
        geometry: Geometry rendering, verified against the sidecar.
        wall_distance_feature: Token-level input feature, verified against the
            sidecar. A model trained with it cannot be scored without it: the
            restored checkpoint carries a feature projection whose input the
            pipeline would then never produce.
        evaluation_split: Official split to score.
        confirm_test_release: Explicit consent required for the test split.
        eval_output_path: Root the evaluation run writes into.
        output_csv: Destination of the per-design metric table.
        eval_run_id: Explicit evaluation run ID, or ``None`` to derive it.
        num_geometry_points: Geometry points sampled per design.
        num_geometry_supernodes: Geometry supernodes per design.
        num_surface_queries: Surface output anchors per design.
        num_volume_queries: Volume output anchors per design.
        eval_point_seed: Seed of the deterministic evaluation sampling.
        precision: Inference precision.
        accelerator: Device kind.
        num_workers: Data-loader workers.
    """

    dataset_root: Path
    protocol_path: Path
    manifest_path: Path
    statistics_path: Path
    target: TargetModel
    task: Task
    method: str
    replicate: int
    sample_size: int
    budget: str
    eval_output_path: Path
    output_csv: Path
    coordinate_frame: CoordinateFrame = "shapenet"
    geometry: GeometryRendering = GeometryRendering()
    wall_distance_feature: bool = False
    evaluation_split: EvaluationSplit = "val"
    confirm_test_release: bool = False
    eval_run_id: str | None = None
    num_geometry_points: int = DEFAULT_GEOMETRY_POINTS
    num_geometry_supernodes: int = DEFAULT_GEOMETRY_SUPERNODES
    num_surface_queries: int = DEFAULT_EVALUATION_POINTS
    num_volume_queries: int = DEFAULT_EVALUATION_POINTS
    eval_point_seed: int = 4242
    precision: Precision = "float16"
    accelerator: Accelerator = "gpu"
    num_workers: int = 8

    def __post_init__(self) -> None:
        """Reject test access without consent and non-positive point counts.

        Raises:
            ValueError: If the test split was requested without explicit
                release consent, or a sampling count is not positive.
        """
        if self.evaluation_split == "test" and not self.confirm_test_release:
            raise ValueError("test evaluation requires explicit confirm_test_release")
        for name in ("num_geometry_points", "num_geometry_supernodes", "num_surface_queries", "num_volume_queries"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")

    def derive_run_id(self) -> str:
        """Return the explicit evaluation run ID or the derived one."""
        if self.eval_run_id is not None:
            return self.eval_run_id
        return (
            f"eval-{self.evaluation_split}-{self.target.run_id}-{self.target.checkpoint_tag}"
            f"-{self.coordinate_frame}-q{self.num_surface_queries}"
        )


def verify_training_provenance(
    request: EvaluationRequest,
    *,
    protocol: ProtocolBinding,
    manifest_cell: ManifestCell,
    statistics: StatisticsBinding,
) -> TrainingProvenance:
    """Validate that one sidecar describes exactly the requested cell.

    Args:
        request: The requested evaluation.
        protocol: Validated protocol binding.
        manifest_cell: Validated manifest cell.
        statistics: Validated statistics binding.

    Returns:
        The validated training provenance.

    Raises:
        ValueError: If any recorded field, checkpoint file, or checkpoint hash
            disagrees with the request or the artifacts on disk.
    """
    provenance = load_training_provenance(request.target.sidecar_path)
    replicate = protocol.replicate(request.replicate)
    expected = {
        "study_id": protocol.study_id,
        "protocol_sha256": protocol.sha256,
        "implementation_git_commit": manifest_cell.implementation_commit,
        "implementation_git_dirty": manifest_cell.implementation_dirty,
        "manifest_raw_sha256": manifest_cell.raw_file_sha256,
        "manifest_payload_sha256": manifest_cell.payload_sha256,
        "target_statistics_sha256": statistics.sha256,
        "task": request.task,
        "method": request.method,
        "replicate": request.replicate,
        "train_sample_size": request.sample_size,
        "coordinate_frame": request.coordinate_frame,
        "position_scale": request.geometry.position_scale,
        "supernode_radius": request.geometry.supernode_radius,
        "wall_distance_feature": request.wall_distance_feature,
        "budget": request.budget,
        "subset_seed": replicate.subset_seed,
        "model_seed": replicate.model_seed,
        "data_loader_seed": replicate.model_seed,
        "validation_seed": request.eval_point_seed,
        "evaluation_seed": request.eval_point_seed,
        "run_id": request.target.run_id,
        "stage_name": request.target.stage_name,
    }
    for key, expected_value in expected.items():
        actual_value = provenance.payload.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"training provenance mismatch for {key}: expected={expected_value!r}, actual={actual_value!r}"
            )
    if provenance["strategy"] != "scratch" and provenance["source_checkpoint_sha256"] != protocol.source_primary_sha256:
        raise ValueError(
            "transfer training provenance must bind the protocol primary source checkpoint: "
            f"sidecar={provenance['source_checkpoint_sha256']!r}, protocol={protocol.source_primary_sha256}"
        )

    checkpoint = request.target.checkpoint_path
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    recorded_hashes: dict[str, str] = provenance["model_checkpoints_sha256"]
    present = {path.name for path in checkpoint.parent.glob("*_model.th")}
    if present != set(recorded_hashes):
        raise ValueError(
            f"training provenance checkpoint set mismatch: recorded={sorted(recorded_hashes)}, actual={sorted(present)}"
        )
    for basename, expected_sha256 in sorted(recorded_hashes.items()):
        actual_sha256 = sha256_file(checkpoint.parent / basename)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"training provenance checkpoint SHA256 mismatch for {basename}: "
                f"expected={expected_sha256}, actual={actual_sha256}"
            )
    if recorded_hashes[checkpoint.name] != request.target.expected_sha256:
        raise ValueError(
            f"target checkpoint SHA256 mismatch: expected={request.target.expected_sha256}, "
            f"actual={recorded_hashes[checkpoint.name]}"
        )
    return provenance


def build_evaluation(
    request: EvaluationRequest,
    *,
    protocol: ProtocolBinding,
    manifest_cell: ManifestCell,
    statistics: StatisticsBinding,
    evaluator_git_state: GitState,
) -> tuple[Any, dict[str, Any]]:
    """Build one official-split inference config bound to frozen artifacts.

    Args:
        request: The requested evaluation.
        protocol: Validated protocol binding.
        manifest_cell: Validated manifest cell.
        statistics: Validated statistics binding.
        evaluator_git_state: Implementation state of the evaluating process.

    Returns:
        The resolved inference configuration and its audit record.

    Raises:
        ValueError: If the evaluator worktree is dirty or a provenance field
            disagrees with the request.
        AssertionError: If the resolved config exposes more than one split.
    """
    if evaluator_git_state.dirty:
        raise ValueError("frozen evaluation requires a clean implementation Git worktree")
    provenance = verify_training_provenance(
        request,
        protocol=protocol,
        manifest_cell=manifest_cell,
        statistics=statistics,
    )

    preset_class = DrivAerMLTransferCommonPreset if request.task == "common" else DrivAerMLTransferFullPreset
    preset = preset_class(
        statistics_artifact=request.statistics_path,
        coordinate_frame=request.coordinate_frame,
        expected_manifest_sha256=manifest_cell.raw_file_sha256,
        expected_train_subset_size=request.sample_size,
        wall_distance_feature=request.wall_distance_feature,
        position_scale=request.geometry.position_scale,
    )
    model_params = dict(CHECKPOINT_ARCHITECTURE)
    model_params["radius"] = request.geometry.supernode_radius
    model_params["initializers"] = [request.target.initializer()]

    evaluation_dataset = preset.build_dataset(
        split=request.evaluation_split,
        root=str(request.dataset_root),
        model_kind=MODEL_KIND,
        seed=request.eval_point_seed,
        num_geometry_points=request.num_geometry_points,
        num_geometry_supernodes=request.num_geometry_supernodes,
        num_surface_anchor_points=request.num_surface_queries,
        num_volume_anchor_points=request.num_volume_queries,
    )
    callback = PairedMetricsExportCallbackConfig(
        every_n_epochs=1,
        dataset_key=request.evaluation_split,
        batch_size=1,
        forward_properties=preset.forward_properties(MODEL_KIND),
        output_csv=str(request.output_csv),
        method=provenance["method"],
        rendering=request.geometry.label(),
        replicate=str(provenance["replicate"]),
        train_sample_size=provenance["train_sample_size"],
    )
    eval_run_id = request.derive_run_id()
    config = preset.build_config(
        model_kind=MODEL_KIND,
        model_params=model_params,
        optimizer=preset.build_optimizer(lr=5e-5, end_lr=None),
        trainer_kind=TRAINER_KIND,
        trainer_params={
            "field_weights": FIELD_WEIGHTS[request.task],
            "precision": request.precision,
            "find_unused_params": False,
            "static_graph": False,
        },
        dataset_root=str(request.dataset_root),
        output_path=str(request.eval_output_path),
        datasets=[],
        extra_datasets={request.evaluation_split: evaluation_dataset},
        callbacks_override=[callback],
        accelerator=request.accelerator,
        max_epochs=0,
        batch_size=1,
        seed=request.eval_point_seed,
        name=f"drivaerml-{request.task}-frozen-{request.evaluation_split}",
        run_id=eval_run_id,
        stage_name=request.evaluation_split,
        num_workers=request.num_workers,
        store_code_in_output=True,
    )
    if set(config.datasets) != {request.evaluation_split}:
        raise AssertionError(
            f"frozen evaluation config must contain {request.evaluation_split!r} only, got {sorted(config.datasets)}"
        )

    official_ids = getattr(DrivAerMLDefaultSplitIDs(), request.evaluation_split)
    audit = {
        "evaluation_run_id": eval_run_id,
        "evaluation_implementation_git_commit": evaluator_git_state.commit,
        "evaluation_implementation_git_dirty": evaluator_git_state.dirty,
        "training_implementation_git_commit": provenance["implementation_git_commit"],
        "training_implementation_git_dirty": provenance["implementation_git_dirty"],
        "training_provenance_sidecar": str(provenance.path),
        "training_provenance_sidecar_sha256": provenance.sha256,
        "target_checkpoint": str(request.target.checkpoint_path.resolve()),
        "target_checkpoint_sha256": request.target.expected_sha256,
        "target_run_id": request.target.run_id,
        "target_checkpoint_tag": request.target.checkpoint_tag,
        "protocol_sha256": protocol.sha256,
        "manifest_raw_sha256": manifest_cell.raw_file_sha256,
        "manifest_payload_sha256": manifest_cell.payload_sha256,
        "target_statistics_sha256": statistics.sha256,
        "task": provenance["task"],
        "strategy": provenance["strategy"],
        "method": provenance["method"],
        "replicate": provenance["replicate"],
        "train_sample_size": provenance["train_sample_size"],
        "coordinate_frame": provenance["coordinate_frame"],
        "position_scale": request.geometry.position_scale,
        "supernode_radius": request.geometry.supernode_radius,
        "supernode_radius_raw_units": provenance["supernode_radius_raw_units"],
        "wall_distance_feature": provenance["wall_distance_feature"],
        "input_feature": provenance["input_feature"],
        "budget": provenance["budget"],
        "model_seed": provenance["model_seed"],
        "training_pipeline_seed": provenance["training_pipeline_seed"],
        "data_loader_seed": provenance["data_loader_seed"],
        "validation_seed": provenance["validation_seed"],
        "evaluation_seed": provenance["evaluation_seed"],
        "evaluation_split": request.evaluation_split,
        "test_release_confirmed": request.confirm_test_release,
        "official_evaluation_design_ids": official_ids,
        "evaluation_design_count": len(official_ids),
        "surface_queries_per_design": request.num_surface_queries,
        "volume_queries_per_design": request.num_volume_queries,
        "output_csv": str(request.output_csv.resolve()),
        "datasets_in_eval_config": sorted(config.datasets),
    }
    return config, audit
