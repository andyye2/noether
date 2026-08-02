# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Construction of one leakage-safe DrivAerML train+validation experiment.

Every scientific choice that separates the paired arms lives here: the frozen
architecture of the ShapeNet-Car source checkpoint, the geometry rendering,
the budget, and the strict source-trunk initialization.  The entry point under
``recipes/aero_cfd/scripts`` only parses arguments.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Literal

from aero_cfd.presets.drivaerml_transfer import DrivAerMLTransferCommonPreset, DrivAerMLTransferFullPreset
from noether.core.schemas.callbacks import (
    BestCheckpointCallbackConfig,
    CheckpointCallbackConfig,
    OfflineLossCallbackConfig,
)
from noether.core.schemas.dataset import SubsetWrapperConfig
from noether.core.schemas.initializers import PreviousRunInitializerConfig
from noether.core.schemas.schema import ConfigSchema

from .integrity import sha256_file
from .manifest import DEFAULT_SIZES, ManifestCell
from .protocol import ProtocolBinding
from .statistics import StatisticsBinding, load_statistics_binding

Task = Literal["common", "full"]
Strategy = Literal["scratch", "finetune"]
Budget = Literal["compute_matched", "fixed_epoch", "smoke"]
CoordinateFrame = Literal["native", "shapenet"]
Precision = Literal["float32", "float16", "bfloat16"]
Accelerator = Literal["cpu", "gpu", "mps"]

TASKS: tuple[Task, ...] = ("common", "full")
STRATEGIES: tuple[Strategy, ...] = ("scratch", "finetune")
BUDGETS: tuple[Budget, ...] = ("compute_matched", "fixed_epoch", "smoke")

#: A run can only request a size the acquisition ladder actually freezes.
SAMPLE_SIZES = DEFAULT_SIZES

MODEL_KIND = "noether.modeling.models.aerodynamics.AeroABUPT"
TRAINER_KIND = "noether.training.trainers.WeightedLossTrainer"
MODEL_NAME = "ab_upt"
STAGE_NAME = "train"

#: Parameters whose target-side readout is reset instead of transferred.
RESET_PATTERN = "backbone.domain_decoder_projections"

#: Preregistered method code of each strategy, used in the metric tables. The
#: key type is widened because consumers look strategies up from parsed JSON.
METHOD_BY_STRATEGY: Mapping[str, str] = {"scratch": "S", "finetune": "P-FT"}

#: Architecture of the ShapeNet-Car source checkpoint.  Scratch and transfer
#: arms share it exactly, so the comparison isolates initialization.
CHECKPOINT_ARCHITECTURE: dict[str, Any] = {
    "hidden_dim": 192,
    "geometry_depth": 1,
    "physics_blocks": ["perceiver", "self", "cross", "self", "cross", "self", "cross", "self", "cross", "self"],
    "num_domain_decoder_blocks": {"surface": 2, "volume": 2},
    "num_heads": 3,
    "mlp_expansion_factor": 4,
    "radius": 9,
}

#: Normalized position upper bound the source checkpoint was pretrained under.
#: The sincos and RoPE frequency buffers are restored from the checkpoint, so a
#: transfer run rendered at another scale would evaluate the frozen bands at
#: shifted phases.
CHECKPOINT_POSITION_SCALE = 1000.0

#: Supernode-pooling radius the source checkpoint was pretrained under.
CHECKPOINT_SUPERNODE_RADIUS = float(CHECKPOINT_ARCHITECTURE["radius"])

FIELD_WEIGHTS: dict[Task, dict[str, float]] = {
    "common": {"surface_pressure": 1.0, "volume_velocity": 1.0},
    "full": {
        "surface_pressure": 1.0,
        "surface_friction": 1.0,
        "volume_pressure": 1.0,
        "volume_velocity": 1.0,
        "volume_vorticity": 1.0,
    },
}

COMPUTE_MATCHED_UPDATES = 40_000
FIXED_EPOCH_EPOCHS = 100
CLIP_GRAD_NORM = 0.25
WARMUP_PERCENT = 0.05
VALIDATIONS_PER_RUN = 20
CHECKPOINTS_PER_RUN = 10


@dataclass(frozen=True)
class GeometryRendering:
    """How raw metres become the normalized coordinates the model sees.

    ``position_scale`` maps the fixed CFD-domain bounds onto ``[0, scale]`` and
    ``supernode_radius`` is expressed in those same normalized units.  The two
    are therefore not independent: rescaling positions without rescaling the
    radius changes the message-passing neighbourhood, which is what made the
    naive transfer arm build a degree-capped geometry graph.

    Attributes:
        position_scale: Upper bound of the normalized position range.
        supernode_radius: Supernode-pooling radius in normalized units.
    """

    position_scale: float = CHECKPOINT_POSITION_SCALE
    supernode_radius: float = CHECKPOINT_SUPERNODE_RADIUS

    def __post_init__(self) -> None:
        """Reject non-finite or non-positive geometry parameters."""
        for name, value in (("position_scale", self.position_scale), ("supernode_radius", self.supernode_radius)):
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")

    @property
    def radius_position_fraction(self) -> float:
        """Return the radius as a fraction of the normalized box edge."""
        return self.supernode_radius / self.position_scale

    def radius_in_raw_units(self, position_span: float) -> float:
        """Return the radius in the raw units of the dataset bounding box.

        Args:
            position_span: Width of the fixed normalization box in raw units
                (metres for DrivAerML).

        Returns:
            The physical message-passing radius.
        """
        return self.radius_position_fraction * position_span

    def label(self) -> str:
        """Return the rendering label recorded in every metric row."""
        return f"ps{self.position_scale:g}-sr{self.supernode_radius:g}"

    def run_id_suffix(self) -> str:
        """Return the run-ID suffix that marks a non-default rendering."""
        suffix = ""
        if self.position_scale != CHECKPOINT_POSITION_SCALE:
            suffix += f"-ps{self.position_scale:g}"
        if self.supernode_radius != CHECKPOINT_SUPERNODE_RADIUS:
            suffix += f"-sr{self.supernode_radius:g}"
        return suffix

    def require_checkpoint_basis(self) -> None:
        """Refuse a rendering that invalidates the restored positional basis.

        Raises:
            ValueError: If the position scale differs from the scale the
                source checkpoint was pretrained under.
        """
        if self.position_scale != CHECKPOINT_POSITION_SCALE:
            raise ValueError(
                "a transfer run must render positions on the pretrained positional basis: "
                f"position_scale must be {CHECKPOINT_POSITION_SCALE}, got {self.position_scale}; "
                "rescale --supernode-radius instead to change the message-passing neighbourhood"
            )


@dataclass(frozen=True)
class TrainingBudget:
    """Exact stopping criteria of one preregistered budget.

    Attributes:
        name: Preregistered budget name.
        max_epochs: Epoch limit, or ``None`` when updates stop the run.
        max_updates: Optimizer-update limit, or ``None`` for epoch budgets.
        expected_updates: Updates the schedule is built for.
    """

    name: Budget
    max_epochs: int | None
    max_updates: int | None
    expected_updates: int


@dataclass(frozen=True)
class SourceCheckpoint:
    """Location of the confirmatory ShapeNet-Car source checkpoint.

    Attributes:
        output_path: Root of the source training outputs.
        run_id: Source run identifier.
        stage_name: Source stage that produced the checkpoint.
        model_name: Model name used in the checkpoint filename.
        checkpoint_tag: Checkpoint tag, for example ``latest``.
        model_info: Optional weight-stream infix such as ``ema=0.9999``.
    """

    output_path: Path
    run_id: str = "2026-04-25_7d0mv"
    stage_name: str = STAGE_NAME
    model_name: str = MODEL_NAME
    checkpoint_tag: str = "latest"
    model_info: str | None = None

    def resolve(self, protocol: ProtocolBinding) -> Path:
        """Resolve and checksum the protocol's confirmatory source checkpoint.

        Args:
            protocol: Protocol binding that pins the confirmatory source.

        Returns:
            Path of the verified checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
            ValueError: If a non-confirmatory source was requested or the file
                bytes disagree with the protocol.
        """
        if self.checkpoint_tag != protocol.source_primary_checkpoint_tag or self.model_info is not None:
            raise ValueError(
                "strict confirmatory training requires the protocol primary source checkpoint: "
                f"tag={protocol.source_primary_checkpoint_tag!r}, model_info=None"
            )
        filename = f"{self.model_name}_cp={self.checkpoint_tag}_model.th"
        checkpoint = self.output_path / self.run_id / self.stage_name / "checkpoints" / filename
        if not checkpoint.is_file():
            raise FileNotFoundError(f"source checkpoint does not exist: {checkpoint}")
        actual = sha256_file(checkpoint)
        if actual != protocol.source_primary_sha256:
            raise ValueError(
                f"source checkpoint SHA256 mismatch: expected={protocol.source_primary_sha256}, actual={actual}"
            )
        return checkpoint

    def initializer(self) -> PreviousRunInitializerConfig:
        """Build the strict source-trunk/fresh-readout initializer.

        The readout projections are removed from the restored state dict and
        re-instantiated, because the DrivAerML fields do not share the source
        output semantics; every other tensor must load strictly.
        """
        return PreviousRunInitializerConfig(
            output_path=self.output_path,
            run_id=self.run_id,
            stage_name=self.stage_name,
            model_name=self.model_name,
            model_info=self.model_info,
            checkpoint_tag=self.checkpoint_tag,
            patterns_to_remove=[RESET_PATTERN],
            patterns_to_instantiate=[RESET_PATTERN],
        )


@dataclass(frozen=True)
class TrainingRequest:
    """One fully specified train+validation cell.

    Attributes:
        dataset_root: Root of the DrivAerML dataset.
        protocol_path: Path of the frozen preregistration.
        manifest_path: Path of the materialized subset manifest.
        statistics_path: Path of the train-subset-only statistics artifact.
        output_path: Root the run writes into.
        task: Preregistered field set.
        strategy: ``scratch`` or ``finetune``.
        sample_size: Training-subset size.
        budget: Preregistered stopping rule.
        smoke_updates: Update count of the ``smoke`` budget.
        coordinate_frame: Frame the dataset is expressed in.
        geometry: Position scale and supernode radius.
        replicate: Preregistered replicate label.
        model_seed: Seed shared by the paired arms.
        eval_point_seed: Seed of the fixed validation point sampling.
        source: Source-checkpoint location, unused by ``scratch``.
        learning_rate: Base learning rate.
        end_learning_rate: Final cosine learning rate.
        weight_decay: Lion weight decay.
        effective_batch_size: Preregistered as one.
        precision: Trainer precision.
        accelerator: Device kind.
        num_workers: Data-loader workers.
        run_id: Explicit run identifier, or ``None`` to derive it.
    """

    dataset_root: Path
    protocol_path: Path
    manifest_path: Path
    statistics_path: Path
    output_path: Path
    task: Task
    strategy: Strategy
    sample_size: int
    replicate: int
    model_seed: int
    source: SourceCheckpoint
    budget: Budget = "compute_matched"
    smoke_updates: int = 10
    coordinate_frame: CoordinateFrame = "shapenet"
    geometry: GeometryRendering = GeometryRendering()
    eval_point_seed: int = 4242
    learning_rate: float = 5e-5
    end_learning_rate: float = 1e-6
    weight_decay: float = 0.05
    effective_batch_size: int = 1
    precision: Precision = "float16"
    accelerator: Accelerator = "gpu"
    num_workers: int = 8
    run_id: str | None = None

    def __post_init__(self) -> None:
        """Reject numerically invalid or protocol-violating requests.

        Raises:
            ValueError: If a hyper-parameter is non-finite, the batch size is
                not the preregistered one, or a transfer arm requests a
                position scale the source checkpoint was not trained under.
        """
        if self.effective_batch_size != 1:
            raise ValueError("the preregistered study requires effective_batch_size 1")
        if self.smoke_updates < 1:
            raise ValueError("smoke_updates must be positive")
        for name, value in (("learning_rate", self.learning_rate), ("end_learning_rate", self.end_learning_rate)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError(f"weight_decay must be finite and non-negative, got {self.weight_decay!r}")
        if self.strategy != "scratch":
            self.geometry.require_checkpoint_basis()

    def resolve_budget(self) -> TrainingBudget:
        """Resolve exact update/epoch stopping criteria for this request."""
        if self.budget == "compute_matched":
            return TrainingBudget(
                name="compute_matched",
                max_epochs=None,
                max_updates=COMPUTE_MATCHED_UPDATES,
                expected_updates=COMPUTE_MATCHED_UPDATES,
            )
        if self.budget == "fixed_epoch":
            return TrainingBudget(
                name="fixed_epoch",
                max_epochs=FIXED_EPOCH_EPOCHS,
                max_updates=None,
                expected_updates=FIXED_EPOCH_EPOCHS * self.sample_size,
            )
        return TrainingBudget(
            name="smoke",
            max_epochs=None,
            max_updates=self.smoke_updates,
            expected_updates=self.smoke_updates,
        )

    def derive_run_id(self, manifest_cell: ManifestCell) -> str:
        """Return the explicit run ID or the deterministic derived one."""
        if self.run_id is not None:
            return self.run_id
        return (
            f"mf-{self.task}-{self.strategy}-r{self.replicate}-n{self.sample_size}"
            f"-m{manifest_cell.payload_sha256[:10]}-s{self.model_seed}"
            f"-{self.coordinate_frame}-{self.budget}{self.geometry.run_id_suffix()}"
        )


@dataclass(frozen=True)
class BuiltExperiment:
    """A resolved experiment together with its audit record.

    Attributes:
        config: Fully resolved Noether configuration.
        audit: JSON-compatible record of every binding the run depends on.
    """

    config: ConfigSchema
    audit: dict[str, Any]


def build_callbacks(expected_updates: int) -> list[Any]:
    """Build update-based validation and checkpoint schedules with no test access.

    Args:
        expected_updates: Updates the run is scheduled for.

    Returns:
        Checkpoint, offline validation-loss, and best-checkpoint callbacks.
    """
    eval_every = max(1, expected_updates // VALIDATIONS_PER_RUN)
    save_every = max(1, expected_updates // CHECKPOINTS_PER_RUN)
    return [
        CheckpointCallbackConfig(
            kind="noether.core.callbacks.CheckpointCallback",
            every_n_updates=save_every,
            save_weights=True,
            save_latest_weights=True,
        ),
        OfflineLossCallbackConfig(
            kind="noether.training.callbacks.OfflineLossCallback",
            every_n_updates=eval_every,
            dataset_key="val",
            batch_size=1,
        ),
        BestCheckpointCallbackConfig(
            kind="noether.core.callbacks.BestCheckpointCallback",
            every_n_updates=eval_every,
            metric_key="loss/val/total",
        ),
    ]


def _position_span(statistics: dict[str, Any]) -> float:
    """Return the widest raw-unit extent of the fixed normalization box."""
    minima = statistics["raw_pos_min"]
    maxima = statistics["raw_pos_max"]
    minima = minima if isinstance(minima, list) else [minima]
    maxima = maxima if isinstance(maxima, list) else [maxima]
    return max(float(upper) - float(lower) for lower, upper in zip(minima, maxima, strict=True))


def build_experiment(
    request: TrainingRequest,
    manifest_cell: ManifestCell,
    protocol: ProtocolBinding,
    *,
    statistics: StatisticsBinding | None = None,
) -> BuiltExperiment:
    """Construct one fully bound train+validation experiment.

    Args:
        request: The requested experiment cell.
        manifest_cell: Validated subset manifest cell.
        protocol: Validated protocol binding.
        statistics: Pre-validated statistics binding; loaded when omitted.

    Returns:
        The resolved configuration and its audit record.

    Raises:
        ValueError: If the manifest, protocol, replicate seeds, or statistics
            artifact do not identify the same experiment cell.
        AssertionError: If the resolved config would expose the test split.
    """
    if manifest_cell.protocol_sha256 != protocol.sha256:
        raise ValueError(
            f"manifest/protocol SHA256 mismatch: manifest={manifest_cell.protocol_sha256}, protocol={protocol.sha256}"
        )
    if manifest_cell.study_id != protocol.study_id:
        raise ValueError(
            f"manifest/protocol study_id mismatch: manifest={manifest_cell.study_id}, protocol={protocol.study_id}"
        )
    replicate = protocol.replicate(request.replicate)
    if manifest_cell.seed != replicate.subset_seed:
        raise ValueError(
            f"replicate {request.replicate} requires subset seed {replicate.subset_seed}, "
            f"manifest has {manifest_cell.seed}"
        )
    if request.model_seed != replicate.model_seed:
        raise ValueError(
            f"replicate {request.replicate} requires model seed {replicate.model_seed}, got {request.model_seed}"
        )
    statistics = statistics or load_statistics_binding(request.statistics_path, protocol=protocol)

    preset_class = DrivAerMLTransferCommonPreset if request.task == "common" else DrivAerMLTransferFullPreset
    preset = preset_class(
        statistics_artifact=request.statistics_path,
        coordinate_frame=request.coordinate_frame,
        expected_manifest_sha256=manifest_cell.raw_file_sha256,
        expected_train_subset_size=request.sample_size,
        position_scale=request.geometry.position_scale,
    )

    source_checkpoint: Path | None = None
    model_params = dict(CHECKPOINT_ARCHITECTURE)
    model_params["radius"] = request.geometry.supernode_radius
    if request.strategy != "scratch":
        source_checkpoint = request.source.resolve(protocol)
        model_params["initializers"] = [request.source.initializer()]

    budget = request.resolve_budget()
    train_dataset = preset.build_dataset(
        split="train",
        root=str(request.dataset_root),
        model_kind=MODEL_KIND,
        wrappers=[
            SubsetWrapperConfig(
                kind="noether.data.base.wrappers.SubsetWrapper",
                indices=list(manifest_cell.base_dataset_indices),
            )
        ],
        seed=None,
    )
    val_dataset = preset.build_dataset(
        split="val",
        root=str(request.dataset_root),
        model_kind=MODEL_KIND,
        seed=request.eval_point_seed,
    )

    optimizer = preset.build_optimizer(
        lr=request.learning_rate,
        weight_decay=request.weight_decay,
        clip_grad_norm=CLIP_GRAD_NORM,
        warmup_percent=WARMUP_PERCENT,
        end_lr=request.end_learning_rate,
    )

    trainer_params: dict[str, Any] = {
        "field_weights": FIELD_WEIGHTS[request.task],
        "precision": request.precision,
        "find_unused_params": False,
        "static_graph": False,
    }
    if budget.max_updates is not None:
        trainer_params["max_epochs"] = None
        trainer_params["max_updates"] = budget.max_updates

    run_id = request.derive_run_id(manifest_cell)
    config = preset.build_config(
        model_kind=MODEL_KIND,
        model_params=model_params,
        optimizer=optimizer,
        trainer_kind=TRAINER_KIND,
        trainer_params=trainer_params,
        dataset_root=str(request.dataset_root),
        output_path=str(request.output_path),
        datasets=[],
        extra_datasets={"train": train_dataset, "val": val_dataset},
        callbacks_override=build_callbacks(budget.expected_updates),
        accelerator=request.accelerator,
        max_epochs=int(budget.max_epochs or 1),
        batch_size=request.effective_batch_size,
        seed=request.model_seed,
        name=f"drivaerml-{request.task}-{request.strategy}",
        run_id=run_id,
        stage_name=STAGE_NAME,
        num_workers=request.num_workers,
        store_code_in_output=True,
    )
    if "test" in config.datasets:
        raise AssertionError("strict training config unexpectedly contains the test dataset")

    position_span = _position_span(preset.dataset_statistics)
    audit: dict[str, Any] = {
        "run_id": run_id,
        "stage_name": STAGE_NAME,
        "study_id": protocol.study_id,
        "protocol_path": str(protocol.path),
        "protocol_sha256": protocol.sha256,
        "task": request.task,
        "strategy": request.strategy,
        "method": METHOD_BY_STRATEGY[request.strategy],
        "replicate": request.replicate,
        "budget": {
            "name": budget.name,
            "max_epochs": budget.max_epochs,
            "max_updates": budget.max_updates,
            "expected_updates": budget.expected_updates,
        },
        "manifest": {
            "path": str(manifest_cell.path),
            "raw_file_sha256": manifest_cell.raw_file_sha256,
            "payload_sha256": manifest_cell.payload_sha256,
            "seed": manifest_cell.seed,
            "base_dataset_indices": list(manifest_cell.base_dataset_indices),
            "run_ids": list(manifest_cell.run_ids),
            "implementation_git_commit": manifest_cell.implementation_commit,
            "implementation_git_dirty": manifest_cell.implementation_dirty,
        },
        "target_statistics": {
            "path": str(statistics.path),
            "sha256": statistics.sha256,
            "protocol_sha256": statistics.protocol_sha256,
            "study_id": statistics.study_id,
        },
        "train_sample_size": request.sample_size,
        "coordinate_frame": request.coordinate_frame,
        "position_scale": request.geometry.position_scale,
        "supernode_radius": request.geometry.supernode_radius,
        "supernode_radius_position_fraction": request.geometry.radius_position_fraction,
        "supernode_radius_raw_units": request.geometry.radius_in_raw_units(position_span),
        "position_span_raw_units": position_span,
        "source_checkpoint": str(source_checkpoint.resolve()) if source_checkpoint else None,
        "source_checkpoint_sha256": protocol.source_primary_sha256 if source_checkpoint else None,
        "model_seed": request.model_seed,
        "training_pipeline_seed": None,
        "data_loader_seed": request.model_seed,
        "validation_seed": request.eval_point_seed,
        "evaluation_seed": request.eval_point_seed,
        "test_dataset_in_training_config": False,
    }
    return BuiltExperiment(config=config, audit=audit)
