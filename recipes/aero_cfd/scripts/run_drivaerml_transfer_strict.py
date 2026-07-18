# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Run leakage-safe ShapeNet-Car to DrivAerML AB-UPT experiments.

This confirmatory entry point deliberately excludes the test dataset.  It binds
each run to a hashed nested-subset manifest, a train-subset-only statistics
artifact, a coordinate frame, and (for transfer methods) a checksum-verified
source checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Literal

import yaml

from aero_cfd.callbacks.staged_transfer import (
    StagedTransferCallbackConfig,
    TransferStageConfig,
)
from aero_cfd.presets.drivaerml_transfer import (
    DrivAerMLTransferCommonPreset,
    DrivAerMLTransferFullPreset,
)
from noether.core.distributed.utils import accelerator_to_device
from noether.core.schemas.callbacks import (
    BestCheckpointCallbackConfig,
    CheckpointCallbackConfig,
    OfflineLossCallbackConfig,
)
from research.multi_fidelity.tools.materialize_nested_manifests import verify_manifest
from noether.core.schemas.dataset import SubsetWrapperConfig
from noether.core.schemas.initializers import PreviousRunInitializerConfig
from noether.core.schemas.optimizers import ParamGroupModifierConfig
from noether.core.schemas.schema import ConfigSchema
from noether.data.datasets.cfd.caeml.drivaerml.split import DrivAerMLDefaultSplitIDs
from noether.training.runners import HydraRunner

MODEL_KIND = "noether.modeling.models.aerodynamics.AeroABUPT"
TRAINER_KIND = "noether.training.trainers.WeightedLossTrainer"
RESET_PATTERN = "backbone.domain_decoder_projections"
DECODER_PATTERN = "backbone.domain_decoder_blocks"
TRANSFER_STRATEGIES = ("scratch", "finetune", "linear_probe", "gradual_unfreeze")
TASKS = ("common", "full")
BUDGETS = ("compute_matched", "fixed_epoch", "smoke")
FROZEN_PROTOCOL_STATUS = "frozen_before_first_target_job"
METHOD_BY_STRATEGY = {
    "scratch": "S",
    "finetune": "P-FT",
    "linear_probe": "P-LP",
    "gradual_unfreeze": "P-GU",
}
PROVENANCE_FILENAME = "training_provenance.json"
REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_ARCHITECTURE: dict[str, Any] = {
    "hidden_dim": 192,
    "geometry_depth": 1,
    "physics_blocks": [
        "perceiver",
        "self",
        "cross",
        "self",
        "cross",
        "self",
        "cross",
        "self",
        "cross",
        "self",
    ],
    "num_domain_decoder_blocks": {"surface": 2, "volume": 2},
    "num_heads": 3,
    "mlp_expansion_factor": 4,
    "radius": 9,
}
FIELD_WEIGHTS = {
    "common": {"surface_pressure": 1.0, "volume_velocity": 1.0},
    "full": {
        "surface_pressure": 1.0,
        "surface_friction": 1.0,
        "volume_pressure": 1.0,
        "volume_velocity": 1.0,
        "volume_vorticity": 1.0,
    },
}
KNOWN_SOURCE_SHA256 = {
    ("latest", None): "261a46b7464d50c26301db60758feaabc27ce3d4f30b511e826201d32154fe38",
    (
        "best_model.loss.test.total",
        None,
    ): "221ce5a1a6803876b5d0fc6157a8f06aa5299abe104c4ee9fcbef3f0aaffc5f7",
    (
        "latest",
        "ema=0.9999",
    ): "f1200056bee2a2be2cac0f390bf9aaa049d1b1f29b680bab0721969ad8ba4aeb",
}


def _canonical_manifest_bytes(payload: dict[str, Any]) -> bytes:
    """Return the canonical representation used by the manifest generator."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _canonical_json_sha256(payload: Any) -> str:
    """Hash a JSON-compatible object with a stable representation."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_protocol_binding(protocol_path: Path) -> dict[str, Any]:
    """Load the preregistration and return its immutable identity and seed table."""
    raw = protocol_path.read_bytes()
    protocol = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(protocol, dict):
        raise TypeError("protocol root must be a mapping")
    study_id = protocol.get("study_id")
    if not isinstance(study_id, str) or not study_id:
        raise ValueError("protocol must define a non-empty study_id")
    status = protocol.get("status")
    if status not in {"preregistration_draft", FROZEN_PROTOCOL_STATUS}:
        raise ValueError(
            f"protocol status must be 'preregistration_draft' or {FROZEN_PROTOCOL_STATUS!r}, got {status!r}"
        )
    replicates = protocol.get("data", {}).get("paired_replicates")
    if not isinstance(replicates, list) or len(replicates) != 8:
        raise ValueError("protocol must define exactly eight paired replicates")
    for expected, entry in enumerate(replicates):
        if not isinstance(entry, dict) or entry.get("replicate") != expected:
            raise ValueError("protocol replicate IDs must be consecutive and ordered from zero")
        if not isinstance(entry.get("subset_seed"), int) or not isinstance(entry.get("model_seed"), int):
            raise ValueError(f"protocol replicate {expected} has invalid subset/model seeds")

    source = protocol.get("source")
    if not isinstance(source, dict):
        raise ValueError("protocol must define a source mapping")
    primary_checkpoint_tag = source.get("primary_checkpoint_tag")
    primary_sha256 = source.get("primary_sha256")
    if not isinstance(primary_checkpoint_tag, str) or not primary_checkpoint_tag:
        raise ValueError("protocol source.primary_checkpoint_tag must be a non-empty string")
    if (
        not isinstance(primary_sha256, str)
        or len(primary_sha256) != 64
        or any(character not in "0123456789abcdef" for character in primary_sha256)
    ):
        raise ValueError("protocol source.primary_sha256 must be a lowercase SHA256")
    known_primary_sha256 = KNOWN_SOURCE_SHA256.get((primary_checkpoint_tag, None))
    if known_primary_sha256 is None or primary_sha256 != known_primary_sha256:
        raise ValueError(
            "protocol primary source checkpoint disagrees with the audited constant: "
            f"tag={primary_checkpoint_tag!r}, protocol={primary_sha256}, known={known_primary_sha256}"
        )
    return {
        "path": str(protocol_path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "study_id": study_id,
        "status": status,
        "replicates": replicates,
        "source_primary_checkpoint_tag": primary_checkpoint_tag,
        "source_primary_sha256": primary_sha256,
    }


def require_frozen_protocol_for_execution(protocol_binding: dict[str, Any]) -> None:
    """Reject production target-data work until the preregistration is frozen."""
    if protocol_binding["status"] != FROZEN_PROTOCOL_STATUS:
        raise ValueError(
            "production execution is blocked while protocol status is "
            f"{protocol_binding['status']!r}; set it to {FROZEN_PROTOCOL_STATUS!r} only after explicit approval"
        )


def load_manifest_cell(
    manifest_path: Path,
    sample_size: int,
    *,
    expected_protocol_sha256: str | None = None,
    expected_study_id: str | None = None,
) -> dict[str, Any]:
    """Validate a frozen manifest and return one exact nested subset cell."""
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("manifest root must be a mapping")
    verify_manifest(manifest)

    claimed_payload_sha = manifest.get("manifest_sha256")
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    actual_payload_sha = hashlib.sha256(_canonical_manifest_bytes(unhashed)).hexdigest()
    if claimed_payload_sha != actual_payload_sha:
        raise ValueError(
            f"manifest payload SHA256 mismatch: claimed={claimed_payload_sha}, actual={actual_payload_sha}"
        )

    study_id = manifest.get("study_id")
    protocol_sha256 = manifest.get("protocol_sha256")
    implementation_git_commit = manifest.get("implementation_git_commit")
    implementation_git_dirty = manifest.get("implementation_git_dirty")
    if not isinstance(study_id, str) or not study_id:
        raise ValueError("manifest must define a non-empty study_id")
    if (
        not isinstance(protocol_sha256, str)
        or len(protocol_sha256) != 64
        or any(character not in "0123456789abcdef" for character in protocol_sha256)
    ):
        raise ValueError("manifest must define a lowercase protocol_sha256")
    if expected_protocol_sha256 is not None and protocol_sha256 != expected_protocol_sha256:
        raise ValueError(
            f"manifest/protocol SHA256 mismatch: manifest={protocol_sha256}, protocol={expected_protocol_sha256}"
        )
    if expected_study_id is not None and study_id != expected_study_id:
        raise ValueError(f"manifest/protocol study_id mismatch: manifest={study_id}, protocol={expected_study_id}")
    if (
        not isinstance(implementation_git_commit, str)
        or len(implementation_git_commit) != 40
        or any(character not in "0123456789abcdef" for character in implementation_git_commit)
    ):
        raise ValueError("manifest must define a lowercase 40-character implementation_git_commit")
    if not isinstance(implementation_git_dirty, bool):
        raise ValueError("manifest must define boolean implementation_git_dirty")

    subsets = manifest.get("subsets")
    if not isinstance(subsets, dict) or not isinstance(subsets.get(str(sample_size)), dict):
        raise ValueError(f"manifest has no v2 subset cell for N={sample_size}")
    cell = subsets[str(sample_size)]
    indices = [int(value) for value in cell.get("base_dataset_indices", [])]
    run_ids = [int(value) for value in cell.get("run_ids", [])]
    if len(indices) != sample_size or len(run_ids) != sample_size:
        raise ValueError(f"manifest N={sample_size} lengths are indices={len(indices)}, run_ids={len(run_ids)}")
    if len(set(indices)) != sample_size or len(set(run_ids)) != sample_size:
        raise ValueError(f"manifest N={sample_size} contains duplicates")

    official = DrivAerMLDefaultSplitIDs()
    resolved = [official.train[index] for index in indices]
    if resolved != run_ids:
        raise ValueError("manifest base-dataset indices do not resolve to the recorded run IDs")
    if manifest.get("official_val_run_ids") != official.val:
        raise ValueError("manifest validation IDs differ from the current official split")
    if manifest.get("official_test_run_ids") != official.test:
        raise ValueError("manifest test IDs differ from the current official split")
    if set(run_ids) & (set(official.val) | set(official.test)):
        raise ValueError("manifest training subset overlaps validation/test")

    return {
        "path": str(manifest_path.resolve()),
        "base_dataset_indices": indices,
        "run_ids": run_ids,
        "raw_file_sha256": hashlib.sha256(raw).hexdigest(),
        "payload_sha256": actual_payload_sha,
        "seed": manifest.get("seed"),
        "study_id": study_id,
        "protocol_sha256": protocol_sha256,
        "implementation_git_commit": implementation_git_commit,
        "implementation_git_dirty": implementation_git_dirty,
    }


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    """Hash a local checkpoint without deserializing it."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_statistics_binding(
    statistics_path: Path,
    *,
    expected_protocol_sha256: str,
    expected_study_id: str,
) -> dict[str, str]:
    """Validate protocol/study binding and hash a train-only statistics artifact."""
    raw = statistics_path.read_bytes()
    artifact = json.loads(raw.decode("utf-8"))
    if not isinstance(artifact, dict):
        raise TypeError("statistics artifact root must be a mapping")
    protocol_sha256 = artifact.get("protocol_sha256")
    study_id = artifact.get("study_id")
    if protocol_sha256 != expected_protocol_sha256:
        raise ValueError(
            f"statistics/protocol SHA256 mismatch: statistics={protocol_sha256}, protocol={expected_protocol_sha256}"
        )
    if study_id != expected_study_id:
        raise ValueError(f"statistics/protocol study_id mismatch: statistics={study_id}, protocol={expected_study_id}")
    return {
        "path": str(statistics_path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "protocol_sha256": protocol_sha256,
        "study_id": study_id,
    }


def validate_replicate_binding(
    args: argparse.Namespace,
    protocol_binding: dict[str, Any],
    manifest_cell: dict[str, Any],
) -> None:
    """Reject labels or seeds that do not identify the protocol replicate."""
    entry = protocol_binding["replicates"][args.replicate]
    if manifest_cell["seed"] != entry["subset_seed"]:
        raise ValueError(
            f"replicate {args.replicate} requires subset seed {entry['subset_seed']}, "
            f"manifest has {manifest_cell['seed']}"
        )
    if args.model_seed != entry["model_seed"]:
        raise ValueError(f"replicate {args.replicate} requires model seed {entry['model_seed']}, got {args.model_seed}")


def implementation_git_state(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Capture the implementation commit and dirty status without mutating Git."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status).hexdigest(),
    }


def validate_implementation_binding(
    manifest_cell: dict[str, Any],
    git_state: dict[str, Any],
) -> None:
    """Ensure execution uses the implementation state frozen into the manifest."""
    if manifest_cell["implementation_git_commit"] != git_state["commit"]:
        raise ValueError(
            "manifest/implementation commit mismatch: "
            f"manifest={manifest_cell['implementation_git_commit']}, runtime={git_state['commit']}"
        )
    if manifest_cell["implementation_git_dirty"] is not git_state["dirty"]:
        raise ValueError(
            "manifest/implementation dirty-state mismatch: "
            f"manifest={manifest_cell['implementation_git_dirty']}, runtime={git_state['dirty']}"
        )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace one JSON artifact after writing it completely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as file:
            encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_training_provenance_sidecar(
    args: argparse.Namespace,
    config: ConfigSchema,
    audit: dict[str, Any],
    git_state: dict[str, Any],
) -> Path:
    """Hash every model checkpoint and atomically publish training provenance."""
    run_stage_path = args.output_path / audit["run_id"] / "train"
    checkpoint_paths = sorted((run_stage_path / "checkpoints").glob("*_model.th"))
    if not checkpoint_paths:
        raise RuntimeError(f"successful training produced no model checkpoints in {run_stage_path / 'checkpoints'}")
    checkpoint_hashes = {path.name: sha256_file(path) for path in checkpoint_paths}
    resolved_config = config.model_dump(mode="json", exclude_computed_fields=True)
    payload = {
        "schema_version": 1,
        "kind": "drivaerml_transfer_training_provenance",
        "study_id": audit["study_id"],
        "protocol_path": audit["protocol_path"],
        "protocol_sha256": audit["protocol_sha256"],
        "implementation_git_commit": git_state["commit"],
        "implementation_git_dirty": git_state["dirty"],
        "implementation_git_status_sha256": git_state["status_sha256"],
        "manifest_path": audit["manifest"]["path"],
        "manifest_raw_sha256": audit["manifest"]["raw_file_sha256"],
        "manifest_payload_sha256": audit["manifest"]["payload_sha256"],
        "target_statistics_path": audit["target_statistics"]["path"],
        "target_statistics_sha256": audit["target_statistics"]["sha256"],
        "source_checkpoint": audit["source_checkpoint"],
        "source_checkpoint_sha256": audit["source_checkpoint_sha256"],
        "task": audit["task"],
        "strategy": audit["strategy"],
        "method": audit["method"],
        "replicate": audit["replicate"],
        "train_sample_size": args.sample_size,
        "coordinate_frame": audit["coordinate_frame"],
        "budget": args.budget,
        "subset_seed": audit["manifest"]["seed"],
        "model_seed": args.model_seed,
        "data_seed": None,
        "training_pipeline_seed": None,
        "data_loader_seed": args.model_seed,
        "validation_seed": args.eval_point_seed,
        "evaluation_seed": args.eval_point_seed,
        "resolved_config_sha256": _canonical_json_sha256(resolved_config),
        "run_id": audit["run_id"],
        "stage_name": "train",
        "model_checkpoints_sha256": checkpoint_hashes,
    }
    sidecar = run_stage_path / PROVENANCE_FILENAME
    atomic_write_json(sidecar, payload)
    return sidecar


def source_checkpoint_path(args: argparse.Namespace, protocol_binding: dict[str, Any]) -> Path:
    """Resolve and checksum the protocol's confirmatory primary source checkpoint."""
    primary_tag = protocol_binding["source_primary_checkpoint_tag"]
    primary_sha256 = protocol_binding["source_primary_sha256"]
    if args.source_checkpoint_tag != primary_tag or args.source_model_info is not None:
        raise ValueError(
            "strict confirmatory training requires the protocol primary source checkpoint: "
            f"tag={primary_tag!r}, model_info=None"
        )
    if args.expected_source_sha256 is not None and args.expected_source_sha256 != primary_sha256:
        raise ValueError(
            "--expected-source-sha256 cannot override the protocol primary source: "
            f"override={args.expected_source_sha256}, protocol={primary_sha256}"
        )

    filename = f"{args.source_model_name}_cp={primary_tag}_model.th"
    checkpoint = args.source_output_path / args.source_run_id / args.source_stage_name / "checkpoints" / filename
    if not checkpoint.is_file():
        raise FileNotFoundError(f"source checkpoint does not exist: {checkpoint}")
    actual = sha256_file(checkpoint)
    if actual != primary_sha256:
        raise ValueError(f"source checkpoint SHA256 mismatch: expected={primary_sha256}, actual={actual}")
    return checkpoint


def training_budget(args: argparse.Namespace) -> dict[str, int | None]:
    """Resolve exact update/epoch stopping criteria for one protocol budget."""
    if args.effective_batch_size != 1:
        raise ValueError("the preregistered study requires --effective-batch-size 1")
    if args.budget == "compute_matched":
        return {"max_epochs": None, "max_updates": 40_000, "expected_updates": 40_000}
    if args.budget == "fixed_epoch":
        return {
            "max_epochs": 100,
            "max_updates": None,
            "expected_updates": 100 * args.sample_size,
        }
    return {
        "max_epochs": None,
        "max_updates": args.smoke_updates,
        "expected_updates": args.smoke_updates,
    }


def transfer_initializer(args: argparse.Namespace) -> PreviousRunInitializerConfig:
    """Build the strict source-trunk/fresh-readout initializer."""
    return PreviousRunInitializerConfig(
        output_path=args.source_output_path,
        run_id=args.source_run_id,
        stage_name=args.source_stage_name,
        model_name=args.source_model_name,
        model_info=args.source_model_info,
        checkpoint_tag=args.source_checkpoint_tag,
        patterns_to_remove=[RESET_PATTERN],
        patterns_to_instantiate=[RESET_PATTERN],
    )


def build_lr_modifiers(args: argparse.Namespace) -> list[ParamGroupModifierConfig] | None:
    """Build head/decoder/transferred-body learning-rate ratios."""
    if args.strategy == "scratch":
        if args.body_lr_multiplier is not None or args.decoder_lr_multiplier is not None:
            raise ValueError("LR multipliers are transfer-only; scratch must use the base LR")
        return None

    body_multiplier = args.body_lr_multiplier
    decoder_multiplier = args.decoder_lr_multiplier
    if body_multiplier is None:
        body_multiplier = 0.1 if args.strategy == "gradual_unfreeze" else 1.0
    if decoder_multiplier is None:
        decoder_multiplier = 0.3 if args.strategy == "gradual_unfreeze" else body_multiplier
    if (
        not math.isfinite(body_multiplier)
        or not math.isfinite(decoder_multiplier)
        or body_multiplier <= 0
        or decoder_multiplier <= 0
    ):
        raise ValueError("body and decoder LR multipliers must be finite and positive")

    modifiers: list[ParamGroupModifierConfig] = []
    if body_multiplier != 1.0:
        modifiers.append(
            ParamGroupModifierConfig(
                kind="aero_cfd.optimizer.transfer_lr_modifiers.LrScaleExceptPatternModifier",
                name=RESET_PATTERN,
                scale=body_multiplier,
            )
        )
    if decoder_multiplier != body_multiplier:
        modifiers.append(
            ParamGroupModifierConfig(
                kind="aero_cfd.optimizer.transfer_lr_modifiers.LrScaleByPatternModifier",
                name=DECODER_PATTERN,
                scale=decoder_multiplier / body_multiplier,
            )
        )
    return modifiers or None


def build_callbacks(args: argparse.Namespace, expected_updates: int) -> list[Any]:
    """Build update-based validation/checkpoint schedules with no test access."""
    eval_every = max(1, expected_updates // 20)
    save_every = max(1, expected_updates // 10)
    callbacks: list[Any] = [
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
    if args.strategy == "linear_probe":
        callbacks.append(
            StagedTransferCallbackConfig(
                every_n_updates=expected_updates,
                stages=[TransferStageConfig(start_update=0, trainable_patterns=[RESET_PATTERN])],
            )
        )
    elif args.strategy == "gradual_unfreeze":
        head_end = max(1, math.floor(0.10 * expected_updates))
        decoder_end = max(head_end + 1, math.floor(0.30 * expected_updates))
        callbacks.append(
            StagedTransferCallbackConfig(
                every_n_updates=head_end,
                stages=[
                    TransferStageConfig(start_update=0, trainable_patterns=[RESET_PATTERN]),
                    TransferStageConfig(
                        start_update=head_end,
                        trainable_patterns=[RESET_PATTERN, DECODER_PATTERN],
                    ),
                    TransferStageConfig(start_update=decoder_end, train_all=True),
                ],
            )
        )
    return callbacks


def build_experiment_config(
    args: argparse.Namespace,
    manifest_cell: dict[str, Any],
) -> tuple[ConfigSchema, dict[str, Any]]:
    """Construct one fully bound train+validation experiment."""
    protocol_binding = load_protocol_binding(args.protocol)
    if manifest_cell["protocol_sha256"] != protocol_binding["sha256"]:
        raise ValueError("manifest/protocol SHA256 mismatch")
    if manifest_cell["study_id"] != protocol_binding["study_id"]:
        raise ValueError("manifest/protocol study_id mismatch")
    validate_replicate_binding(args, protocol_binding, manifest_cell)
    statistics_binding = load_statistics_binding(
        args.target_statistics,
        expected_protocol_sha256=protocol_binding["sha256"],
        expected_study_id=protocol_binding["study_id"],
    )

    task: Literal["common", "full"] = args.task
    preset_class = DrivAerMLTransferCommonPreset if task == "common" else DrivAerMLTransferFullPreset
    preset = preset_class(
        statistics_artifact=args.target_statistics,
        coordinate_frame=args.coordinate_frame,
        expected_manifest_sha256=manifest_cell["raw_file_sha256"],
        expected_train_subset_size=args.sample_size,
    )

    source_checkpoint = None
    source_checkpoint_sha256 = None
    model_params = dict(CHECKPOINT_ARCHITECTURE)
    if args.strategy != "scratch":
        source_checkpoint = source_checkpoint_path(args, protocol_binding)
        source_checkpoint_sha256 = protocol_binding["source_primary_sha256"]
        model_params["initializers"] = [transfer_initializer(args)]

    budget = training_budget(args)
    expected_updates = int(budget["expected_updates"])
    train_dataset = preset.build_dataset(
        split="train",
        root=str(args.dataset_root),
        model_kind=MODEL_KIND,
        wrappers=[
            SubsetWrapperConfig(
                kind="noether.data.base.wrappers.SubsetWrapper",
                indices=manifest_cell["base_dataset_indices"],
            )
        ],
        seed=None,
    )
    val_dataset = preset.build_dataset(
        split="val",
        root=str(args.dataset_root),
        model_kind=MODEL_KIND,
        seed=args.eval_point_seed,
    )

    optimizer = preset.build_optimizer(
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        clip_grad_norm=0.25,
        warmup_percent=0.05,
        end_lr=args.end_learning_rate,
    )
    optimizer = optimizer.model_copy(update={"param_group_modifiers_config": build_lr_modifiers(args)})

    trainer_params: dict[str, Any] = {
        "field_weights": FIELD_WEIGHTS[task],
        "precision": args.precision,
        "find_unused_params": args.strategy in {"linear_probe", "gradual_unfreeze"},
        "static_graph": False,
    }
    if budget["max_updates"] is not None:
        trainer_params["max_epochs"] = None
        trainer_params["max_updates"] = budget["max_updates"]

    run_id = args.run_id or (
        f"mf-{task}-{args.strategy}-r{args.replicate}-n{args.sample_size}"
        f"-m{manifest_cell['payload_sha256'][:10]}-s{args.model_seed}"
        f"-{args.coordinate_frame}-{args.budget}"
    )
    config = preset.build_config(
        model_kind=MODEL_KIND,
        model_params=model_params,
        optimizer=optimizer,
        trainer_kind=TRAINER_KIND,
        trainer_params=trainer_params,
        dataset_root=str(args.dataset_root),
        output_path=str(args.output_path),
        datasets=[],
        extra_datasets={"train": train_dataset, "val": val_dataset},
        callbacks_override=build_callbacks(args, expected_updates),
        accelerator=args.accelerator,
        max_epochs=int(budget["max_epochs"] or 1),
        batch_size=args.effective_batch_size,
        seed=args.model_seed,
        name=f"drivaerml-{task}-{args.strategy}",
        run_id=run_id,
        stage_name="train",
        num_workers=args.num_workers,
        store_code_in_output=True,
    )
    audit = {
        "run_id": run_id,
        "study_id": protocol_binding["study_id"],
        "protocol_path": protocol_binding["path"],
        "protocol_sha256": protocol_binding["sha256"],
        "task": task,
        "strategy": args.strategy,
        "method": METHOD_BY_STRATEGY[args.strategy],
        "replicate": args.replicate,
        "budget": budget,
        "manifest": manifest_cell,
        "target_statistics": statistics_binding,
        "coordinate_frame": args.coordinate_frame,
        "source_checkpoint": str(source_checkpoint.resolve()) if source_checkpoint else None,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "model_seed": args.model_seed,
        "data_seed": None,
        "training_pipeline_seed": None,
        "data_loader_seed": args.model_seed,
        "validation_seed": args.eval_point_seed,
        "evaluation_seed": args.eval_point_seed,
        "test_dataset_in_training_config": "test" in config.datasets,
    }
    if audit["test_dataset_in_training_config"]:
        raise AssertionError("strict training config unexpectedly contains the test dataset")
    return config, audit


def parse_args() -> argparse.Namespace:
    """Parse one frozen experiment cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-statistics", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, default="common")
    parser.add_argument("--strategy", choices=TRANSFER_STRATEGIES, default="scratch")
    parser.add_argument("--sample-size", type=int, choices=(25, 50, 100, 200, 400), required=True)
    parser.add_argument("--budget", choices=BUDGETS, default="compute_matched")
    parser.add_argument("--smoke-updates", type=int, default=10)
    parser.add_argument("--coordinate-frame", choices=("native", "shapenet"), default="shapenet")
    parser.add_argument("--replicate", type=int, choices=range(8), required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--eval-point-seed", type=int, default=4242)
    parser.add_argument("--run-id")

    parser.add_argument("--source-output-path", type=Path, default=Path("/home/feng/Projects/ABUPT/outputs"))
    parser.add_argument("--source-run-id", default="2026-04-25_7d0mv")
    parser.add_argument("--source-stage-name", default="train")
    parser.add_argument("--source-model-name", default="ab_upt")
    parser.add_argument("--source-model-info")
    parser.add_argument("--source-checkpoint-tag", default="latest")
    parser.add_argument("--expected-source-sha256")

    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--end-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--body-lr-multiplier", type=float)
    parser.add_argument("--decoder-lr-multiplier", type=float)
    parser.add_argument("--effective-batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--accelerator", choices=("cpu", "gpu", "mps"), default="gpu")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.smoke_updates < 1:
        parser.error("--smoke-updates must be positive")
    if (
        not math.isfinite(args.learning_rate)
        or not math.isfinite(args.end_learning_rate)
        or args.learning_rate <= 0
        or args.end_learning_rate <= 0
    ):
        parser.error("learning rates must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be finite and non-negative")
    return args


def main() -> None:
    """Audit-print or execute one strict train+validation cell."""
    args = parse_args()
    protocol_binding = load_protocol_binding(args.protocol)
    if not args.dry_run:
        require_frozen_protocol_for_execution(protocol_binding)
    manifest_cell = load_manifest_cell(
        args.manifest,
        args.sample_size,
        expected_protocol_sha256=protocol_binding["sha256"],
        expected_study_id=protocol_binding["study_id"],
    )
    git_state = implementation_git_state()
    validate_implementation_binding(manifest_cell, git_state)
    config, audit = build_experiment_config(args, manifest_cell)
    audit["implementation_git_state"] = git_state
    if args.dry_run:
        output = {
            "audit": audit,
            "resolved_config": config.model_dump(mode="json", exclude_computed_fields=True),
        }
        print(yaml.safe_dump(output, sort_keys=False))
        return
    HydraRunner().main(device=accelerator_to_device(args.accelerator), config=config)
    sidecar = write_training_provenance_sidecar(args, config, audit, git_state)
    print(
        yaml.safe_dump(
            {
                "training_provenance_sidecar": str(sidecar.resolve()),
                "training_provenance_sidecar_sha256": sha256_file(sidecar),
            },
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
