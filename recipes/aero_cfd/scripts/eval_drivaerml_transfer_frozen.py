# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Run a frozen one-shot DrivAerML validation/test export for one trained model."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from aero_cfd.callbacks.paired_metrics_export import PairedMetricsExportCallbackConfig
from aero_cfd.presets.drivaerml_transfer import (
    DrivAerMLTransferCommonPreset,
    DrivAerMLTransferFullPreset,
)
from noether.core.distributed.utils import accelerator_to_device
from noether.core.schemas.initializers import PreviousRunInitializerConfig
from noether.data.datasets.cfd.caeml.drivaerml.split import DrivAerMLDefaultSplitIDs
from noether.inference.runners.inference_runner import InferenceRunner

from recipes.aero_cfd.scripts.run_drivaerml_transfer_strict import (
    BUDGETS,
    CHECKPOINT_ARCHITECTURE,
    FIELD_WEIGHTS,
    METHOD_BY_STRATEGY,
    MODEL_KIND,
    PROVENANCE_FILENAME,
    TASKS,
    TRAINER_KIND,
    atomic_write_json,
    load_manifest_cell,
    load_protocol_binding,
    load_statistics_binding,
    require_frozen_protocol_for_execution,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _implementation_git_state(repo_root: Path) -> tuple[str, bool]:
    """Return the evaluator commit and whether its worktree is dirty.

    Args:
        repo_root: Git repository containing the evaluation implementation.

    Returns:
        The lowercase Git commit and a Boolean dirty-worktree indicator.
    """
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError(f"invalid evaluator Git commit {commit!r}")
    return commit, bool(status.strip())


def _target_checkpoint(args: argparse.Namespace) -> Path:
    """Resolve and verify the exact frozen target-model checkpoint."""
    filename = f"{args.target_model_name}_cp={args.target_checkpoint_tag}_model.th"
    path = args.target_output_path / args.target_run_id / args.target_stage_name / "checkpoints" / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_training_provenance(
    args: argparse.Namespace,
    *,
    protocol_binding: dict[str, Any],
    manifest_cell: dict[str, Any],
    statistics_binding: dict[str, str],
    target_checkpoint: Path,
) -> tuple[dict[str, Any], Path, str, str]:
    """Validate the immutable training sidecar and every recorded model checkpoint."""
    sidecar_path = args.target_output_path / args.target_run_id / args.target_stage_name / PROVENANCE_FILENAME
    raw = sidecar_path.read_bytes()
    sidecar = json.loads(raw.decode("utf-8"))
    if not isinstance(sidecar, dict):
        raise TypeError("training provenance sidecar root must be a mapping")
    if sidecar.get("schema_version") != 1 or sidecar.get("kind") != "drivaerml_transfer_training_provenance":
        raise ValueError("unsupported training provenance sidecar schema")

    replicate_entry = protocol_binding["replicates"][args.replicate]
    expected = {
        "study_id": protocol_binding["study_id"],
        "protocol_sha256": protocol_binding["sha256"],
        "implementation_git_commit": manifest_cell["implementation_git_commit"],
        "implementation_git_dirty": manifest_cell["implementation_git_dirty"],
        "manifest_raw_sha256": manifest_cell["raw_file_sha256"],
        "manifest_payload_sha256": manifest_cell["payload_sha256"],
        "target_statistics_sha256": statistics_binding["sha256"],
        "task": args.task,
        "method": args.method,
        "replicate": args.replicate,
        "train_sample_size": args.sample_size,
        "coordinate_frame": args.coordinate_frame,
        "budget": args.budget,
        "subset_seed": replicate_entry["subset_seed"],
        "model_seed": replicate_entry["model_seed"],
        "data_seed": None,
        "training_pipeline_seed": None,
        "data_loader_seed": replicate_entry["model_seed"],
        "validation_seed": args.eval_point_seed,
        "evaluation_seed": args.eval_point_seed,
        "run_id": args.target_run_id,
        "stage_name": args.target_stage_name,
    }
    for key, expected_value in expected.items():
        actual_value = sidecar.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"training provenance mismatch for {key}: expected={expected_value!r}, actual={actual_value!r}"
            )

    strategy = sidecar.get("strategy")
    if METHOD_BY_STRATEGY.get(strategy) != sidecar.get("method"):
        raise ValueError(
            f"training provenance strategy/method mismatch: strategy={strategy!r}, method={sidecar.get('method')!r}"
        )
    source_checkpoint = sidecar.get("source_checkpoint")
    source_checkpoint_sha256 = sidecar.get("source_checkpoint_sha256")
    if strategy == "scratch":
        if source_checkpoint is not None or source_checkpoint_sha256 is not None:
            raise ValueError("scratch training provenance must not record a source checkpoint")
    else:
        if (
            not isinstance(source_checkpoint, str)
            or not Path(source_checkpoint).is_absolute()
            or source_checkpoint_sha256 != protocol_binding["source_primary_sha256"]
        ):
            raise ValueError(
                "transfer training provenance must bind the protocol primary source checkpoint: "
                f"source={source_checkpoint!r}, sha256={source_checkpoint_sha256!r}, "
                f"protocol={protocol_binding['source_primary_sha256']}"
            )
    if not isinstance(sidecar.get("implementation_git_commit"), str):
        raise ValueError("training provenance has no implementation Git commit")
    if not isinstance(sidecar.get("implementation_git_dirty"), bool):
        raise ValueError("training provenance has no implementation Git dirty state")
    if not isinstance(sidecar.get("resolved_config_sha256"), str) or len(sidecar["resolved_config_sha256"]) != 64:
        raise ValueError("training provenance has no valid resolved config SHA256")

    checkpoint_hashes = sidecar.get("model_checkpoints_sha256")
    if not isinstance(checkpoint_hashes, dict) or not checkpoint_hashes:
        raise ValueError("training provenance has no model checkpoint hashes")
    checkpoint_dir = target_checkpoint.parent
    actual_checkpoint_names = {path.name for path in checkpoint_dir.glob("*_model.th")}
    if actual_checkpoint_names != set(checkpoint_hashes):
        raise ValueError(
            "training provenance checkpoint set mismatch: "
            f"recorded={sorted(checkpoint_hashes)}, actual={sorted(actual_checkpoint_names)}"
        )
    actual_hashes: dict[str, str] = {}
    for basename, expected_sha256 in sorted(checkpoint_hashes.items()):
        if not isinstance(basename, str) or not isinstance(expected_sha256, str):
            raise ValueError("training provenance checkpoint hash mapping is malformed")
        actual_sha256 = sha256_file(checkpoint_dir / basename)
        actual_hashes[basename] = actual_sha256
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"training provenance checkpoint SHA256 mismatch for {basename}: "
                f"expected={expected_sha256}, actual={actual_sha256}"
            )

    target_sha256 = actual_hashes[target_checkpoint.name]
    if target_sha256 != args.expected_target_sha256:
        raise ValueError(
            f"target checkpoint SHA256 mismatch: expected={args.expected_target_sha256}, actual={target_sha256}"
        )
    return sidecar, sidecar_path, hashlib.sha256(raw).hexdigest(), target_sha256


def build_eval_config(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    """Build one official-split inference config bound to frozen artifacts."""
    if args.evaluation_split == "test" and not args.confirm_test_release:
        raise ValueError("test evaluation requires explicit --confirm-test-release")
    evaluation_commit, evaluation_dirty = _implementation_git_state(REPO_ROOT)
    if evaluation_dirty:
        raise ValueError("frozen evaluation requires a clean implementation Git worktree")
    protocol_binding = load_protocol_binding(args.protocol)
    manifest_cell = load_manifest_cell(
        args.manifest,
        args.sample_size,
        expected_protocol_sha256=protocol_binding["sha256"],
        expected_study_id=protocol_binding["study_id"],
    )
    statistics_binding = load_statistics_binding(
        args.target_statistics,
        expected_protocol_sha256=protocol_binding["sha256"],
        expected_study_id=protocol_binding["study_id"],
    )
    target_checkpoint = _target_checkpoint(args)
    sidecar, sidecar_path, sidecar_sha256, target_checkpoint_sha256 = _load_training_provenance(
        args,
        protocol_binding=protocol_binding,
        manifest_cell=manifest_cell,
        statistics_binding=statistics_binding,
        target_checkpoint=target_checkpoint,
    )
    preset_class = DrivAerMLTransferCommonPreset if args.task == "common" else DrivAerMLTransferFullPreset
    preset = preset_class(
        statistics_artifact=args.target_statistics,
        coordinate_frame=args.coordinate_frame,
        expected_manifest_sha256=manifest_cell["raw_file_sha256"],
        expected_train_subset_size=args.sample_size,
    )

    model_params = dict(CHECKPOINT_ARCHITECTURE)
    model_params["initializers"] = [
        PreviousRunInitializerConfig(
            output_path=args.target_output_path,
            run_id=args.target_run_id,
            stage_name=args.target_stage_name,
            model_name=args.target_model_name,
            checkpoint_tag=args.target_checkpoint_tag,
        )
    ]
    evaluation_dataset = preset.build_dataset(
        split=args.evaluation_split,
        root=str(args.dataset_root),
        model_kind=MODEL_KIND,
        seed=args.eval_point_seed,
        num_geometry_points=args.num_geometry_points,
        num_geometry_supernodes=args.num_geometry_supernodes,
        num_surface_anchor_points=args.num_surface_queries,
        num_volume_anchor_points=args.num_volume_queries,
    )
    callback = PairedMetricsExportCallbackConfig(
        every_n_epochs=1,
        dataset_key=args.evaluation_split,
        batch_size=1,
        forward_properties=preset.forward_properties(MODEL_KIND),
        output_csv=str(args.output_csv),
        method=sidecar["method"],
        replicate=str(sidecar["replicate"]),
        train_sample_size=sidecar["train_sample_size"],
    )
    optimizer = preset.build_optimizer(lr=5e-5, end_lr=None)
    eval_run_id = args.eval_run_id or (
        f"eval-{args.evaluation_split}-{args.target_run_id}-{args.target_checkpoint_tag}"
        f"-{args.coordinate_frame}-q{args.num_surface_queries}"
    )
    config = preset.build_config(
        model_kind=MODEL_KIND,
        model_params=model_params,
        optimizer=optimizer,
        trainer_kind=TRAINER_KIND,
        trainer_params={
            "field_weights": FIELD_WEIGHTS[args.task],
            "precision": args.precision,
            "find_unused_params": False,
            "static_graph": False,
        },
        dataset_root=str(args.dataset_root),
        output_path=str(args.eval_output_path),
        datasets=[],
        extra_datasets={args.evaluation_split: evaluation_dataset},
        callbacks_override=[callback],
        accelerator=args.accelerator,
        max_epochs=0,
        batch_size=1,
        seed=args.eval_point_seed,
        name=f"drivaerml-{args.task}-frozen-{args.evaluation_split}",
        run_id=eval_run_id,
        stage_name=args.evaluation_split,
        num_workers=args.num_workers,
        store_code_in_output=True,
    )
    official_ids = getattr(DrivAerMLDefaultSplitIDs(), args.evaluation_split)
    audit = {
        "evaluation_implementation_git_commit": evaluation_commit,
        "evaluation_implementation_git_dirty": evaluation_dirty,
        "training_implementation_git_commit": sidecar["implementation_git_commit"],
        "training_implementation_git_dirty": sidecar["implementation_git_dirty"],
        "training_provenance_sidecar": str(sidecar_path.resolve()),
        "training_provenance_sidecar_sha256": sidecar_sha256,
        "target_checkpoint": str(target_checkpoint.resolve()),
        "target_checkpoint_sha256": target_checkpoint_sha256,
        "sidecar_target_checkpoint_sha256": sidecar["model_checkpoints_sha256"][target_checkpoint.name],
        "target_run_id": args.target_run_id,
        "target_checkpoint_tag": args.target_checkpoint_tag,
        "protocol_sha256": protocol_binding["sha256"],
        "manifest_raw_sha256": manifest_cell["raw_file_sha256"],
        "manifest_payload_sha256": manifest_cell["payload_sha256"],
        "target_statistics_sha256": statistics_binding["sha256"],
        "task": sidecar["task"],
        "strategy": sidecar["strategy"],
        "method": sidecar["method"],
        "replicate": sidecar["replicate"],
        "train_sample_size": sidecar["train_sample_size"],
        "coordinate_frame": sidecar["coordinate_frame"],
        "budget": sidecar["budget"],
        "model_seed": sidecar["model_seed"],
        "training_pipeline_seed": sidecar["training_pipeline_seed"],
        "data_loader_seed": sidecar["data_loader_seed"],
        "validation_seed": sidecar["validation_seed"],
        "evaluation_seed": sidecar["evaluation_seed"],
        "evaluation_split": args.evaluation_split,
        "test_release_confirmed": args.confirm_test_release,
        "official_evaluation_design_ids": official_ids,
        "evaluation_design_count": len(official_ids),
        "official_test_design_ids": official_ids if args.evaluation_split == "test" else None,
        "test_design_count": len(official_ids) if args.evaluation_split == "test" else None,
        "surface_queries_per_design": args.num_surface_queries,
        "volume_queries_per_design": args.num_volume_queries,
        "output_csv": str(args.output_csv.resolve()),
        "datasets_in_eval_config": sorted(config.datasets),
    }
    if set(config.datasets) != {args.evaluation_split}:
        raise AssertionError(
            f"frozen evaluation config must contain {args.evaluation_split!r} only, got {config.datasets.keys()}"
        )
    return config, audit


def parse_args() -> argparse.Namespace:
    """Parse one frozen evaluation cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-statistics", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--coordinate-frame", choices=("native", "shapenet"), default="shapenet")
    parser.add_argument("--sample-size", type=int, choices=(25, 50, 100, 200, 400), required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--replicate", type=int, choices=range(8), required=True)
    parser.add_argument("--budget", choices=BUDGETS, required=True)
    parser.add_argument("--evaluation-split", choices=("val", "test"), default="val")
    parser.add_argument("--confirm-test-release", action="store_true")

    parser.add_argument("--target-output-path", type=Path, required=True)
    parser.add_argument("--target-run-id", required=True)
    parser.add_argument("--target-stage-name", default="train")
    parser.add_argument("--target-model-name", default="ab_upt")
    parser.add_argument("--target-checkpoint-tag", default="latest")
    parser.add_argument("--expected-target-sha256", required=True)

    parser.add_argument("--eval-output-path", type=Path, required=True)
    parser.add_argument("--eval-run-id")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--num-geometry-points", type=int, default=16384)
    parser.add_argument("--num-geometry-supernodes", type=int, default=1024)
    parser.add_argument("--num-surface-queries", type=int, default=16384)
    parser.add_argument("--num-volume-queries", type=int, default=16384)
    parser.add_argument("--eval-point-seed", type=int, default=4242)
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--accelerator", choices=("cpu", "gpu", "mps"), default="gpu")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.evaluation_split == "test" and not args.confirm_test_release:
        parser.error("--evaluation-split test requires explicit --confirm-test-release")
    for name in (
        "num_geometry_points",
        "num_geometry_supernodes",
        "num_surface_queries",
        "num_volume_queries",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    """Audit-print or execute one frozen validation/test export."""
    args = parse_args()
    if not args.dry_run:
        require_frozen_protocol_for_execution(load_protocol_binding(args.protocol))
    config, audit = build_eval_config(args)
    if args.dry_run:
        print(
            yaml.safe_dump(
                {
                    "audit": audit,
                    "resolved_config": config.model_dump(mode="json", exclude_computed_fields=True),
                },
                sort_keys=False,
            )
        )
        return
    InferenceRunner.main(device=accelerator_to_device(args.accelerator), config=config)
    if not args.output_csv.is_file():
        raise FileNotFoundError(f"evaluation completed without metrics CSV: {args.output_csv}")
    audit["output_csv_sha256"] = sha256_file(args.output_csv)
    audit_path = args.output_csv.with_suffix(args.output_csv.suffix + ".audit.json")
    atomic_write_json(audit_path, audit)
    print(
        yaml.safe_dump(
            {
                "evaluation_audit": str(audit_path.resolve()),
                "evaluation_audit_sha256": sha256_file(audit_path),
            },
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
