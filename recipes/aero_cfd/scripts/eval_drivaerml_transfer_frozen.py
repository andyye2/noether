# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Run one frozen single-split DrivAerML export for one trained model.

The scientific labels are read from the training provenance sidecar and must
match what the caller claims, so a metric row can never be attributed to the
wrong cell.  Every rule lives in :mod:`aero_cfd.multi_fidelity`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from aero_cfd.multi_fidelity.evaluation import (
    DEFAULT_EVALUATION_POINTS,
    DEFAULT_GEOMETRY_POINTS,
    DEFAULT_GEOMETRY_SUPERNODES,
    EvaluationRequest,
    TargetModel,
    build_evaluation,
)
from aero_cfd.multi_fidelity.experiment import (
    BUDGETS,
    CHECKPOINT_POSITION_SCALE,
    CHECKPOINT_SUPERNODE_RADIUS,
    SAMPLE_SIZES,
    TASKS,
    GeometryRendering,
)
from aero_cfd.multi_fidelity.integrity import atomic_write_json, read_git_state, sha256_file
from aero_cfd.multi_fidelity.manifest import load_manifest_cell
from aero_cfd.multi_fidelity.protocol import load_protocol_binding, require_frozen_protocol_for_execution
from aero_cfd.multi_fidelity.statistics import load_statistics_binding
from noether.core.distributed.utils import accelerator_to_device
from noether.inference.runners.inference_runner import InferenceRunner

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse one frozen evaluation cell.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-statistics", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--coordinate-frame", choices=("native", "shapenet"), default="shapenet")
    parser.add_argument(
        "--position-scale",
        type=float,
        default=CHECKPOINT_POSITION_SCALE,
        help="Normalized position upper bound; must equal the value recorded in the training sidecar.",
    )
    parser.add_argument(
        "--supernode-radius",
        type=float,
        default=CHECKPOINT_SUPERNODE_RADIUS,
        help="Supernode-pooling radius; must equal the value recorded in the training sidecar.",
    )
    parser.add_argument(
        "--wall-distance-feature",
        action="store_true",
        help="Render the volume wall-distance input feature; must equal the value recorded in the training sidecar.",
    )
    parser.add_argument("--sample-size", type=int, choices=SAMPLE_SIZES, required=True)
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
    parser.add_argument("--num-geometry-points", type=int, default=DEFAULT_GEOMETRY_POINTS)
    parser.add_argument("--num-geometry-supernodes", type=int, default=DEFAULT_GEOMETRY_SUPERNODES)
    parser.add_argument("--num-surface-queries", type=int, default=DEFAULT_EVALUATION_POINTS)
    parser.add_argument("--num-volume-queries", type=int, default=DEFAULT_EVALUATION_POINTS)
    parser.add_argument("--eval-point-seed", type=int, default=4242)
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--accelerator", choices=("cpu", "gpu", "mps"), default="gpu")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.evaluation_split == "test" and not args.confirm_test_release:
        parser.error("--evaluation-split test requires explicit --confirm-test-release")
    return args


def request_from_args(args: argparse.Namespace) -> EvaluationRequest:
    """Build the validated request described by one argument vector.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The validated :class:`~aero_cfd.multi_fidelity.evaluation.EvaluationRequest`.
    """
    return EvaluationRequest(
        dataset_root=args.dataset_root,
        protocol_path=args.protocol,
        manifest_path=args.manifest,
        statistics_path=args.target_statistics,
        target=TargetModel(
            output_path=args.target_output_path,
            run_id=args.target_run_id,
            stage_name=args.target_stage_name,
            model_name=args.target_model_name,
            checkpoint_tag=args.target_checkpoint_tag,
            expected_sha256=args.expected_target_sha256,
        ),
        task=args.task,
        method=args.method,
        replicate=args.replicate,
        sample_size=args.sample_size,
        budget=args.budget,
        coordinate_frame=args.coordinate_frame,
        geometry=GeometryRendering(
            position_scale=args.position_scale,
            supernode_radius=args.supernode_radius,
        ),
        wall_distance_feature=args.wall_distance_feature,
        evaluation_split=args.evaluation_split,
        confirm_test_release=args.confirm_test_release,
        eval_output_path=args.eval_output_path,
        output_csv=args.output_csv,
        eval_run_id=args.eval_run_id,
        num_geometry_points=args.num_geometry_points,
        num_geometry_supernodes=args.num_geometry_supernodes,
        num_surface_queries=args.num_surface_queries,
        num_volume_queries=args.num_volume_queries,
        eval_point_seed=args.eval_point_seed,
        precision=args.precision,
        accelerator=args.accelerator,
        num_workers=args.num_workers,
    )


def main(argv: list[str] | None = None) -> None:
    """Audit-print or execute one frozen validation/test export.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv``.

    Raises:
        FileNotFoundError: If inference completed without writing metrics.
    """
    args = parse_args(argv)
    request = request_from_args(args)
    protocol = load_protocol_binding(request.protocol_path)
    if not args.dry_run:
        require_frozen_protocol_for_execution(protocol)

    manifest_cell = load_manifest_cell(
        request.manifest_path,
        request.sample_size,
        expected_protocol_sha256=protocol.sha256,
        expected_study_id=protocol.study_id,
    )
    statistics = load_statistics_binding(request.statistics_path, protocol=protocol)
    config, audit = build_evaluation(
        request,
        protocol=protocol,
        manifest_cell=manifest_cell,
        statistics=statistics,
        evaluator_git_state=read_git_state(REPO_ROOT),
    )
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
    if not request.output_csv.is_file():
        raise FileNotFoundError(f"evaluation completed without metrics CSV: {request.output_csv}")
    audit["output_csv_sha256"] = sha256_file(request.output_csv)
    audit_path = request.output_csv.with_suffix(request.output_csv.suffix + ".audit.json")
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
