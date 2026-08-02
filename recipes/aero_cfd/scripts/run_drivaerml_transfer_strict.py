# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Run one leakage-safe ShapeNet-Car to DrivAerML AB-UPT experiment.

This confirmatory entry point deliberately excludes the test dataset.  It binds
each run to a hashed nested-subset manifest, a train-subset-only statistics
artifact, a coordinate frame, a geometry rendering, and (for transfer methods)
a checksum-verified source checkpoint.

Every rule lives in :mod:`aero_cfd.multi_fidelity`; this module only parses
arguments and reports what it published.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from aero_cfd.multi_fidelity.experiment import (
    BUDGETS,
    CHECKPOINT_POSITION_SCALE,
    CHECKPOINT_SUPERNODE_RADIUS,
    SAMPLE_SIZES,
    STRATEGIES,
    TASKS,
    GeometryRendering,
    SourceCheckpoint,
    TrainingRequest,
    build_experiment,
)
from aero_cfd.multi_fidelity.integrity import read_git_state, sha256_file
from aero_cfd.multi_fidelity.manifest import load_manifest_cell
from aero_cfd.multi_fidelity.protocol import load_protocol_binding, require_frozen_protocol_for_execution
from aero_cfd.multi_fidelity.provenance import write_training_provenance
from noether.core.distributed.utils import accelerator_to_device
from noether.training.runners import HydraRunner

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse one frozen experiment cell.

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
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, default="common")
    parser.add_argument("--strategy", choices=STRATEGIES, default="scratch")
    parser.add_argument("--sample-size", type=int, choices=SAMPLE_SIZES, required=True)
    parser.add_argument("--budget", choices=BUDGETS, default="compute_matched")
    parser.add_argument("--smoke-updates", type=int, default=10)
    parser.add_argument("--coordinate-frame", choices=("native", "shapenet"), default="shapenet")
    parser.add_argument(
        "--position-scale",
        type=float,
        default=CHECKPOINT_POSITION_SCALE,
        help=(
            "Normalized position upper bound; positions map linearly to [0, scale]. "
            f"Transfer runs must keep the pretrained {CHECKPOINT_POSITION_SCALE:g} because the sincos and "
            "RoPE frequency buffers are restored from the source checkpoint."
        ),
    )
    parser.add_argument(
        "--supernode-radius",
        type=float,
        default=CHECKPOINT_SUPERNODE_RADIUS,
        help=(
            "Supernode-pooling radius in normalized position units. It must be rescaled whenever the "
            "target geometry occupies a different fraction of the normalization box than the source did, "
            "otherwise the radius graph saturates the degree cap and stops carrying shape information."
        ),
    )
    parser.add_argument("--replicate", type=int, choices=range(8), required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--eval-point-seed", type=int, default=4242)
    parser.add_argument("--run-id")

    parser.add_argument("--source-output-path", type=Path, required=True)
    parser.add_argument("--source-run-id", default="2026-04-25_7d0mv")
    parser.add_argument("--source-stage-name", default="train")
    parser.add_argument("--source-model-name", default="ab_upt")
    parser.add_argument("--source-model-info")
    parser.add_argument("--source-checkpoint-tag", default="latest")

    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--end-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--effective-batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--accelerator", choices=("cpu", "gpu", "mps"), default="gpu")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def request_from_args(args: argparse.Namespace) -> TrainingRequest:
    """Build the validated request described by one argument vector.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The validated :class:`~aero_cfd.multi_fidelity.experiment.TrainingRequest`.
    """
    return TrainingRequest(
        dataset_root=args.dataset_root,
        protocol_path=args.protocol,
        manifest_path=args.manifest,
        statistics_path=args.target_statistics,
        output_path=args.output_path,
        task=args.task,
        strategy=args.strategy,
        sample_size=args.sample_size,
        budget=args.budget,
        smoke_updates=args.smoke_updates,
        coordinate_frame=args.coordinate_frame,
        geometry=GeometryRendering(
            position_scale=args.position_scale,
            supernode_radius=args.supernode_radius,
        ),
        replicate=args.replicate,
        model_seed=args.model_seed,
        eval_point_seed=args.eval_point_seed,
        run_id=args.run_id,
        source=SourceCheckpoint(
            output_path=args.source_output_path,
            run_id=args.source_run_id,
            stage_name=args.source_stage_name,
            model_name=args.source_model_name,
            checkpoint_tag=args.source_checkpoint_tag,
            model_info=args.source_model_info,
        ),
        learning_rate=args.learning_rate,
        end_learning_rate=args.end_learning_rate,
        weight_decay=args.weight_decay,
        effective_batch_size=args.effective_batch_size,
        precision=args.precision,
        accelerator=args.accelerator,
        num_workers=args.num_workers,
    )


def main(argv: list[str] | None = None) -> None:
    """Audit-print or execute one strict train+validation cell.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv``.
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
    git_state = read_git_state(REPO_ROOT)
    manifest_cell.require_implementation(git_state)

    experiment = build_experiment(request, manifest_cell, protocol)
    if args.dry_run:
        print(
            yaml.safe_dump(
                {
                    "audit": {
                        **experiment.audit,
                        "implementation_git_state": {
                            "commit": git_state.commit,
                            "dirty": git_state.dirty,
                            "status_sha256": git_state.status_sha256,
                        },
                    },
                    "resolved_config": experiment.config.model_dump(mode="json", exclude_computed_fields=True),
                },
                sort_keys=False,
            )
        )
        return

    HydraRunner().main(device=accelerator_to_device(args.accelerator), config=experiment.config)
    sidecar = write_training_provenance(
        request,
        experiment.audit,
        experiment.config.model_dump(mode="python", exclude_computed_fields=True),
        git_state,
    )
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
