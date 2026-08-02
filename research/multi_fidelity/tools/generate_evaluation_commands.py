# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Generate frozen evaluation commands from training provenance sidecars.

Cells are discovered, never declared: the generator walks the training output
root, reads each ``training_provenance.json``, and emits one command per
completed run.  Scientific labels therefore cannot be attached by hand, and
the geometry rendering of a run is carried into its evaluation instead of
silently falling back to the frozen defaults.

Example:

    .. code-block:: bash

        uv run python -m research.multi_fidelity.tools.generate_evaluation_commands \\
            --repo-root /scratch/andyye2/ABUPT/multi_fidelity \\
            --dataset-root /scratch/andyye2/data/drivaerml_subsampled_10x \\
            --training-output-root <outputs>/n100-r0-paper \\
            --eval-output-root     <outputs>/n100-r0-paper-eval \\
            --metrics-root         <artifacts>/metrics \\
            --split val \\
            --output <artifacts>/commands/evaluation.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from aero_cfd.multi_fidelity.provenance import PROVENANCE_FILENAME, TrainingProvenance, load_training_provenance

from .commands import EVALUATION_SCRIPT, uv_command, write_command_file

CHECKPOINT_TAGS = ("latest", "best_model.loss.val.total")


@dataclass(frozen=True)
class EvaluationCell:
    """One discovered training result selected for evaluation.

    Attributes:
        provenance: Validated training provenance sidecar.
        checkpoint_tag: Checkpoint tag to evaluate.
        checkpoint_sha256: SHA256 recorded for that checkpoint.
    """

    provenance: TrainingProvenance
    checkpoint_tag: str
    checkpoint_sha256: str

    @property
    def run_id(self) -> str:
        """Return the training run identifier."""
        return str(self.provenance["run_id"])

    @property
    def target_output_path(self) -> Path:
        """Return the training output root that contains this run."""
        return Path(self.provenance.path).parents[2]

    def sort_key(self) -> tuple[str, int, int, str, str]:
        """Return a deterministic ordering key for the command list."""
        return (
            str(self.provenance["method"]),
            int(self.provenance["replicate"]),
            int(self.provenance["train_sample_size"]),
            self.run_id,
            self.checkpoint_tag,
        )


def discover_cells(training_output_root: Path, checkpoint_tags: tuple[str, ...]) -> list[EvaluationCell]:
    """Find every completed run under one training output root.

    Args:
        training_output_root: Root that the training runs wrote into.
        checkpoint_tags: Checkpoint tags to evaluate when a run recorded them.

    Returns:
        Discovered cells in a deterministic order.
    """
    cells: list[EvaluationCell] = []
    for sidecar_path in sorted(training_output_root.glob(f"*/*/{PROVENANCE_FILENAME}")):
        provenance = load_training_provenance(sidecar_path)
        hashes: dict[str, str] = provenance["model_checkpoints_sha256"]
        for tag in checkpoint_tags:
            basename = f"ab_upt_cp={tag}_model.th"
            if basename in hashes:
                cells.append(
                    EvaluationCell(provenance=provenance, checkpoint_tag=tag, checkpoint_sha256=hashes[basename])
                )
    return sorted(cells, key=EvaluationCell.sort_key)


def command_for_cell(
    cell: EvaluationCell,
    *,
    repo_root: Path,
    protocol_path: Path,
    dataset_root: Path,
    eval_output_root: Path,
    metrics_root: Path,
    split: str,
) -> str:
    """Render the evaluation command line of one discovered cell.

    Args:
        cell: Discovered training result.
        repo_root: Repository root on the executing host.
        protocol_path: Path of the frozen preregistration.
        dataset_root: Root of the DrivAerML dataset.
        eval_output_root: Root the evaluation runs write into.
        metrics_root: Directory the per-design metric tables are written to.
        split: Official split to score.

    Returns:
        One shell-quoted command line.
    """
    provenance = cell.provenance
    tag_slug = cell.checkpoint_tag.replace(".", "-")
    csv_path = (
        metrics_root
        / split
        / str(provenance["task"])
        / (
            f"{provenance['method']}_r{provenance['replicate']}_n{provenance['train_sample_size']}"
            f"_{provenance['coordinate_frame']}_{tag_slug}_{cell.run_id}.csv"
        )
    )
    arguments = [
        "--protocol",
        str(protocol_path),
        "--dataset-root",
        str(dataset_root),
        "--manifest",
        str(provenance["manifest_path"]),
        "--target-statistics",
        str(provenance["target_statistics_path"]),
        "--task",
        str(provenance["task"]),
        "--coordinate-frame",
        str(provenance["coordinate_frame"]),
        "--position-scale",
        f"{float(provenance['position_scale']):g}",
        "--supernode-radius",
        f"{float(provenance['supernode_radius']):g}",
        "--sample-size",
        str(provenance["train_sample_size"]),
        "--method",
        str(provenance["method"]),
        "--replicate",
        str(provenance["replicate"]),
        "--budget",
        str(provenance["budget"]),
        "--evaluation-split",
        split,
        "--target-output-path",
        str(cell.target_output_path),
        "--target-run-id",
        cell.run_id,
        "--target-stage-name",
        str(provenance["stage_name"]),
        "--target-checkpoint-tag",
        cell.checkpoint_tag,
        "--expected-target-sha256",
        cell.checkpoint_sha256,
        "--eval-output-path",
        str(eval_output_root),
        "--output-csv",
        str(csv_path),
    ]
    if split == "test":
        arguments.append("--confirm-test-release")
    return uv_command(repo_root, [str(repo_root / EVALUATION_SCRIPT)], arguments)


def main(argv: list[str] | None = None) -> None:
    """Write the auditable evaluation command list.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv``.

    Raises:
        ValueError: If the test split is requested without an explicit release.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--training-output-root", type=Path, required=True)
    parser.add_argument("--eval-output-root", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--checkpoint-tags", nargs="+", choices=CHECKPOINT_TAGS, default=["latest"])
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--confirm-test-release", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.split == "test" and not args.confirm_test_release:
        raise ValueError("generating test-split commands requires explicit --confirm-test-release")
    repo_root = args.repo_root.resolve()
    cells = discover_cells(args.training_output_root, tuple(args.checkpoint_tags))
    if args.methods:
        selected = set(args.methods)
        cells = [cell for cell in cells if cell.provenance["method"] in selected]
    write_command_file(
        args.output,
        [
            command_for_cell(
                cell,
                repo_root=repo_root,
                protocol_path=args.protocol or repo_root / "research/multi_fidelity/experiment_protocol.yaml",
                dataset_root=args.dataset_root,
                eval_output_root=args.eval_output_root,
                metrics_root=args.metrics_root,
                split=args.split,
            )
            for cell in cells
        ],
    )


if __name__ == "__main__":
    main()
