# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Generate frozen validation/test commands exclusively from training provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Sequence
import yaml

PROVENANCE_FILENAME = "training_provenance.json"
PROVENANCE_KIND = "drivaerml_transfer_training_provenance"
CHECKPOINT_MODEL_NAME = "ab_upt"
CHECKPOINT_IDENTITY_BY_TAG = {
    "latest": "latest",
    "best_model.loss.val.total": "best-model-loss-val-total",
}
EVALUATION_RUNNER_REPO_PATH = Path("recipes/aero_cfd/scripts/eval_drivaerml_transfer_frozen.py")
METHOD_BY_STRATEGY = {
    "scratch": "S",
    "finetune": "P-FT",
    "linear_probe": "P-LP",
    "gradual_unfreeze": "P-GU",
}
#: Scope that leaves the method label bare, so every already-published label is
#: unchanged. This tool stays free of recipe imports; ``test_method_label_rule``
#: pins it to :func:`run_drivaerml_transfer_strict.method_label`.
DEFAULT_RESET_SCOPE = "readout"
VALID_TASKS = frozenset(("common", "full"))
VALID_SAMPLE_SIZES = frozenset((25, 50, 100, 200, 400))
VALID_FRAMES = frozenset(("native", "shapenet"))
VALID_BUDGETS = frozenset(("compute_matched", "fixed_epoch", "smoke"))
FROZEN_PROTOCOL_STATUS = "frozen_before_first_target_job"


def method_label(strategy: str, reset_scope: str | None = None) -> str | None:
    """Return the method label a training run with this strategy must carry.

    Args:
        strategy: Strategy recorded in the training sidecar.
        reset_scope: Reset scope recorded in the sidecar, or ``None`` for a run
            written before the option existed, which used the default scope.

    Returns:
        The expected label, or ``None`` for an unknown strategy.
    """
    base = METHOD_BY_STRATEGY.get(strategy)
    if base is None or strategy == "scratch" or reset_scope is None or reset_scope == DEFAULT_RESET_SCOPE:
        return base
    return f"{base}-rs{reset_scope}"


@dataclass(frozen=True)
class EvaluationCell:
    """One immutable evaluation cell recovered from a training sidecar."""

    sidecar_path: Path
    target_output_path: Path
    task: str
    strategy: str
    method: str
    replicate: int
    sample_size: int
    coordinate_frame: str
    budget: str
    manifest_path: Path
    statistics_path: Path
    run_id: str
    stage_name: str
    checkpoint_name: str
    checkpoint_tag: str
    checkpoint_identity: str
    checkpoint_sha256: str

    @property
    def key(self) -> tuple[str, str, int, int, str, str, str]:
        """Return the downstream metric identity, which must be unique."""
        return (
            self.task,
            self.method,
            self.replicate,
            self.sample_size,
            self.coordinate_frame,
            self.budget,
            self.checkpoint_identity,
        )


def _require_string(payload: dict, key: str, *, sidecar_path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{sidecar_path}: {key} must be a non-empty string")
    return value


def _require_absolute_path(payload: dict, key: str, *, sidecar_path: Path) -> Path:
    value = Path(_require_string(payload, key, sidecar_path=sidecar_path))
    if not value.is_absolute():
        raise ValueError(f"{sidecar_path}: {key} must be absolute, got {value}")
    return value


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_evaluation_cell(
    sidecar_path: Path,
    *,
    expected_protocol_sha256: str,
    checkpoint_tag: str = "latest",
) -> EvaluationCell:
    """Load one sidecar and bind its labels to one selected frozen checkpoint."""
    checkpoint_identity = CHECKPOINT_IDENTITY_BY_TAG.get(checkpoint_tag)
    if checkpoint_identity is None:
        raise ValueError(f"unsupported checkpoint tag {checkpoint_tag!r}")
    checkpoint_name = f"{CHECKPOINT_MODEL_NAME}_cp={checkpoint_tag}_model.th"
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{sidecar_path}: sidecar root must be a mapping")
    if payload.get("schema_version") != 1 or payload.get("kind") != PROVENANCE_KIND:
        raise ValueError(f"{sidecar_path}: unsupported training provenance schema")
    if payload.get("protocol_sha256") != expected_protocol_sha256:
        raise ValueError(
            f"{sidecar_path}: protocol SHA256 mismatch: "
            f"sidecar={payload.get('protocol_sha256')}, protocol={expected_protocol_sha256}"
        )

    task = _require_string(payload, "task", sidecar_path=sidecar_path)
    strategy = _require_string(payload, "strategy", sidecar_path=sidecar_path)
    method = _require_string(payload, "method", sidecar_path=sidecar_path)
    coordinate_frame = _require_string(payload, "coordinate_frame", sidecar_path=sidecar_path)
    budget = _require_string(payload, "budget", sidecar_path=sidecar_path)
    run_id = _require_string(payload, "run_id", sidecar_path=sidecar_path)
    stage_name = _require_string(payload, "stage_name", sidecar_path=sidecar_path)
    replicate = payload.get("replicate")
    sample_size = payload.get("train_sample_size")

    if task not in VALID_TASKS:
        raise ValueError(f"{sidecar_path}: unsupported task {task!r}")
    if method_label(strategy, payload.get("reset_scope")) != method:
        raise ValueError(
            f"{sidecar_path}: strategy {strategy!r} with reset scope "
            f"{payload.get('reset_scope')!r} does not imply method {method!r}"
        )
    if not isinstance(replicate, int) or isinstance(replicate, bool) or replicate not in range(8):
        raise ValueError(f"{sidecar_path}: replicate must be an integer in [0, 7]")
    if not isinstance(sample_size, int) or isinstance(sample_size, bool) or sample_size not in VALID_SAMPLE_SIZES:
        raise ValueError(f"{sidecar_path}: invalid train_sample_size {sample_size!r}")
    if coordinate_frame not in VALID_FRAMES:
        raise ValueError(f"{sidecar_path}: unsupported coordinate_frame {coordinate_frame!r}")
    if budget not in VALID_BUDGETS:
        raise ValueError(f"{sidecar_path}: unsupported budget {budget!r}")

    manifest_path = _require_absolute_path(payload, "manifest_path", sidecar_path=sidecar_path)
    statistics_path = _require_absolute_path(payload, "target_statistics_path", sidecar_path=sidecar_path)
    checkpoint_hashes = payload.get("model_checkpoints_sha256")
    if not isinstance(checkpoint_hashes, dict):
        raise ValueError(f"{sidecar_path}: model_checkpoints_sha256 must be a mapping")
    checkpoint_sha256 = checkpoint_hashes.get(checkpoint_name)
    if not _is_sha256(checkpoint_sha256):
        raise ValueError(f"{sidecar_path}: missing valid selected checkpoint SHA256 for {checkpoint_name}")

    if sidecar_path.parent.name != stage_name or sidecar_path.parent.parent.name != run_id:
        raise ValueError(f"{sidecar_path}: location does not match recorded run/stage {run_id!r}/{stage_name!r}")
    target_output_path = sidecar_path.parents[2]
    checkpoint_path = sidecar_path.parent / "checkpoints" / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{sidecar_path}: recorded selected checkpoint is missing: {checkpoint_path}")
    actual_checkpoint_sha256 = _sha256_file(checkpoint_path)
    if actual_checkpoint_sha256 != checkpoint_sha256:
        raise ValueError(
            f"{sidecar_path}: selected checkpoint SHA256 mismatch: "
            f"sidecar={checkpoint_sha256}, actual={actual_checkpoint_sha256}"
        )

    return EvaluationCell(
        sidecar_path=sidecar_path,
        target_output_path=target_output_path,
        task=task,
        strategy=strategy,
        method=method,
        replicate=replicate,
        sample_size=sample_size,
        coordinate_frame=coordinate_frame,
        budget=budget,
        manifest_path=manifest_path,
        statistics_path=statistics_path,
        run_id=run_id,
        stage_name=stage_name,
        checkpoint_name=checkpoint_name,
        checkpoint_tag=checkpoint_tag,
        checkpoint_identity=checkpoint_identity,
        checkpoint_sha256=checkpoint_sha256,
    )


def discover_evaluation_cells(
    training_output_root: Path,
    protocol_path: Path,
    *,
    checkpoint_tag: str = "latest",
    methods: Iterable[str] | None = None,
    replicates: Iterable[int] | None = None,
    sample_sizes: Iterable[int] | None = None,
    tasks: Iterable[str] | None = None,
    budgets: Iterable[str] | None = None,
    frames: Iterable[str] | None = None,
) -> list[EvaluationCell]:
    """Recursively discover, validate, filter, sort, and deduplicate cells."""
    protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    sidecar_paths = sorted(training_output_root.rglob(PROVENANCE_FILENAME))
    if not sidecar_paths:
        raise FileNotFoundError(f"no {PROVENANCE_FILENAME} files below {training_output_root}")
    cells = [
        load_evaluation_cell(
            sidecar_path,
            expected_protocol_sha256=protocol_sha256,
            checkpoint_tag=checkpoint_tag,
        )
        for sidecar_path in sidecar_paths
    ]

    method_filter = set(methods) if methods is not None else None
    replicate_filter = set(replicates) if replicates is not None else None
    size_filter = set(sample_sizes) if sample_sizes is not None else None
    task_filter = set(tasks) if tasks is not None else None
    budget_filter = set(budgets) if budgets is not None else None
    frame_filter = set(frames) if frames is not None else None
    cells = [
        cell
        for cell in cells
        if (method_filter is None or cell.method in method_filter)
        and (replicate_filter is None or cell.replicate in replicate_filter)
        and (size_filter is None or cell.sample_size in size_filter)
        and (task_filter is None or cell.task in task_filter)
        and (budget_filter is None or cell.budget in budget_filter)
        and (frame_filter is None or cell.coordinate_frame in frame_filter)
    ]
    if not cells:
        raise ValueError("evaluation filters selected no training provenance cells")

    unique: dict[tuple[str, str, int, int, str, str, str], EvaluationCell] = {}
    for cell in cells:
        previous = unique.get(cell.key)
        if previous is not None:
            raise ValueError(f"duplicate evaluation cell {cell.key}: {previous.sidecar_path} and {cell.sidecar_path}")
        unique[cell.key] = cell
    return sorted(
        unique.values(),
        key=lambda cell: (
            cell.task,
            cell.method,
            cell.replicate,
            cell.sample_size,
            cell.coordinate_frame,
            cell.budget,
            cell.checkpoint_identity,
            cell.run_id,
        ),
    )


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
    """Render one shell-safe command whose scientific labels all come from the sidecar."""
    if split not in {"val", "test"}:
        raise ValueError(f"unsupported evaluation split {split!r}")
    runner = repo_root / EVALUATION_RUNNER_REPO_PATH
    eval_run_id = (
        f"frozen-{split}-{cell.task}-{cell.method}-r{cell.replicate}-n{cell.sample_size}"
        f"-{cell.coordinate_frame}-{cell.budget}-{cell.checkpoint_identity}"
        f"-{cell.checkpoint_sha256[:12]}"
    )
    output_csv = (
        metrics_root
        / split
        / cell.task
        / cell.method
        / cell.budget
        / cell.checkpoint_identity
        / f"r{cell.replicate}_n{cell.sample_size}_{cell.coordinate_frame}.csv"
    )
    arguments = [
        "env",
        f"PYTHONPATH={repo_root}:{repo_root / 'src'}:{repo_root / 'recipes/aero_cfd/src'}",
        "uv",
        "run",
        "--project",
        str(repo_root),
        "--no-sync",
        "python",
        str(runner),
        "--dataset-root",
        str(dataset_root),
        "--protocol",
        str(protocol_path),
        "--manifest",
        str(cell.manifest_path),
        "--target-statistics",
        str(cell.statistics_path),
        "--task",
        cell.task,
        "--coordinate-frame",
        cell.coordinate_frame,
        "--sample-size",
        str(cell.sample_size),
        "--budget",
        cell.budget,
        "--method",
        cell.method,
        "--replicate",
        str(cell.replicate),
        "--evaluation-split",
        split,
        "--target-output-path",
        str(cell.target_output_path),
        "--target-run-id",
        cell.run_id,
        "--target-stage-name",
        cell.stage_name,
        "--target-model-name",
        "ab_upt",
        "--target-checkpoint-tag",
        cell.checkpoint_tag,
        "--expected-target-sha256",
        cell.checkpoint_sha256,
        "--eval-output-path",
        str(eval_output_root / split / cell.task / cell.method / cell.budget / cell.checkpoint_identity),
        "--eval-run-id",
        eval_run_id,
        "--output-csv",
        str(output_csv),
    ]
    if split == "test":
        arguments.append("--confirm-test-release")
    return shlex.join(arguments)


def generate_evaluation_commands(
    *,
    repo_root: Path,
    dataset_root: Path,
    protocol_path: Path,
    training_output_root: Path,
    eval_output_root: Path,
    metrics_root: Path,
    split: str,
    confirm_test_release: bool = False,
    checkpoint_tag: str = "latest",
    methods: Iterable[str] | None = None,
    replicates: Iterable[int] | None = None,
    sample_sizes: Iterable[int] | None = None,
    tasks: Iterable[str] | None = None,
    budgets: Iterable[str] | None = None,
    frames: Iterable[str] | None = None,
) -> list[str]:
    """Build a deterministic command list after enforcing the coordinated test gate."""
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or protocol.get("status") != FROZEN_PROTOCOL_STATUS:
        status = protocol.get("status") if isinstance(protocol, dict) else None
        raise ValueError(
            f"evaluation command generation requires protocol status {FROZEN_PROTOCOL_STATUS!r}; got {status!r}"
        )
    if split == "test" and not confirm_test_release:
        raise ValueError("test evaluation requires explicit --confirm-test-release")
    roots = {
        "repo_root": repo_root,
        "dataset_root": dataset_root,
        "protocol_path": protocol_path,
        "training_output_root": training_output_root,
        "eval_output_root": eval_output_root,
        "metrics_root": metrics_root,
    }
    for name, path in roots.items():
        if not path.is_absolute():
            raise ValueError(f"{name} must be absolute, got {path}")
    cells = discover_evaluation_cells(
        training_output_root,
        protocol_path,
        checkpoint_tag=checkpoint_tag,
        methods=methods,
        replicates=replicates,
        sample_sizes=sample_sizes,
        tasks=tasks,
        budgets=budgets,
        frames=frames,
    )
    return [
        command_for_cell(
            cell,
            repo_root=repo_root,
            protocol_path=protocol_path,
            dataset_root=dataset_root,
            eval_output_root=eval_output_root,
            metrics_root=metrics_root,
            split=split,
        )
        for cell in cells
    ]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--training-output-root", type=Path, required=True)
    parser.add_argument("--eval-output-root", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument(
        "--checkpoint-tag",
        choices=tuple(CHECKPOINT_IDENTITY_BY_TAG),
        default="latest",
    )
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--reps", nargs="+", type=int)
    parser.add_argument("--ns", nargs="+", type=int)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--budgets", nargs="+", choices=("smoke", "compute_matched", "fixed_epoch"))
    parser.add_argument("--frames", nargs="+", choices=("native", "shapenet"))
    parser.add_argument("--confirm-test-release", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Write one auditable frozen-evaluation command list."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.split == "test" and not args.confirm_test_release:
        parser.error("--split test requires explicit --confirm-test-release")
    commands = generate_evaluation_commands(
        repo_root=args.repo_root,
        dataset_root=args.dataset_root,
        protocol_path=args.protocol,
        training_output_root=args.training_output_root,
        eval_output_root=args.eval_output_root,
        metrics_root=args.metrics_root,
        split=args.split,
        confirm_test_release=args.confirm_test_release,
        checkpoint_tag=args.checkpoint_tag,
        methods=args.methods,
        replicates=args.reps,
        sample_sizes=args.ns,
        tasks=args.tasks,
        budgets=args.budgets,
        frames=args.frames,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"split={args.split} jobs={len(commands)} output={args.output}")


if __name__ == "__main__":
    main()
