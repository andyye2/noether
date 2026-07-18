# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Tests for provenance-derived frozen-evaluation command generation."""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path

import pytest

from .generate_evaluation_commands import generate_evaluation_commands


def _argument(tokens: list[str], flag: str) -> str:
    return tokens[tokens.index(flag) + 1]


def _build_tree(
    tmp_path: Path,
    *,
    task: str = "full",
    strategy: str = "finetune",
    method: str = "P-FT",
    replicate: int = 3,
    sample_size: int = 100,
    coordinate_frame: str = "native",
    run_id: str = "training-run",
    budget: str = "compute_matched",
) -> dict[str, Path | str]:
    repo_root = tmp_path / "remote-repo"
    dataset_root = tmp_path / "dataset"
    protocol_path = tmp_path / "input-protocol.yaml"
    training_root = tmp_path / "training"
    eval_root = tmp_path / "evaluation"
    metrics_root = tmp_path / "metrics"
    manifest = tmp_path / "manifests" / "nested.json"
    statistics = tmp_path / "statistics" / "stats.json"
    for directory in (
        repo_root,
        dataset_root,
        training_root,
        eval_root,
        metrics_root,
        manifest.parent,
        statistics.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(
        "study_id: generator_unit_test\nstatus: frozen_before_first_target_job\n", encoding="utf-8"
    )
    protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    manifest.write_text("{}\n", encoding="utf-8")
    statistics.write_text("{}\n", encoding="utf-8")

    stage = training_root / "phase" / run_id / "train"
    checkpoint_dir = stage / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "ab_upt_cp=latest_model.th"
    checkpoint.write_bytes(b"latest checkpoint")
    best_checkpoint = checkpoint_dir / "ab_upt_cp=best_model.loss.val.total_model.th"
    best_checkpoint.write_bytes(b"best validation checkpoint")
    recorded_checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    best_checkpoint_sha256 = hashlib.sha256(best_checkpoint.read_bytes()).hexdigest()
    sidecar = {
        "schema_version": 1,
        "kind": "drivaerml_transfer_training_provenance",
        "protocol_sha256": protocol_sha256,
        "manifest_path": str(manifest),
        "target_statistics_path": str(statistics),
        "task": task,
        "strategy": strategy,
        "method": method,
        "replicate": replicate,
        "train_sample_size": sample_size,
        "coordinate_frame": coordinate_frame,
        "budget": budget,
        "run_id": run_id,
        "stage_name": "train",
        "model_checkpoints_sha256": {
            "ab_upt_cp=latest_model.th": recorded_checkpoint_sha256,
            "ab_upt_cp=best_model.loss.val.total_model.th": best_checkpoint_sha256,
        },
    }
    sidecar_path = stage / "training_provenance.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return {
        "repo_root": repo_root,
        "dataset_root": dataset_root,
        "protocol_path": protocol_path,
        "training_root": training_root,
        "eval_root": eval_root,
        "metrics_root": metrics_root,
        "manifest": manifest,
        "statistics": statistics,
        "recorded_checkpoint_sha256": recorded_checkpoint_sha256,
        "best_checkpoint_sha256": best_checkpoint_sha256,
    }


def test_validation_command_succeeds_and_inherits_every_scientific_label(tmp_path: Path) -> None:
    tree = _build_tree(tmp_path)

    commands = generate_evaluation_commands(
        repo_root=tree["repo_root"],
        dataset_root=tree["dataset_root"],
        protocol_path=tree["protocol_path"],
        training_output_root=tree["training_root"],
        eval_output_root=tree["eval_root"],
        metrics_root=tree["metrics_root"],
        split="val",
    )

    assert len(commands) == 1
    tokens = shlex.split(commands[0])
    assert tokens[:3] == [
        "env",
        f"PYTHONPATH={tree['repo_root']}:{tree['repo_root'] / 'src'}:{tree['repo_root'] / 'recipes/aero_cfd/src'}",
        "uv",
    ]
    assert tokens[tokens.index("uv") : tokens.index("python")] == [
        "uv",
        "run",
        "--project",
        str(tree["repo_root"]),
        "--no-sync",
    ]
    assert _argument(tokens, "--protocol") == str(tree["protocol_path"])
    assert _argument(tokens, "--task") == "full"
    assert _argument(tokens, "--method") == "P-FT"
    assert _argument(tokens, "--replicate") == "3"
    assert _argument(tokens, "--sample-size") == "100"
    assert _argument(tokens, "--budget") == "compute_matched"
    assert _argument(tokens, "--coordinate-frame") == "native"
    assert _argument(tokens, "--manifest") == str(tree["manifest"])
    assert _argument(tokens, "--target-statistics") == str(tree["statistics"])
    assert _argument(tokens, "--target-run-id") == "training-run"
    assert _argument(tokens, "--target-output-path") == str(tree["training_root"] / "phase")
    assert _argument(tokens, "--expected-target-sha256") == tree["recorded_checkpoint_sha256"]
    assert _argument(tokens, "--evaluation-split") == "val"
    assert "compute_matched" in _argument(tokens, "--eval-run-id")
    assert _argument(tokens, "--target-checkpoint-tag") == "latest"
    assert Path(_argument(tokens, "--output-csv")).parent.name == "latest"
    assert Path(_argument(tokens, "--output-csv")).parent.parent.name == "compute_matched"
    assert Path(_argument(tokens, "--eval-output-path")).name == "latest"
    assert "--confirm-test-release" not in tokens


def test_test_split_requires_explicit_release_confirmation(tmp_path: Path) -> None:
    tree = _build_tree(tmp_path)
    kwargs = {
        "repo_root": tree["repo_root"],
        "dataset_root": tree["dataset_root"],
        "protocol_path": tree["protocol_path"],
        "training_output_root": tree["training_root"],
        "eval_output_root": tree["eval_root"],
        "metrics_root": tree["metrics_root"],
        "split": "test",
    }

    with pytest.raises(ValueError, match="--confirm-test-release"):
        generate_evaluation_commands(**kwargs)

    commands = generate_evaluation_commands(**kwargs, confirm_test_release=True)
    assert len(commands) == 1
    tokens = shlex.split(commands[0])
    assert _argument(tokens, "--budget") == "compute_matched"
    assert "--confirm-test-release" in tokens


def test_latest_and_best_checkpoints_generate_disjoint_outputs(tmp_path: Path) -> None:
    tree = _build_tree(tmp_path)
    common = {
        "repo_root": tree["repo_root"],
        "dataset_root": tree["dataset_root"],
        "protocol_path": tree["protocol_path"],
        "training_output_root": tree["training_root"],
        "eval_output_root": tree["eval_root"],
        "metrics_root": tree["metrics_root"],
        "split": "val",
    }
    latest_tokens = shlex.split(generate_evaluation_commands(**common, checkpoint_tag="latest")[0])
    best_tokens = shlex.split(generate_evaluation_commands(**common, checkpoint_tag="best_model.loss.val.total")[0])

    assert _argument(latest_tokens, "--target-checkpoint-tag") == "latest"
    assert _argument(best_tokens, "--target-checkpoint-tag") == "best_model.loss.val.total"
    assert _argument(latest_tokens, "--expected-target-sha256") == tree["recorded_checkpoint_sha256"]
    assert _argument(best_tokens, "--expected-target-sha256") == tree["best_checkpoint_sha256"]
    for flag in ("--eval-run-id", "--eval-output-path", "--output-csv"):
        assert _argument(latest_tokens, flag) != _argument(best_tokens, flag)
    assert "best-model-loss-val-total" in _argument(best_tokens, "--eval-run-id")
    assert Path(_argument(best_tokens, "--eval-output-path")).name == "best-model-loss-val-total"


def test_budget_and_frame_filters_allow_parallel_budget_cells(tmp_path: Path) -> None:
    tree = _build_tree(tmp_path)
    _build_tree(tmp_path, run_id="fixed-epoch-run", budget="fixed_epoch")
    _build_tree(tmp_path, run_id="shapenet-run", coordinate_frame="shapenet")
    common = {
        "repo_root": tree["repo_root"],
        "dataset_root": tree["dataset_root"],
        "protocol_path": tree["protocol_path"],
        "training_output_root": tree["training_root"],
        "eval_output_root": tree["eval_root"],
        "metrics_root": tree["metrics_root"],
        "split": "val",
    }

    commands = generate_evaluation_commands(**common)
    assert len(commands) == 3
    assert any("compute_matched" in _argument(shlex.split(command), "--eval-run-id") for command in commands)
    assert any("fixed_epoch" in _argument(shlex.split(command), "--eval-run-id") for command in commands)

    selected = generate_evaluation_commands(
        **common,
        budgets=["compute_matched"],
        frames=["shapenet"],
    )
    assert len(selected) == 1
    tokens = shlex.split(selected[0])
    assert _argument(tokens, "--target-run-id") == "shapenet-run"
    assert _argument(tokens, "--coordinate-frame") == "shapenet"
    assert Path(_argument(tokens, "--output-csv")).parent.parent.name == "compute_matched"


def test_tampered_checkpoint_is_rejected_before_command_generation(tmp_path: Path) -> None:
    tree = _build_tree(tmp_path)
    checkpoint = next(tree["training_root"].rglob("ab_upt_cp=latest_model.th"))
    checkpoint.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="selected checkpoint SHA256 mismatch"):
        generate_evaluation_commands(
            repo_root=tree["repo_root"],
            dataset_root=tree["dataset_root"],
            protocol_path=tree["protocol_path"],
            training_output_root=tree["training_root"],
            eval_output_root=tree["eval_root"],
            metrics_root=tree["metrics_root"],
            split="val",
        )


def test_duplicate_cell_and_missing_checkpoint_are_rejected(tmp_path: Path) -> None:
    tree = _build_tree(tmp_path)
    first_sidecar = next(tree["training_root"].rglob("training_provenance.json"))
    duplicate_stage = tree["training_root"] / "other-phase" / "training-run-duplicate" / "train"
    duplicate_stage.mkdir(parents=True)
    payload = json.loads(first_sidecar.read_text(encoding="utf-8"))
    payload["run_id"] = "training-run-duplicate"
    (duplicate_stage / "training_provenance.json").write_text(json.dumps(payload), encoding="utf-8")
    checkpoint = duplicate_stage / "checkpoints" / "ab_upt_cp=latest_model.th"

    with pytest.raises(FileNotFoundError, match="selected checkpoint is missing"):
        generate_evaluation_commands(
            repo_root=tree["repo_root"],
            dataset_root=tree["dataset_root"],
            protocol_path=tree["protocol_path"],
            training_output_root=tree["training_root"],
            eval_output_root=tree["eval_root"],
            metrics_root=tree["metrics_root"],
            split="val",
        )

    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"latest checkpoint")
    with pytest.raises(ValueError, match="duplicate evaluation cell"):
        generate_evaluation_commands(
            repo_root=tree["repo_root"],
            dataset_root=tree["dataset_root"],
            protocol_path=tree["protocol_path"],
            training_output_root=tree["training_root"],
            eval_output_root=tree["eval_root"],
            metrics_root=tree["metrics_root"],
            split="val",
        )


def test_evaluation_generation_rejects_draft_protocol(tmp_path: Path) -> None:
    """No validation or test commands are emitted from a draft protocol."""
    tree = _build_tree(tmp_path)
    tree["protocol_path"].write_text("study_id: generator_unit_test\nstatus: preregistration_draft\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires protocol status"):
        generate_evaluation_commands(
            repo_root=tree["repo_root"],
            dataset_root=tree["dataset_root"],
            protocol_path=tree["protocol_path"],
            training_output_root=tree["training_root"],
            eval_output_root=tree["eval_root"],
            metrics_root=tree["metrics_root"],
            split="val",
        )
