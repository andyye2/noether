# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Write protocol manifests with both v2 indices and stats-tool aliases.

This is the primary manifest entry point for the study.  It delegates the
permutation and index/run-ID validation to ``materialize_nested_manifests`` and
adds explicit compatibility aliases used by the train-only statistics tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml

if __package__:
    from .materialize_nested_manifests import (
        DEFAULT_SEEDS,
        DEFAULT_SIZES,
        build_manifest,
        canonical_bytes,
        verify_manifest,
    )
else:
    from materialize_nested_manifests import (
        DEFAULT_SEEDS,
        DEFAULT_SIZES,
        build_manifest,
        canonical_bytes,
        verify_manifest,
    )


FROZEN_PROTOCOL_STATUS = "frozen_before_first_target_job"


def _git_state(repo_root: Path) -> tuple[str, bool]:
    """Return the implementation commit and whether tracked/untracked files differ."""
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status.strip())


def build_study_manifest(
    seed: int,
    sizes: tuple[int, ...],
    *,
    study_id: str,
    protocol_sha256: str,
    implementation_git_commit: str,
    implementation_git_dirty: bool,
) -> dict:
    """Build one hashed manifest accepted by training and statistics tools."""
    manifest = build_manifest(seed=seed, sizes=sizes)
    manifest.pop("manifest_sha256")
    manifest["study_id"] = study_id
    manifest["protocol_sha256"] = protocol_sha256
    manifest["implementation_git_commit"] = implementation_git_commit
    manifest["implementation_git_dirty"] = implementation_git_dirty
    manifest["val_run_ids"] = list(manifest["official_val_run_ids"])
    manifest["test_run_ids"] = list(manifest["official_test_run_ids"])
    manifest["nested_train_run_ids"] = {str(size): list(manifest["subsets"][str(size)]["run_ids"]) for size in sizes}
    manifest["manifest_sha256"] = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    verify_manifest(manifest)
    return manifest


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Durably publish one complete manifest."""
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as file:
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    """Materialize every preregistered acquisition ladder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiment_protocol.yaml",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    args = parser.parse_args()

    protocol_raw = args.protocol.read_bytes()
    protocol = yaml.safe_load(protocol_raw)
    if not isinstance(protocol, dict) or not isinstance(protocol.get("study_id"), str):
        raise ValueError("protocol must be a mapping with a string study_id")
    if protocol.get("status") != FROZEN_PROTOCOL_STATUS:
        raise ValueError(
            f"production manifests require protocol status {FROZEN_PROTOCOL_STATUS!r}; got {protocol.get('status')!r}"
        )
    protocol_sizes = protocol.get("data", {}).get("train_sample_sizes")
    if protocol_sizes != list(DEFAULT_SIZES):
        raise ValueError(f"protocol train_sample_sizes must be {list(DEFAULT_SIZES)}")
    replicate_entries = protocol.get("data", {}).get("paired_replicates")
    if not isinstance(replicate_entries, list) or len(replicate_entries) != len(DEFAULT_SEEDS):
        raise ValueError(f"protocol must contain exactly {len(DEFAULT_SEEDS)} paired replicates")
    protocol_seeds = tuple(entry.get("subset_seed") for entry in replicate_entries if isinstance(entry, dict))
    if protocol_seeds != DEFAULT_SEEDS:
        raise ValueError(f"protocol subset seeds must be {list(DEFAULT_SEEDS)}")
    if tuple(args.seeds) != protocol_seeds:
        raise ValueError("--seeds must exactly match the frozen protocol order")
    if tuple(args.sizes) != tuple(protocol_sizes):
        raise ValueError("--sizes must exactly match the frozen protocol order")
    protocol_sha256 = hashlib.sha256(protocol_raw).hexdigest()
    repo_root = Path(__file__).resolve().parents[3]
    implementation_git_commit, implementation_git_dirty = _git_state(repo_root)
    if implementation_git_dirty:
        raise ValueError("production manifests require a clean implementation Git worktree")
    output_dir = args.output_dir.resolve()
    resolved_repo_root = repo_root.resolve()
    if output_dir == resolved_repo_root or resolved_repo_root in output_dir.parents:
        raise ValueError("production manifests must be written outside the implementation Git worktree")
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = tuple(args.sizes)
    for seed in args.seeds:
        manifest = build_study_manifest(
            seed=seed,
            sizes=sizes,
            study_id=protocol["study_id"],
            protocol_sha256=protocol_sha256,
            implementation_git_commit=implementation_git_commit,
            implementation_git_dirty=implementation_git_dirty,
        )
        output = output_dir / f"drivaerml_nested_seed{seed}.json"
        _atomic_write_json(output, manifest)
        print(f"{output} {manifest['manifest_sha256']}")


if __name__ == "__main__":
    main()
