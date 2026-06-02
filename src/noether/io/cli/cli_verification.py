#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from pathlib import Path  # noqa: TCH003
from typing import Literal

import typer
from rich.traceback import install

install(show_locals=False)

from noether.io.cli.cli_utils import resolve_dir, run_cli
from noether.io.interfaces.huggingface import fetch_huggingface_file
from noether.io.interfaces.s3 import fetch_s3_file
from noether.io.verification import (
    FailAction,
    build_manifest,
    load_manifest,
    save_manifest,
    verify_tree,
)

RICH_MARKUP_MODE: Literal["markdown", "rich"] = "rich"

# --- General CLI declaration:
CTX = {"help_option_names": ["-h", "--help"], "max_content_width": 100}
verification_app = typer.Typer(
    name="verification",
    help="Data verification commands",
    no_args_is_help=True,
    rich_markup_mode=RICH_MARKUP_MODE,
    context_settings=CTX,
)


# =====================================================================================================================
#                                                 DATA VERIFICATION
# ---------------------------------------------------------------------------------------------------------------------
@verification_app.command("build", short_help="Create a manifest for a directory.")
def verify_build(
    root: Path = typer.Option(..., "--root", "-r", help="Directory to scan", dir_okay=True, file_okay=False),
    manifest: Path = typer.Option(..., "--manifest", "-m", help="JSON manifest to save"),
    jobs: int = typer.Option(8, "--jobs", "-j", help="Number of jobs to run", show_default=True),
    include_hash: bool = typer.Option(True, "--hash/--no-hash", help="Include SHA-256 in manifest"),
) -> None:
    root = root.expanduser().resolve()
    manifest_path = manifest.expanduser().resolve()

    def _task() -> None:
        _manifest = build_manifest(root=root, jobs=jobs, include_hash=include_hash)
        save_manifest(_manifest, manifest_path)
        typer.echo(f"Manifest saved to {manifest_path}")

    run_cli(_task)


@verification_app.command("check", short_help="Verify a directory against a manifest.")
def verify_check(
    root: Path = typer.Option(..., "--root", "-r", help="Directory to check", dir_okay=True, file_okay=False),
    manifest: Path = typer.Option(..., "--manifest", "-m", help="Path to manifest.json"),
    jobs: int = typer.Option(8, "--jobs", "-j", help="Hash workers", show_default=True),
    require_hash: bool = typer.Option(True, "--require-hash/--no-require-hash", show_default=True),
    on_fail: FailAction = typer.Option(FailAction.WARN, "--action", "-a", case_sensitive=False),
) -> None:
    root = resolve_dir(root)  # ensures exists; fine for check
    manifest_path = manifest.expanduser().resolve()

    def _task() -> None:
        _manifest = load_manifest(manifest_path)
        res = verify_tree(root, _manifest, jobs=jobs, require_hash=require_hash)
        typer.echo(
            f"OK={len(res.ok)} missing={len(res.missing)} extra={len(res.extra)} "
            f"size_mismatch={len(res.size_mismatch)} hash_mismatch={len(res.hash_mismatch)}"
        )

        # --- Handle user specified actions:
        if on_fail is FailAction.REDOWNLOAD:
            targets = sorted({*res.missing, *res.size_mismatch, *res.hash_mismatch})
            if not targets:
                return

            redownloaded = 0
            for relative_path in targets:
                record = _manifest.get(relative_path)
                src = record.source if record else None
                if not src:
                    typer.echo(f"[skip] No provenance for {relative_path}", err=True)
                    continue

                provider = src.get("provider")
                try:
                    if provider == "huggingface":
                        fetch_huggingface_file(
                            repo_id=src["repo_id"],
                            filename=src["filename"],
                            local_dir=root,
                            repo_type=src.get("repo_type", "model"),  # type: ignore
                            revision=src.get("revision", "main"),
                        )
                    elif provider == "s3":
                        fetch_s3_file(
                            bucket=src["bucket"],
                            key=src["key"],
                            local_dir=root,
                        )
                    else:
                        typer.echo(f"[skip] Unsupported provider {provider!r} for {relative_path}", err=True)
                        continue
                    redownloaded += 1
                except Exception as exc:
                    typer.echo(f"[fail] {relative_path}: {exc}", err=True)

            typer.echo(f"[redownload] attempted={len(targets)} ok={redownloaded}")

            # Re-verify after redownload:
            res2 = verify_tree(root, _manifest, jobs=jobs, require_hash=require_hash)
            typer.echo(
                f"[recheck] OK={len(res2.ok)} missing={len(res2.missing)} extra={len(res2.extra)} "
                f"size_mismatch={len(res2.size_mismatch)} hash_mismatch={len(res2.hash_mismatch)}"
            )

        if on_fail is FailAction.ABORT and (res.missing or res.extra or res.size_mismatch or res.hash_mismatch):
            raise typer.Exit(code=1)

    run_cli(_task)


# =====================================================================================================================
