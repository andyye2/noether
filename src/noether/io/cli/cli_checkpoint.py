#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Literal

import typer
from rich.traceback import install

from noether.io.checkpoint.resolver import resolve_checkpoint
from noether.io.cli.cli_utils import run_cli

RICH_MARKUP_MODE: Literal["markdown", "rich"] = "rich"
install(show_locals=False)

CTX = {"help_option_names": ["-h", "--help"], "max_content_width": 100}

checkpoint_app = typer.Typer(
    name="checkpoint",
    help="Checkpoint utilities (fetch/verify).",
    no_args_is_help=True,
    rich_markup_mode=RICH_MARKUP_MODE,
    context_settings=CTX,
)


def to_jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set):
        return [to_jsonable(x) for x in obj]
    return obj


@checkpoint_app.command("fetch", short_help="Fetch a checkpoint URI to local cache and verify.")
def checkpoint_fetch(
    checkpoint: str = typer.Argument(..., help="URI or local path (hf://, s3://, file:// or plain path)"),
    cache_dir: Path = typer.Option(Path.home() / ".cache" / "emmi" / "checkpoints", "--cache-dir"),
    expected_sha256: str | None = typer.Option(None, "--sha256", help="Expected SHA-256 (hex)"),
    verify_load: str | None = typer.Option(None, "--verify-load", help="pt|ts|onnx"),
    min_free_mb: int | None = typer.Option(None, "--min-free-mb"),
):
    def _task() -> None:
        meta = resolve_checkpoint(
            checkpoint,
            cache_dir=cache_dir,
            expected_sha256=expected_sha256,
            verify_load=verify_load,
            min_free_bytes=(min_free_mb * 1024 * 1024) if min_free_mb else None,
        )
        payload = {
            # ensure the *URI* is a string (don’t pass a Path here)
            "uri": str(getattr(meta, "source_uri", checkpoint)),
            "saved_to": str(meta.local_path),
            "size": meta.local_path.stat().st_size if meta.local_path.exists() else None,
            "sha256": getattr(meta, "sha256", None),
            # if you also expose provider info, make sure it’s jsonable:
            "provider": getattr(meta, "provider", None),
            "extra": getattr(meta, "extra", None),  # may contain Paths → handled by to_jsonable
        }
        typer.echo(json.dumps(to_jsonable(payload), ensure_ascii=False))

    run_cli(_task)


@checkpoint_app.command("verify", short_help="Validate checksum and/or load a local checkpoint.")
def checkpoint_verify(
    path: Path = typer.Argument(..., exists=True),
    expected_sha256: str | None = typer.Option(None, "--sha256"),
    verify_load: str | None = typer.Option(None, "--verify-load", help="pt|ts|onnx"),
):
    from noether.io.verification import hash_file

    def _task() -> None:
        if expected_sha256:
            sha = hash_file(path)
            if sha != expected_sha256:
                raise typer.Exit(code=1)
        if verify_load:
            from noether.io.checkpoint.resolver import _smoke_load

            _smoke_load(path, verify_load)
        typer.echo("OK")

    run_cli(_task)
