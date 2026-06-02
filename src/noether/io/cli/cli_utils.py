#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable  # noqa: TCH003
from pathlib import Path

import typer

from noether.io.verification import ParallelErrors

_TOKEN_PAT = re.compile(r"(hf_[A-Za-z0-9]{6})[A-Za-z0-9]+")


def resolve_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def fmt_bytes(num: int) -> str:
    """Converts a number of bytes to a human-readable string.

    Args:
        num: Input number of bytes.

    Returns:
        - Human readable string with trailing storage units.
    """
    value: int | float = num
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return "0 B"


def sanitize(msg: str) -> str:
    if not msg:
        return msg
    # redact typical HF token pattern and exact env value if set
    msg = _TOKEN_PAT.sub(r"\1***REDACTED***", msg)
    tok = os.getenv("HF_TOKEN")
    if tok:
        msg = msg.replace(tok, "hf_***REDACTED***")
    return msg


def run_cli(fn: Callable[[], None]) -> None:
    try:
        fn()
    except ParallelErrors as exc:
        # Print each failing input & its error, then exit(1)
        for item, err in exc.errors:
            typer.echo(f"Task failed for input: {item!r} → {err}", err=True)
        raise typer.Exit(code=1) from None
    except Exception as exc:
        typer.secho(f"Error: {sanitize(str(exc))}", fg=typer.colors.RED, err=True)
        sys.exit(1)
