#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import Literal

import typer
from rich.traceback import install

from noether.io.cli.cli_aws import aws_s3_app
from noether.io.cli.cli_checkpoint import checkpoint_app
from noether.io.cli.cli_huggingface import hf_app
from noether.io.cli.cli_verification import verification_app
from noether.io.logging_config import configure_logging

RICH_MARKUP_MODE: Literal["markdown", "rich"] = "rich"
install(show_locals=False)

CTX = {"help_option_names": ["-h", "--help"], "max_content_width": 100}

app = typer.Typer(
    name="Emmi Data Management CLI",
    help="Data fetching utilities (HuggingFace, S3, ...).",
    no_args_is_help=True,
    rich_markup_mode=RICH_MARKUP_MODE,
    context_settings=CTX,
    pretty_exceptions_enable=False,
    add_completion=False,
)


@app.callback()
def main(debug: bool = typer.Option(False, help="Enable debug logging")) -> None:
    configure_logging(debug)


# Merge subcommands
app.add_typer(hf_app, no_args_is_help=True)
app.add_typer(aws_s3_app, no_args_is_help=True)
app.add_typer(checkpoint_app, no_args_is_help=True)
app.add_typer(verification_app, no_args_is_help=True)

if __name__ == "__main__":
    app()
