#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import os
import re
from enum import Enum
from pathlib import Path  # noqa: TCH003
from typing import Literal

import typer
from rich.traceback import install

install(show_locals=False)

from noether.io.cli.cli_utils import fmt_bytes, run_cli
from noether.io.interfaces.huggingface import (
    estimate_hf_repo_size,
    fetch_huggingface_by_extension,
    fetch_huggingface_file,
    fetch_huggingface_repo_snapshot,
)
from noether.io.verification import (
    FailAction,
    FileRecord,
    hash_file,
    load_manifest,
    save_manifest,
    verify_tree,
)

RICH_MARKUP_MODE: Literal["markdown", "rich"] = "rich"

# --- General CLI declaration:
CTX = {"help_option_names": ["-h", "--help"], "max_content_width": 100}
hf_app = typer.Typer(
    name="huggingface",
    help="HuggingFace commands",
    no_args_is_help=True,
    rich_markup_mode=RICH_MARKUP_MODE,
    context_settings=CTX,
)

_TOKEN_PAT = re.compile(r"(hf_[A-Za-z0-9]{6})[A-Za-z0-9]+")


def _sanitize(msg: str) -> str:
    if not msg:
        return msg
    # redact typical HF token pattern and exact env value if set
    msg = _TOKEN_PAT.sub(r"\1***REDACTED***", msg)
    tok = os.getenv("HF_TOKEN")
    if tok:
        msg = msg.replace(tok, "hf_***REDACTED***")
    return msg


class HuggingFaceRepoType(str, Enum):
    MODEL = "model"
    DATASET = "dataset"


def _resolve_dir(p: Path) -> Path:
    p = p.expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


# =====================================================================================================================
#                                                   HUGGINGFACE
# ---------------------------------------------------------------------------------------------------------------------
@hf_app.callback()
def hf_docs():
    """
    **HuggingFace commands**

    **Examples**
    ```
    # snapshot whole repo
    emmi-data huggingface snapshot user/dataset ./data

    # single file from a dataset repo
    emmi-data huggingface file user/dataset data.hd5 ./data --type dataset

    # all *.jsonl from a dataset
    emmi-data huggingface ext user/model .th ./data

    # size estimate (uses HEAD fallback for LFS files)
    emmi-data huggingface estimate EmmiAI/AB-UPT
    ```
    """


@hf_app.command("snapshot", short_help="Create a local snapshot of a repo.")
def hf_snapshot(
    repo_id: str = typer.Argument(..., help="Repo ID, e.g. user/dataset"),
    local_dir: Path = typer.Argument(..., help="Destination directory", dir_okay=True, file_okay=False),
    verify: bool = typer.Option(False, "--verify", help="Run checksum verification"),
) -> None:
    local_dir = _resolve_dir(local_dir)
    fetch_huggingface_repo_snapshot(repo_id=repo_id, local_dir=local_dir)
    if verify:
        typer.echo("Verification not implemented yet.")


@hf_app.command("file", short_help="Download a single file from a repo.")
def hf_file(
    repo_id: str = typer.Argument(..., help="Repo ID"),
    filename: str = typer.Argument(..., help="Exact path in repo"),
    local_dir: Path = typer.Argument(..., help="Destination directory", dir_okay=True, file_okay=False),
    repo_type: HuggingFaceRepoType = typer.Option(
        HuggingFaceRepoType.MODEL, "--type", "-t", case_sensitive=False, help="model|dataset", show_default=True
    ),
    revision: str = typer.Option("main", "--revision", "-r", help="branch|tag|SHA", show_default=True),
) -> None:
    local_dir = _resolve_dir(local_dir)
    fetch_huggingface_file(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        repo_type=repo_type.value,  # type: ignore
        revision=revision,
    )


@hf_app.command("ext", short_help="Download all files with an extension from the repo.")
def hf_ext(
    repo_id: str = typer.Argument(..., help="Dataset repo ID"),
    extension: str = typer.Argument(..., help="e.g. .jsonl, .csv, .parquet"),
    local_dir: Path = typer.Argument(..., help="Destination directory", dir_okay=True, file_okay=False),
    revision: str = typer.Option("main", "--revision", "-r", help="branch|tag|SHA", show_default=True),
    repo_type: HuggingFaceRepoType = typer.Option(HuggingFaceRepoType.DATASET, "--type", "-t", case_sensitive=False),
    jobs: int = typer.Option(8, "--jobs", "-j", help="Parallel downloads"),
    verify: bool = typer.Option(False, "--verify", help="Verify files after download using a manifest"),
    manifest: Path | None = typer.Option(None, "--manifest", "-m", help="Path to manifest.json"),
    manifest_out: Path | None = typer.Option(None, "--manifest-out", "-mo", help="Path to provenance manifest"),
    on_fail: FailAction = typer.Option(
        FailAction.WARN, "--action", "-a", case_sensitive=False, help="Action on verification failures"
    ),
) -> None:
    local_dir = _resolve_dir(local_dir)

    def _task() -> None:
        files = fetch_huggingface_by_extension(
            repo_id=repo_id,
            extension=extension,
            local_dir=local_dir,
            revision=revision,
            repo_type=repo_type.value,  # type: ignore
            max_workers=jobs,
        )

        if manifest_out is not None:
            _manifest: dict[str, FileRecord] = {}
            for relative_path in files:
                _path = local_dir / relative_path
                _manifest[relative_path] = FileRecord(
                    path=relative_path,
                    size=_path.stat().st_size,
                    hash=hash_file(_path),  # or compute if you want strong integrity here
                    etag=None,
                    source={
                        "provider": "huggingface",
                        "repo_id": repo_id,
                        "repo_type": repo_type.value,
                        "revision": revision,
                        "filename": relative_path,
                    },
                )
            save_manifest(_manifest, manifest_out.expanduser().resolve())
            typer.echo(f"[provenance] wrote {manifest_out}")

        if verify:
            if not manifest:
                typer.echo("`--verify` requires `--manifest`.", err=True)
                raise typer.Exit(code=2)
            man = load_manifest(manifest.expanduser().resolve())
            res = verify_tree(local_dir, man, jobs=jobs, require_hash=True)
            typer.echo(
                f"[verify] OK={len(res.ok)} missing={len(res.missing)} extra={len(res.extra)} "
                f"size_mismatch={len(res.size_mismatch)} hash_mismatch={len(res.hash_mismatch)}"
            )
            if on_fail is FailAction.ABORT and (res.missing or res.extra or res.size_mismatch or res.hash_mismatch):
                raise typer.Exit(code=1)

    run_cli(_task)


@hf_app.command("estimate", short_help="Estimate total repo size.")
def hf_estimate(
    repo_id: str = typer.Argument(..., help="Repo ID"),
    repo_type: HuggingFaceRepoType = typer.Option(
        HuggingFaceRepoType.MODEL, "--type", "-t", case_sensitive=False, help="model|dataset", show_default=True
    ),
    extension: str | None = typer.Option(None, "--extension", "-e", help="Filter by extension"),
    revision: str = typer.Option("main", "--revision", "-r", help="branch|tag|SHA", show_default=True),
) -> None:
    size = estimate_hf_repo_size(
        repo_id=repo_id,
        repo_type=repo_type.value,  # type: ignore
        revision=revision,
        extension=extension,
    )
    typer.echo(f"{repo_id}: ~{fmt_bytes(size)}")


# =====================================================================================================================
