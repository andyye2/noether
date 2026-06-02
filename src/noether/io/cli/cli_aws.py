#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import typer

from noether.io.cli.cli_utils import fmt_bytes, resolve_dir, run_cli
from noether.io.interfaces.s3 import (
    estimate_s3_size,
    fetch_s3_file,
    fetch_s3_prefix,  # noqa: F401
    get_s3_client,
    list_s3_objects,
)
from noether.io.verification import FileRecord, hash_file, save_manifest

RICH_MARKUP_MODE: Literal["markdown", "rich"] = "rich"


CLI_HELP_MARKDOWN = """
    AWS S3 commands

    **Examples**
    ```
    # estimate bytes under a prefix
    emmi-data aws estimate my-bucket data/prefix/ --extension .pt

    # download a single object
    emmi-data aws file my-bucket data/prefix/file.pt ./data

    # download a prefix (parallel) and write a provenance manifest with hashes
    emmi-data aws fetch my-bucket data/prefix/ ./data \
      --extension .pt \
      --jobs 16 \
      --manifest-out ./s3-manifest.json \
      --hash --hash-jobs 8 --hash-chunk-mb 8
    ```
    """
CTX = {"help_option_names": ["-h", "--help"], "max_content_width": 100}
aws_s3_app = typer.Typer(
    name="aws",
    help="AWS S3 commands",
    no_args_is_help=True,
    rich_markup_mode=RICH_MARKUP_MODE,
)


@aws_s3_app.callback(help=CLI_HELP_MARKDOWN)
def docs() -> None:
    """
    AWS S3 commands

    Examples
    --------
    .. code-block:: bash

       # estimate bytes under a prefix
       emmi-data aws estimate my-bucket data/prefix/ --extension .pt

       # download a single object
       emmi-data aws file my-bucket data/prefix/file.pt ./data

       # download a prefix (parallel) and write a provenance manifest with hashes
       emmi-data aws fetch my-bucket data/prefix/ ./data \
         --extension .pt \
         --jobs 16 \
         --manifest-out ./s3-manifest.json \
         --hash --hash-jobs 8 --hash-chunk-mb 8
    """


@aws_s3_app.command("estimate", short_help="Estimate total size under a bucket/prefix.")
def aws_estimate(
    bucket: str = typer.Argument(..., help="S3 bucket name"),
    prefix: str = typer.Argument(..., help="S3 key prefix"),
    extension: str | None = typer.Option(None, "--extension", "-e", help="Filter by extension"),
) -> None:
    total_bytes, count = estimate_s3_size(bucket, prefix, extension=extension)
    ext_filter = f" (ext={extension!r})" if extension else ""
    typer.echo(f"s3://{bucket}/{prefix}{ext_filter} → {count} files, ~{fmt_bytes(total_bytes)}")


@aws_s3_app.command("file", short_help="Download a single S3 object.")
def aws_file(
    bucket: str = typer.Argument(..., help="S3 bucket"),
    key: str = typer.Argument(..., help="S3 object key"),
    local_dir: Path = typer.Argument(..., dir_okay=True, file_okay=False, help="Destination directory"),
) -> None:
    local_dir = resolve_dir(local_dir)

    def _task() -> None:
        local_path = fetch_s3_file(bucket, key, local_dir)
        typer.echo(f"Wrote: {local_path}")

    run_cli(_task)


@aws_s3_app.command("fetch", short_help="Download all objects under a prefix.")
def aws_fetch(
    bucket: str = typer.Argument(..., help="S3 bucket"),
    prefix: str = typer.Argument(..., help="S3 key prefix"),
    local_dir: Path = typer.Argument(..., dir_okay=True, file_okay=False, help="Destination directory"),
    extension: str | None = typer.Option(None, "--extension", "-e", help="Filter by extension"),
    jobs: int = typer.Option(8, "--jobs", "-j", help="Parallel downloads"),
    manifest_out: Path | None = typer.Option(None, "--manifest-out", "-m", help="Write a provenance manifest JSON"),
    hash_after: bool = typer.Option(True, "--hash/--no-hash", help="Compute SHA-256 for manifest entries"),
    hash_jobs: int = typer.Option(8, "--hash-jobs", help="Parallel hash workers"),
    hash_chunk_mb: int = typer.Option(8, "--hash-chunk-mb", help="Hash chunk size (MiB)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only report counts/size; do not download"),
    strip_prefix: bool = typer.Option(
        False, "--strip-prefix/--no-strip-prefix", help="Save keys relative to the provided prefix"
    ),
    flatten: bool = typer.Option(False, "--flatten", help="Save by basename only; error on collisions"),
    strict_hash: bool = typer.Option(False, "--strict-hash/--no-strict-hash", help="Exit non-zero if any hash fails"),
) -> None:
    local_dir = resolve_dir(local_dir)

    def _task() -> None:
        # List first so mapping uses original keys (and we can write accurate provenance)
        objects = list_s3_objects(bucket, prefix, extension=extension)

        # Filter out S3 directory markers:
        objects = [o for o in objects if not _is_directory_marker(o["key"])]

        if not objects:
            typer.echo("0 objects matched; nothing to fetch.")
            # still respect manifest_out with an empty manifest
            if manifest_out is not None:
                save_manifest({}, manifest_out.expanduser().resolve() if manifest_out != Path("-") else manifest_out)
                if manifest_out != Path("-"):
                    typer.echo(f"[provenance] wrote empty manifest: {manifest_out}")
            return

        total_bytes = sum(o.get("size") or 0 for o in objects)
        if dry_run:
            typer.echo(f"[dry-run] s3://{bucket}/{prefix} → {len(objects)} files, ~{fmt_bytes(total_bytes)}")
            return

        s3_client = get_s3_client()

        # Build a per-invocation mapping: full S3 key -> relative output path:
        def map_rel_out(key: str) -> str:
            _relative_path = _strip_prefix(key, prefix) if strip_prefix else key
            if flatten:
                _relative_path = Path(_relative_path).name
            _validate_rel_path(_relative_path)
            return _relative_path

        # Collision check when flattening:
        if flatten:
            names = [Path(map_rel_out(o["key"])).name for o in objects]
            dups = {n for n in names if names.count(n) > 1}
            if dups:
                raise typer.BadParameter(
                    "Basename collisions under --flatten: "
                    + ", ".join(sorted(dups))
                    + ". Consider removing --flatten or using distinct prefixes."
                )

        def _download_to_final(s3_client, bucket: str, key: str, target: Path) -> Path:
            """
            Stream an S3 object directly into `target` using a temp .part file, then atomically move into place.
            """
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".part")
            try:
                with tmp.open("wb") as fh:
                    s3_client.download_fileobj(Bucket=bucket, Key=key, Fileobj=fh)
                tmp.replace(target)
                return target
            finally:
                if tmp.exists() and not target.exists():
                    try:
                        tmp.unlink()
                    except Exception:
                        pass

        def _download_and_place(o: dict) -> str:
            key: str = o["key"]
            _relative_path = map_rel_out(key)
            target = local_dir / _relative_path
            _download_to_final(s3_client, bucket, key, target)
            return _relative_path

        # Parallel downloads:
        relative_paths: list[str] = []
        failures: list[tuple[str, BaseException]] = []

        with ThreadPoolExecutor(max_workers=min(jobs, max(1, len(objects)))) as executor:
            future_to_key: dict[Future[str], str] = {
                executor.submit(_download_and_place, o): o["key"]  # type: ignore[arg-type]
                for o in objects
            }
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    rel = future.result()
                    relative_paths.append(rel)
                except BaseException as exc:
                    failures.append((key, exc))

        if failures:
            for key, exc in failures:  # type: ignore[misc]
                typer.echo(f"[download-fail] s3://{bucket}/{key} → {exc}", err=True)  # type: ignore[misc]
            raise typer.Exit(code=1)

        # [Optional]: Write provenance manifest (with or without hashes), always using ORIGINAL key in `source`.
        if manifest_out is not None:
            manifest_path = manifest_out
            manifest: dict[str, FileRecord] = {}

            # Build fast lookup from rel to original key:
            key_by_relative_path = {map_rel_out(o["key"]): o["key"] for o in objects}

            if not relative_paths:
                # should not happen here; but to keep behavior consistent:
                if manifest_path == Path("-"):
                    typer.echo("{}", nl=True)
                else:
                    save_manifest({}, manifest_path.expanduser().resolve())
                    typer.echo(f"[provenance] wrote empty manifest: {manifest_path}")
                return

            if hash_after:

                def _record(_relative_path: str) -> tuple[str, FileRecord | None, tuple[str, Exception] | None]:
                    _path = local_dir / _relative_path
                    try:
                        digest = hash_file(_path, chunk_size_mb=hash_chunk_mb)
                        return (
                            _relative_path,
                            FileRecord(
                                path=_relative_path,
                                size=_path.stat().st_size,
                                hash=digest,
                                etag=None,
                                source={
                                    "provider": "s3",
                                    "bucket": bucket,
                                    "key": key_by_relative_path[_relative_path],
                                },
                            ),
                            None,
                        )
                    except Exception as exc:
                        return _relative_path, None, (_relative_path, exc)

                hash_failures: list[tuple[str, Exception]] = []
                with ThreadPoolExecutor(max_workers=min(hash_jobs, len(relative_paths))) as executor:
                    futures: dict[Future[tuple[str, FileRecord | None, tuple[str, Exception] | None]], str] = {
                        executor.submit(_record, rel): rel for rel in relative_paths
                    }
                    for f in as_completed(futures):
                        relative_path, rec, err = f.result()
                        if rec is not None:
                            manifest[relative_path] = rec
                        if err is not None:
                            hash_failures.append(err)

                for relative_path, error in hash_failures:
                    typer.echo(f"[hash-fail] {relative_path} → {error}", err=True)
                if strict_hash and hash_failures:
                    raise typer.Exit(code=1)
            else:
                for relative_path in relative_paths:
                    path = local_dir / relative_path
                    manifest[relative_path] = FileRecord(
                        path=relative_path,
                        size=path.stat().st_size,
                        hash=None,
                        etag=None,
                        source={"provider": "s3", "bucket": bucket, "key": key_by_relative_path[rel]},
                    )

            # Write manifest to file or stdout
            if manifest_path == Path("-"):
                # serialize without Paths in the structure
                import json

                data = {
                    k: {
                        "path": v.path,
                        "size": v.size,
                        "hash": v.hash,
                        "etag": v.etag,
                        "source": v.source,
                    }
                    for k, v in manifest.items()
                }
                typer.echo(json.dumps(data, indent=2, sort_keys=True))
                return
            else:
                save_manifest(manifest, manifest_path.expanduser().resolve())
                typer.echo(f"[provenance] wrote {manifest_path}")

        typer.echo(f"Download complete → {local_dir}")

    run_cli(_task)


# =====================================================================================================================
#                                                   UTILITIES
# ---------------------------------------------------------------------------------------------------------------------
def _is_directory_marker(key: str) -> bool:
    # S3 "folders" can be stored as 0-byte objects with trailing slash
    return key.endswith("/")


def _strip_prefix(key: str, prefix: str) -> str:
    # Normalize to ensure prefix ends with '/'
    path = prefix if prefix.endswith("/") else f"{prefix}/"
    return key.removeprefix(path)


def _validate_rel_path(relative_path: str) -> None:
    # Avoid writing outside target tree; disallow absolute or '..'
    path = Path(relative_path)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise typer.BadParameter(f"Illegal relative path derived from S3 key: {relative_path!r}")


def _prune_empty_dirs(root: Path, candidates: Iterable[Path]) -> None:
    # Remove empty directories bottom-up under root, only where we actually created structure
    seen_dirs = set()
    for path in candidates:
        _dir = path.parent
        while _dir != root and root in _dir.parents:
            seen_dirs.add(_dir)
            _dir = _dir.parent
    for _dir in sorted(seen_dirs, key=lambda x: len(x.parts), reverse=True):
        try:
            if _dir.exists() and not any(_dir.iterdir()):
                _dir.rmdir()
        except Exception:
            pass


# =====================================================================================================================
