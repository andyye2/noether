#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import shutil
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Self
from urllib.parse import urlparse

from loguru import logger

from noether.io.interfaces.s3 import get_s3_client


class S3FileUploader:
    """
    Asynchronously uploads existing local files to an S3 path.

    This class uses a thread pool to upload multiple files in parallel.
    It's designed to run *after* a process (like AsyncWriter)
    has finished writing files to a local directory.

    Its API mirrors AsyncWriter, but `write()` takes a local file path.

    Attributes:
        local_root (Path): The root directory of the local files.
                           Used to calculate the relative path for S3 keys.
        bucket (str): The name of the S3 bucket.
        prefix (str): The key prefix (folder) within the S3 bucket.

    Raises:
        ValueError: If `s3_path` doesn't start with 's3://'.
        ValueError: If local path is not inside a local directory.
        ImportError: If `rich` is not installed and `show_progress` is not True.
    """

    def __init__(
        self,
        local_root: Path | str,
        s3_path: str,
        workers: int = 4,
        max_inflight: int = 256,
    ) -> None:
        """
        Args:
            local_root: The local root directory (e.g., './my_local_results').
            s3_path: The full S3 destination path (e.g., 's3://my-bucket/results/').
            workers: Thread-pool size for parallel uploads.
            max_inflight: Max number of files to upload in parallel before blocking the main thread.
        """
        if not s3_path.startswith("s3://"):
            raise ValueError("s3_path must start with 's3://'")

        self.local_root = Path(local_root).resolve()

        # Parse S3 path:
        parsed = urlparse(s3_path)
        self.bucket = parsed.netloc
        self.prefix = parsed.path.lstrip("/")

        # Thread pool management:
        self.max_inflight = max_inflight
        self.pool = ThreadPoolExecutor(max_workers=max(1, workers))
        self._futures: list[Future[object]] = list()

        # self.s3_client = boto3.client("s3", config=s3_config, endpoint_url=s3_endpoint_url)
        self.s3_client = get_s3_client()
        logger.debug(
            f"S3FileUploader initialized. Uploading from '{self.local_root}' "
            f"to 's3://{self.bucket}/{self.prefix}' with {workers} workers."
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write(self, local_file: Path | str) -> None:
        """Schedules a single local file for upload.

        The S3 key is automatically determined from the file's path relative to the `local_root` given
        at initialization.

        Example:
            .. code-block:: python

                local_root = "/tmp/results"
                S3FileUploader.write("/tmp/results/subdir/file.pt")
                # Uploads to: s3://[bucket]/[prefix]/subdir/file.pt

        Args:
            local_file: The path to the local file to upload.
        """
        local_file_path = Path(local_file).resolve()

        # Calculate the relative path to use as the S3 key
        try:
            relative_path = local_file_path.relative_to(self.local_root)
        except ValueError:
            raise ValueError(
                f"File {local_file_path} is not inside the specified local_root {self.local_root}"
            ) from None

        # Create the final S3 key (using POSIX paths, which S3 expects)
        s3_key = (Path(self.prefix) / relative_path).as_posix()

        # Enqueue the upload task
        self._enqueue(
            self._upload_file_to_s3,
            local_file_path=str(local_file_path),
            s3_key=s3_key,
        )

    def _upload_file_to_s3(self, local_file_path: str, s3_key: str) -> None:
        """
        The actual work function to upload a single file.
        This runs in the thread pool.
        """
        logger.trace(f"Uploading {local_file_path} to s3://{self.bucket}/{s3_key}")
        self.s3_client.upload_file(Filename=local_file_path, Bucket=self.bucket, Key=s3_key)
        logger.trace(f"Finished uploading {local_file_path}")

    def _enqueue(self, fn, *args, **kwargs) -> None:
        """Submit a background upload task with backpressure."""
        # This logic is identical to AsyncWriter
        if len(self._futures) >= self.max_inflight:
            logger.debug(f"Upload queue full ({self.max_inflight}). Waiting...")
            # Wait for at least one to complete
            done = next(iter(as_completed(self._futures)))
            self._futures.remove(done)
            done.result()  # Propagates exceptions from the thread

        self._futures.append(self.pool.submit(fn, *args, **kwargs))

    def close(self) -> None:
        """
        Waits for all in-flight uploads to complete and shuts down the thread pool.

        Propagates any exceptions raised by background upload tasks.
        """
        for future in as_completed(self._futures):
            future.result()  # Wait and check for exceptions
        self.pool.shutdown(wait=True)
        logger.debug(f"{self.__class__.__name__} pool shut down.")

    def upload_all(
        self,
        remove_source: bool = False,
        show_progress: bool = True,
    ) -> None:
        """A high-level helper to find all files in 'local_root', upload them, and optionally remove the local files.

        This is a class-based alternative to the `aws s3 sync` command.

        Args:
            remove_source: If True, deletes the entire `local_root` directory after all uploads are successful.
            show_progress: If True, shows a progress bar. Requires `rich` to be installed.
        """
        logger.info(f"Scanning {self.local_root} for files to upload...")
        files_to_upload = [f for f in self.local_root.rglob("*") if f.is_file()]

        if not files_to_upload:
            logger.warning("No files found to upload.")
            return

        logger.info(f"Found {len(files_to_upload)} files. Enqueuing for upload...")

        iterable = files_to_upload
        if show_progress:
            try:
                from rich.progress import track

                iterable = track(files_to_upload, description="Uploading...")  # type: ignore[assignment]
            except ImportError:
                logger.warning("Note: 'rich' not installed. Progress bar will not be shown.")

        # Enqueue all file uploads
        for file in iterable:
            self.write(file)

        # Wait for all uploads to finish
        logger.info("All files enqueued. Waiting for uploads to complete...")
        self.close()  # This waits for all futures to finish
        logger.info("All uploads complete.")

        # --- Optional: Remove Source ---
        if remove_source:
            logger.info(f"Removing local source directory: {self.local_root}")
            try:
                shutil.rmtree(self.local_root)
                logger.success("Local files removed.")
            except OSError as exc:
                logger.error(f"Error: Failed to remove local directory '{self.local_root}'. Details: {exc}")
