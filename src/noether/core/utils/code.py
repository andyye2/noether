#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
import shutil
import subprocess
from pathlib import Path

_logger = logging.getLogger(__name__)


def store_code_archive(code_path: Path, output_path: Path) -> Path | None:
    """Store a copy of the code in the output directory for reproducibility."""

    if shutil.which("git") is None:
        _logger.warning("git is not installed -> cannot create code archive")
        return None

    archive_path = output_path / "code.tar.gz"
    # Create a git stash including untracked files to capture the current state
    try:
        stash_name = subprocess.check_output(
            ["git", "stash", "create", "--include-untracked"], cwd=code_path, text=True
        ).strip()
        if not stash_name:
            stash_name = "HEAD"
        # Create archive excluding files from .gitignore
        subprocess.run(["git", "archive", "-o", str(archive_path), stash_name], cwd=code_path, check=True)
        return archive_path
    except subprocess.CalledProcessError:
        _logger.warning("Failed to create code archive using git.")
        return None
