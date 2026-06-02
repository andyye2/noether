#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
import os
import platform
import shlex
import shutil
import sys
from pathlib import Path

import psutil  # type: ignore
import torch

from noether.core.distributed import get_local_rank, get_rank
from noether.core.utils.logging import log_from_all_ranks, short_number_str
from noether.core.utils.platform.worker import get_total_cpu_count

logger = logging.getLogger(__name__)


def get_cli_command() -> str:
    """Return the command with which the script was called.

    Attention: This assumes the script was started with `python` (not `python3` or similar).

    Returns:
        The command with which the script was called.
    """
    # print the command with which the script was called
    # https://stackoverflow.com/questions/37658154/get-command-line-arguments-as-string
    script_str = f"python {Path(sys.argv[0]).name}"
    argstr = " ".join(map(shlex.quote, sys.argv[1:]))
    return f"{script_str} {argstr}"


def get_installed_cuda_version() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    """Returns the installed CUDA version or None if nvidia-smi is not available."""
    nvidia_smi_lines = os.popen("nvidia-smi").read().strip().split("\n")
    for line in nvidia_smi_lines:
        if "CUDA Version:" in line:
            return line[line.index("CUDA Version: ") + len("CUDA Version: ") : -1].strip()
    return None


def log_system_info() -> None:
    """Logs system information like OS, CUDA version, and Python version."""
    logger.debug("SYSTEM INFO")
    logger.debug(f"CWD: {os.getcwd()}")
    logger.debug(f"host name: {platform.uname().node}")
    logger.debug(f"OS: {platform.platform()}")
    logger.debug(f"OS version: {platform.version()}")
    logger.debug(f"sys.executable: {sys.executable}")
    logger.debug(f"torch.version: {torch.__version__}")
    logger.debug(f"torch.version.cuda: {torch.version.cuda if torch.version.cuda is not None else 'N/A'}")
    cuda_version = get_installed_cuda_version()
    if cuda_version is not None:
        logger.debug(f"CUDA version: {cuda_version}")
    else:
        logger.debug("CUDA not installed or nvidia-smi not available")

    # print hash of latest git commit (git describe or similar stuff is a bit ugly because it would require the
    # git.exe path to be added in path as conda/python do something with the path and don't use the system
    # PATH variable by default)
    git_hash_file = Path(".git") / "FETCH_HEAD"
    if git_hash_file.exists():
        with open(git_hash_file) as f:
            lines = f.readlines()
            if len(lines) == 0:
                # this happened when I didn't have internet
                logger.warning(".git/FETCH_HEAD has no content")
            else:
                git_hash = lines[0][:40]
                logger.debug(f"Framework Git hash: {git_hash}")
        git_tag = os.popen("git describe --abbrev=0").read().strip()
        logger.debug(f"Framework git tag: {git_tag}")
    else:
        logger.warning("could not retrieve current git commit hash from ./.git/FETCH_HEAD")
    with log_from_all_ranks():
        logger.debug(
            f"initialized process rank={get_rank()} local_rank={get_local_rank()} pid={os.getpid()} "
            f"hostname={platform.uname().node}"
        )
    logger.debug(f"total_cpu_count: {get_total_cpu_count()}")
    logger.debug(f"total_memory: {short_number_str(psutil.virtual_memory().total)}")
