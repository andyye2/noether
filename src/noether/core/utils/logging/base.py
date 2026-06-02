#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
import sys
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from logging import LogRecord
from pathlib import Path

from noether.core.distributed.config import is_rank0
from noether.core.utils.logging.formatting import CustomFormatter


@contextmanager
def log_from_all_ranks() -> Generator[None, None, None]:
    """ContextManager that changes the log level to Info for all ranks and resets it to CRITICAL for non-rank0 ranks."""
    logger = logging.getLogger()
    prev_level = logger.level
    logger.setLevel(logging.DEBUG)
    yield
    logger.setLevel(level=prev_level)


class MessageCounter(logging.Handler):
    """A logging handler that counts the number of messages logged at each level starting from WARNING."""

    def __init__(self) -> None:
        super().__init__()
        self.min_level = logging.WARNING
        self.counts: dict[int, int] = defaultdict(int)

    def emit(self, record: LogRecord) -> None:
        if record.levelno >= self.min_level:
            self.counts[record.levelno] += 1

    def log(self) -> None:
        logger = logging.getLogger(__name__)
        for level in [logging.WARNING, logging.ERROR]:
            logger.info(f"Encountered {self.counts[level]} {logging.getLevelName(level).lower()}s")


def add_global_handlers(log_file_uri: Path | None = None, debug: bool = False) -> MessageCounter:
    """Set up `logging.getLogger()` to log to stdout and optionally to a file.

    Sets up logging for distributed runs: only rank0 logs to console and file,
    other ranks log only CRITICAL messages to suppress output.
    This also adds a MessageCounter handler to count the number of messages logged at each level.

    Args:
        log_file_uri: The path to the log file. If None, no file logging is done.
        debug: Whether to log debug messages to stdout.

    Returns:
        The MessageCounter handler.
    """
    log_format = "%(asctime)s %(levelname).1s %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"
    logger = logging.getLogger()
    # set log level to DEBUG to capture all messages; handlers will filter levels as needed
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(CustomFormatter(log_format, datefmt))
    logger.addHandler(console_handler)
    if is_rank0():
        if log_file_uri is not None:
            # always log DEBUG to file
            file_handler = logging.FileHandler(log_file_uri, mode="a")
            file_handler.setFormatter(CustomFormatter(log_format, datefmt, colors=False))
            logger.addHandler(file_handler)
            logger.info(f"logging to file: {log_file_uri.as_posix()}")
    else:
        # subprocesses log warnings to stderr --> logging.CRITICAL prevents this
        logger.setLevel(logging.CRITICAL)
    message_counter = MessageCounter()
    logger.addHandler(message_counter)
    return message_counter
