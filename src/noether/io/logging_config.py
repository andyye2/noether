#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import sys

from loguru import logger


def configure_logging(debug: bool = False) -> None:
    """Configure loguru logging format.

    Args:
        debug: Include caller details if True, keep minimal format otherwise. Defaults to False.

    Returns:

    """
    logger.remove()
    if debug:
        logger.add(
            sys.stderr,
            level="DEBUG",
            format=(
                "<green>{time:DD-MM-YYYY HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
        )
    else:
        logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:DD-MM-YYYY HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}",
        )
