#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import time

from loguru import logger

from noether.io.cli.cli_utils import fmt_bytes


class LogProgress:
    """Percent-based progress logger.
    - If total is known: log at N% steps (adaptive: 20% for small files, 10% for larger).
    - If total unknown: log every `mb_step` MiB.

    Args:
        label: A label to use, e.g. a filename.
        total_bytes: [Optional] The total number of bytes to log.
        mb_step: [Optional] The number of MiB to log.
    """

    def __init__(self, *, label: str, total_bytes: int | None, mb_step: int = 32) -> None:
        self.label = label
        self.total = total_bytes or 0
        # adaptive steps: 20% for <100MB, else 10%
        self.step = 20 if (self.total and self.total < 100 * 1024 * 1024) else 10
        self.next_pct = self.step
        self.bytes_done = 0
        self._last_chunk_log_at = 0
        self._chunk_threshold = mb_step * 1024 * 1024
        self._start = time.monotonic()

    def update(self, delta: int) -> None:
        self.bytes_done += delta
        if self.total > 0:
            pct = int(self.bytes_done * 100 / self.total)
            logged = False
            while pct >= self.next_pct and self.next_pct <= 100:
                logger.info(f"[{self.label}] {self.next_pct}% ({fmt_bytes(self.bytes_done)}/{fmt_bytes(self.total)})")
                self.next_pct += self.step
                logged = True
            # If we just crossed 100% exactly, ensure a final line.
            if not logged and pct >= 100:
                logger.info(f"[{self.label}] 100% ({fmt_bytes(self.bytes_done)}/{fmt_bytes(self.total)})")
        else:
            # unknown total → log every N MiB
            if self.bytes_done - self._last_chunk_log_at >= self._chunk_threshold:
                self._last_chunk_log_at = self.bytes_done
                logger.info(f"[{self.label}] downloaded {fmt_bytes(self.bytes_done)}")

    def close(self) -> None:
        # Ensure we log completion for known totals
        if self.total > 0 and self.bytes_done >= self.total:
            return  # already logged 100%
        if self.total > 0:
            logger.info(f"[{self.label}] done ({fmt_bytes(self.bytes_done)}/{fmt_bytes(self.total)})")
        else:
            logger.info(f"[{self.label}] done ({fmt_bytes(self.bytes_done)})")
