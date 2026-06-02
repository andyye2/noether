#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from noether.core.utils.logging.base import MessageCounter, add_global_handlers, log_from_all_ranks

MODULE_PATH = "noether.core.utils.logging.base"


@pytest.fixture
def reset_logging():
    """
    Saves the state of the root logger before a test and restores it afterwards.
    This is crucial because 'add_global_handlers' modifies global state.
    """
    logger = logging.getLogger()
    old_level = logger.level
    old_handlers = logger.handlers[:]

    yield

    # Teardown: Restore
    logger.setLevel(old_level)
    logger.handlers = old_handlers


@pytest.fixture
def mock_dependencies():
    """
    Mocks external dependencies to avoid ImportErrors and isolate logic.
    """
    with (
        patch(f"{MODULE_PATH}.is_rank0") as mock_rank0,
    ):
        yield mock_rank0


def test_message_counter_counting():
    """Test that it counts Warnings and Errors, but ignores Info."""
    counter = MessageCounter()

    # Create fake records
    # Level 20 = INFO, 30 = WARNING, 40 = ERROR
    info_rec = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0, msg="hi", args=(), exc_info=None
    )
    warn_rec = logging.LogRecord(
        name="test", level=logging.WARNING, pathname="", lineno=0, msg="oops", args=(), exc_info=None
    )
    error_rec = logging.LogRecord(
        name="test", level=logging.ERROR, pathname="", lineno=0, msg="bad", args=(), exc_info=None
    )

    counter.emit(info_rec)
    counter.emit(warn_rec)
    counter.emit(warn_rec)  # Second warning
    counter.emit(error_rec)

    assert counter.counts[logging.INFO] == 0
    assert counter.counts[logging.WARNING] == 2
    assert counter.counts[logging.ERROR] == 1


def test_message_counter_logging(caplog):
    """Test the log() method outputs the summary correctly."""
    counter = MessageCounter()
    counter.counts[logging.WARNING] = 5
    counter.counts[logging.ERROR] = 2

    with caplog.at_level(logging.INFO):
        counter.log()

    # Check that it logged the summary:
    assert "Encountered 5 warnings" in caplog.text
    assert "Encountered 2 errors" in caplog.text


def test_log_from_all_ranks_context(reset_logging):
    """Test context manager switches level to DEBUG and restores it."""
    logger = logging.getLogger()

    logger.setLevel(logging.CRITICAL)

    with log_from_all_ranks():
        assert logger.level == logging.DEBUG

    assert logger.level == logging.CRITICAL


def test_add_handlers_rank0_console_only(reset_logging, mock_dependencies):
    """
    Scenario: Rank 0 (Master), no file logging, debug=False.
    Expectation: StreamHandler (INFO) + MessageCounter added.
    """
    mock_rank0 = mock_dependencies
    mock_rank0.return_value = True  # Simulate Main Process

    handler = add_global_handlers(log_file_uri=None, debug=False)

    logger = logging.getLogger()
    assert isinstance(handler, MessageCounter)

    # Should have StreamHandler and MessageCounter:
    assert len(logger.handlers) == 2
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    # Check console handler level:
    console = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)][0]  # noqa: RUF015
    assert console.level == logging.INFO


def test_add_handlers_rank0_debug_mode(reset_logging, mock_dependencies):
    """
    Scenario: Rank 0, debug=True.
    Expectation: Console handler set to DEBUG.
    """
    mock_rank0 = mock_dependencies
    mock_rank0.return_value = True

    add_global_handlers(log_file_uri=None, debug=True)

    logger = logging.getLogger()
    console = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)][0]  # noqa: RUF015
    assert console.level == logging.DEBUG


def test_add_handlers_rank0_with_file(reset_logging, mock_dependencies, tmp_path):
    """
    Scenario: Rank 0, file logging enabled.
    Expectation: FileHandler added.
    """
    mock_rank0 = mock_dependencies
    mock_rank0.return_value = True

    log_file = tmp_path / "test.log"

    add_global_handlers(log_file_uri=log_file, debug=False)

    logger = logging.getLogger()
    # Stream + File + Counter = 3 handlers
    assert len(logger.handlers) == 3

    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    # Check it points to our temp file (resolving paths to avoid symlink mismatches):
    assert Path(file_handlers[0].baseFilename).resolve() == log_file.resolve()


def test_add_handlers_non_rank0(reset_logging, mock_dependencies):
    """
    Scenario: Rank 1 (Worker).
    Expectation: Logger level set to `CRITICAL` to suppress output.
    """
    mock_rank0 = mock_dependencies
    mock_rank0.return_value = False  # Simulate Worker Process

    add_global_handlers(log_file_uri=None)

    logger = logging.getLogger()
    assert logger.level == logging.CRITICAL
