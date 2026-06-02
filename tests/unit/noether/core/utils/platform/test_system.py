#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import sys
from unittest.mock import mock_open, patch

import pytest

from noether.core.utils.platform import system


def test_get_cli_command():
    """Test that sys.argv is correctly reconstructed into a string."""
    fake_argv = ["/usr/bin/train.py", "--batch_size", "32", "--name", "my experiment"]

    with patch.object(sys, "argv", fake_argv):
        cmd = system.get_cli_command()

        # Expect: python train.py --batch_size 32 --name 'my experiment'
        assert "python train.py" in cmd
        assert "--batch_size 32" in cmd
        assert "'my experiment'" in cmd


@patch("shutil.which")
def test_get_installed_cuda_version_not_found(mock_which):
    """Test behavior when nvidia-smi is not in PATH."""
    mock_which.return_value = None
    assert system.get_installed_cuda_version() is None


@patch("shutil.which")
@patch("os.popen")
def test_get_installed_cuda_version_found(mock_popen, mock_which):
    """Test parsing a valid nvidia-smi output."""
    mock_which.return_value = "/usr/bin/nvidia-smi"

    # Mock the file-like object returned by os.popen:
    mock_popen.return_value.read.return_value = """
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.13    Driver Version: 525.60.13    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
    """

    version = system.get_installed_cuda_version()
    assert version == "12.0"


@patch("shutil.which")
@patch("os.popen")
def test_get_installed_cuda_version_parsing_failure(mock_popen, mock_which):
    """Test behavior when output format changes or doesn't match expectation."""
    mock_which.return_value = "/usr/bin/nvidia-smi"
    # Output without "CUDA Version:"
    mock_popen.return_value.read.return_value = "NVIDIA-SMI 525.60.13  Driver Version: 525.60.13"

    version = system.get_installed_cuda_version()
    assert version is None


@pytest.fixture
def mock_dependencies():
    """Mock all the heavy external dependencies for log_system_info."""
    with (
        patch("noether.core.utils.platform.system.logger") as mock_logger,
        patch("platform.uname") as mock_uname,
        patch("platform.platform") as mock_platform,
        patch("psutil.virtual_memory") as mock_mem,
        patch("noether.core.utils.platform.system.get_total_cpu_count") as mock_cpu,
        patch("noether.core.utils.platform.system.get_installed_cuda_version") as mock_cuda,
        patch("torch.__version__", "2.0.0"),
        patch("torch.version.cuda", "11.8"),
    ):
        # Setup decent defaults
        mock_uname.return_value.node = "test-node"
        mock_platform.return_value = "Linux-Test"
        mock_mem.return_value.total = 16_000_000_000  # 16GB
        mock_cpu.return_value = 8

        yield mock_logger, mock_cuda


def test_log_system_info_basic(mock_dependencies):
    """Test standard logging flow with no git info."""
    mock_logger, mock_cuda = mock_dependencies
    mock_cuda.return_value = "12.0"

    # Mock Path.exists to return False for .git/FETCH_HEAD
    with patch("pathlib.Path.exists", return_value=False):
        system.log_system_info()

    # Check that critical info was logged.
    # We verify that specific strings were passed to logger.debug:
    logs = [call.args[0] for call in mock_logger.debug.call_args_list]

    assert any("SYSTEM INFO" in l for l in logs)
    assert any("test-node" in l for l in logs)  # hostname
    assert any("CUDA version: 12.0" in l for l in logs)
    assert any("could not retrieve current git commit" in call.args[0] for call in mock_logger.warning.call_args_list)


def test_log_system_info_with_git(mock_dependencies):
    """Test logging when .git files exist."""
    mock_logger, _ = mock_dependencies

    # Mock .git/FETCH_HEAD existing
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data="a1b2c3d4e5f67890" * 3)),
        patch("os.popen") as mock_popen,
    ):
        # Mock git describe output:
        mock_popen.return_value.read.return_value = "v1.0.0"

        system.log_system_info()

    logs = [call.args[0] for call in mock_logger.debug.call_args_list]

    assert any("Framework Git hash: a1b2c3d4" in l for l in logs)
    assert any("Framework git tag: v1.0.0" in l for l in logs)


def test_log_system_info_empty_git_file(mock_dependencies):
    """Test edge case where FETCH_HEAD exists but is empty."""
    mock_logger, _ = mock_dependencies

    with patch("pathlib.Path.exists", return_value=True), patch("builtins.open", mock_open(read_data="")):  # Empty file
        system.log_system_info()

    # Should warn about empty content
    warnings = [call.args[0] for call in mock_logger.warning.call_args_list]
    assert any("has no content" in w for w in warnings)


def test_log_system_info_no_cuda(mock_dependencies):
    """Test logging when CUDA is missing."""
    mock_logger, mock_cuda = mock_dependencies
    mock_cuda.return_value = None

    with patch("pathlib.Path.exists", return_value=False):
        system.log_system_info()

    logs = [call.args[0] for call in mock_logger.debug.call_args_list]
    assert any("CUDA not installed" in l for l in logs)
