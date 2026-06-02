#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import os
from unittest.mock import patch

import pytest

from noether.core.utils.platform import worker

MODULE_PATH = "noether.core.utils.platform.worker"


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear LRU caches before every test to ensure fresh execution."""
    worker.get_total_cpu_count.cache_clear()
    worker._get_device_count.cache_clear()
    yield
    worker.get_total_cpu_count.cache_clear()
    worker._get_device_count.cache_clear()


@pytest.fixture
def mock_nvidia_smi():
    """Mock the subprocess.run call for nvidia-smi."""
    with patch("subprocess.run") as mock_run:
        yield mock_run


@pytest.fixture
def mock_cpu_count():
    """Mock OS CPU counting mechanisms."""
    # create=True is REQUIRED here because os.sched_getaffinity does not exist on Windows/Mac, so standard patching
    # would fail with AttributeError.
    with patch("os.sched_getaffinity", create=True) as mock_affinity, patch("os.cpu_count") as mock_os_count:
        yield mock_affinity, mock_os_count


def test_device_count_no_nvidia_smi(mock_nvidia_smi):
    """Test behavior when nvidia-smi is missing (OSError)."""
    mock_nvidia_smi.side_effect = OSError("No file")

    assert worker._get_device_count() == 1  # Fallback to 1 CPU device


def test_device_count_empty_output(mock_nvidia_smi):
    """Test when nvidia-smi returns empty string."""
    mock_nvidia_smi.return_value.stdout = ""
    assert worker._get_device_count() == 1


def test_device_count_standard_gpus(mock_nvidia_smi):
    """Test parsing of standard GPUs (no MIG)."""
    output = """
GPU 0: NVIDIA A100 (UUID: GPU-1)
GPU 1: NVIDIA A100 (UUID: GPU-2)
    """.strip()
    mock_nvidia_smi.return_value.stdout = output

    assert worker._get_device_count() == 2


def test_device_count_mig_instances(mock_nvidia_smi):
    """Test parsing when MIG is enabled."""
    # GPU 0 has 2 MIG instances, GPU 1 has no MIG (so it counts as 1 itself)
    output = """
GPU 0: NVIDIA A100 (UUID: GPU-1)
  MIG 1g.5gb     Device 0: (UUID: MIG-A)
  MIG 1g.5gb     Device 1: (UUID: MIG-B)
GPU 1: NVIDIA A100 (UUID: GPU-2)
    """.strip()
    mock_nvidia_smi.return_value.stdout = output

    # Expected: 2 (from GPU 0) + 1 (from GPU 1) = 3
    assert worker._get_device_count() == 3


def test_device_count_mixed_parsing(mock_nvidia_smi):
    """Test robust parsing with blank lines or partial MIG data."""
    output = """
GPU 0: NVIDIA A100 (UUID: GPU-1)

GPU 1: NVIDIA A100 (UUID: GPU-2)
  MIG 1g.5gb     Device 0: (UUID: MIG-A)
    """.strip()
    mock_nvidia_smi.return_value.stdout = output

    # GPU 0 (1) + GPU 1 (1 MIG) = 2
    assert worker._get_device_count() == 2


@patch("sys.platform", "linux")
@patch("os.name", "posix")
def test_total_cpu_count_linux(mock_cpu_count):
    """Test standard Linux behavior using sched_getaffinity."""
    mock_affinity, _ = mock_cpu_count
    # simulate returning a set of 32 CPUs
    mock_affinity.return_value = set(range(32))

    assert worker.get_total_cpu_count() == 32


@patch("sys.platform", "linux")
@patch("os.name", "posix")
def test_total_cpu_count_fallback(mock_cpu_count):
    """Test fallback to os.cpu_count if affinity fails."""
    mock_affinity, mock_os_count = mock_cpu_count
    mock_affinity.side_effect = AttributeError("Not on Linux")
    mock_os_count.return_value = 16

    assert worker.get_total_cpu_count() == 16


@patch("os.name", "nt")
def test_total_cpu_count_windows(mock_cpu_count):
    """Test that Windows uses os.cpu_count()."""
    mock_affinity, mock_os_count = mock_cpu_count

    # Simulate sched_getaffinity failing (which happens on Windows)
    # This forces the code to catch the exception and use os.cpu_count
    mock_affinity.side_effect = AttributeError
    mock_os_count.return_value = 12

    assert worker.get_total_cpu_count() == 12


@patch("sys.platform", "darwin")
def test_total_cpu_count_mac(mock_cpu_count):
    """Test that Mac uses os.cpu_count()."""
    mock_affinity, mock_os_count = mock_cpu_count

    # Simulate sched_getaffinity failing (which happens on Mac)
    mock_affinity.side_effect = AttributeError
    mock_os_count.return_value = 10

    assert worker.get_total_cpu_count() == 10


@patch(f"{MODULE_PATH}.is_managed")
@patch(f"{MODULE_PATH}.get_total_cpu_count")
@patch(f"{MODULE_PATH}._get_device_count")
def test_fair_cpu_local_env(mock_devices, mock_total_cpus, mock_managed):
    """
    Scenario: Local machine (not managed/SLURM).
    Logic: Total CPUs / Device Count - Reserve
    """
    mock_managed.return_value = False
    mock_total_cpus.return_value = 32
    mock_devices.return_value = 4

    # 32 / 4 = 8. Reserve 1 = 7.
    assert worker.get_fair_cpu_count(reserve_for_main=1) == 7


@patch(f"{MODULE_PATH}.is_managed")
@patch(f"{MODULE_PATH}.get_total_cpu_count")
@patch(f"{MODULE_PATH}._get_device_count")
def test_fair_cpu_local_env_clamping(mock_devices, mock_total_cpus, mock_managed):
    """Test that result doesn't go below 0."""
    mock_managed.return_value = False
    mock_total_cpus.return_value = 4
    mock_devices.return_value = 4

    # 4 / 4 = 1. Reserve 2 = -1. Should clamp to 0.
    assert worker.get_fair_cpu_count(reserve_for_main=2) == 0


@patch(f"{MODULE_PATH}.is_managed")
@patch(f"{MODULE_PATH}.get_total_cpu_count")
@patch(f"{MODULE_PATH}._get_device_count")
def test_fair_cpu_slurm_cpus_per_task(mock_devices, mock_total_cpus, mock_managed):
    """
    Scenario: SLURM with explicit SLURM_CPUS_PER_TASK.
    """
    mock_managed.return_value = True
    mock_total_cpus.return_value = 40
    mock_devices.return_value = 1

    env_vars = {"SLURM_CPUS_PER_TASK": "10", "SLURM_NTASKS_PER_NODE": "1"}

    with patch.dict(os.environ, env_vars):
        # Result should be CPUS_PER_TASK (10) - Reserve (1) = 9
        # It ignores the 'total_cpu_count' variable for the calculation, using the env var instead.
        assert worker.get_fair_cpu_count(reserve_for_main=1) == 9


@patch(f"{MODULE_PATH}.is_managed")
@patch(f"{MODULE_PATH}.get_total_cpu_count")
@patch(f"{MODULE_PATH}._get_device_count")
def test_fair_cpu_slurm_derived(mock_devices, mock_total_cpus, mock_managed):
    """
    Scenario: SLURM without CPUS_PER_TASK, derived from ON_NODE / TASKS_PER_NODE.
    """
    mock_managed.return_value = True
    mock_total_cpus.return_value = 64
    mock_devices.return_value = 8

    # Unset CPUS_PER_TASK, set others
    env_vars = {
        "SLURM_CPUS_ON_NODE": "64",
        "SLURM_NTASKS_PER_NODE": "8",  # 8 tasks per node
    }

    with patch.dict(os.environ, env_vars, clear=True):
        # 64 / 8 = 8 CPUs per task.
        # Reserve 1 = 7.
        assert worker.get_fair_cpu_count(reserve_for_main=1) == 7


@patch(f"{MODULE_PATH}.is_managed")
@patch(f"{MODULE_PATH}.get_total_cpu_count")
@patch(f"{MODULE_PATH}._get_device_count")
def test_fair_cpu_slurm_fallback(mock_devices, mock_total_cpus, mock_managed):
    """
    Scenario: SLURM managed but missing environment variables.
    Should fallback to local division logic.
    """
    mock_managed.return_value = True
    mock_total_cpus.return_value = 20
    mock_devices.return_value = 2

    # Empty env:
    with patch.dict(os.environ, {}, clear=True):
        # Fallback: 20 / 2 = 10. Reserve 1 = 9.
        assert worker.get_fair_cpu_count(reserve_for_main=1) == 9


def test_get_int_env():
    """Test safe environment variable parsing."""
    with patch.dict(os.environ, {"TEST_INT": "123", "TEST_BAD": "abc"}):
        # Valid
        assert worker._get_int_env("TEST_INT") == 123

        # Missing
        assert worker._get_int_env("MISSING") is None
        assert worker._get_int_env("MISSING", default=5) == 5

        # Malformed (logs warning and returns default)
        assert worker._get_int_env("TEST_BAD", default=10) == 10
