#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from unittest.mock import patch

import pytest

from noether.core.utils.code import store_code_archive


@pytest.fixture
def mock_paths(tmp_path):
    """Returns dummy code and output paths."""
    code_path = tmp_path / "src"
    output_path = tmp_path / "output"
    code_path.mkdir()
    output_path.mkdir()
    return code_path, output_path


@patch("shutil.which")
def test_git_not_installed(mock_which, mock_paths):
    """Test that function returns None if git is missing."""
    code_path, output_path = mock_paths
    mock_which.return_value = None  # git not found

    result = store_code_archive(code_path, output_path)

    assert result is None
    # Ensure no subprocess calls were made:
    assert mock_which.called


@patch("shutil.which")
@patch("subprocess.check_output")
@patch("subprocess.run")
def test_archive_creation_with_stash(mock_run, mock_check_output, mock_which, mock_paths):
    """Test flow where git stash creates a commit hash."""
    code_path, output_path = mock_paths

    mock_which.return_value = "/usr/bin/git"
    mock_check_output.return_value = "stash_hash_123\n"  # Simulate git stash output

    result = store_code_archive(code_path, output_path)

    expected_archive = output_path / "code.tar.gz"
    assert result == expected_archive

    # Verify git stash was called:
    mock_check_output.assert_called_with(["git", "stash", "create", "--include-untracked"], cwd=code_path, text=True)

    # Verify git archive was called with the stash hash:
    mock_run.assert_called_with(
        ["git", "archive", "-o", str(expected_archive), "stash_hash_123"], cwd=code_path, check=True
    )


@patch("shutil.which")
@patch("subprocess.check_output")
@patch("subprocess.run")
def test_archive_creation_no_stash_changes(mock_run, mock_check_output, mock_which, mock_paths):
    """Test flow where git stash returns empty (clean repo), defaulting to HEAD."""
    code_path, output_path = mock_paths

    # Mocks:
    mock_which.return_value = "/usr/bin/git"
    mock_check_output.return_value = ""  # Simulate clean repo (no stash needed)

    _ = store_code_archive(code_path, output_path)

    expected_archive = output_path / "code.tar.gz"

    # Verify git archive was called with HEAD:
    mock_run.assert_called_with(["git", "archive", "-o", str(expected_archive), "HEAD"], cwd=code_path, check=True)
