#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
from typer.testing import CliRunner

from noether.io.cli import cli

runner = CliRunner()

MODULE_PATH = "noether.io.cli.cli_huggingface"


@pytest.fixture
def dummy_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy-token")
    # Prevent actual HF auth calls
    monkeypatch.setattr(
        "noether.io.credentials.get_credentials",
        lambda provider: {"HF_TOKEN": "dummy-token"},
    )


def test_cli_help(dummy_token):
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "huggingface" in result.stdout
    assert "aws" in result.stdout


def test_cli_hf_estimate_mocked(dummy_token, monkeypatch):
    monkeypatch.setattr(
        f"{MODULE_PATH}.estimate_hf_repo_size",  # patch where CLI uses it
        lambda *a, **kw: 10485760,  # 10 MB
    )
    result = runner.invoke(cli.app, ["huggingface", "estimate", "repo-id"])
    assert result.exit_code == 0
    assert "~10.00 MB" in result.stdout


def test_cli_hf_ext_no_matches(dummy_token, monkeypatch):
    monkeypatch.setattr(
        f"{MODULE_PATH}.fetch_huggingface_by_extension",
        lambda *a, **kw: print("No files with extension"),
    )
    result = runner.invoke(cli.app, ["huggingface", "ext", "repo-id", ".xyz", "/tmp"])
    assert result.exit_code == 0
    assert "No files with extension" in result.stdout


def test_cli_hf_file_mocked(dummy_token, monkeypatch):
    called = {}
    monkeypatch.setattr(
        f"{MODULE_PATH}.fetch_huggingface_file",
        lambda *a, **kw: called.setdefault("ok", True),
    )
    result = runner.invoke(cli.app, ["huggingface", "file", "repo-id", "file.txt", "/tmp"])
    assert result.exit_code == 0
    assert called.get("ok", False)


def test_cli_hf_snapshot_mocked(dummy_token, monkeypatch):
    called = {}
    monkeypatch.setattr(
        f"{MODULE_PATH}.fetch_huggingface_repo_snapshot",
        lambda *a, **kw: called.setdefault("ok", True),
    )
    result = runner.invoke(cli.app, ["huggingface", "snapshot", "repo-id", "/tmp"])
    assert result.exit_code == 0
    assert called.get("ok", False)
