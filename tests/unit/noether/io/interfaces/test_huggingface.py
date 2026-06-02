#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

from noether.io.interfaces import huggingface

MODULE_PATH = "noether.io.interfaces.huggingface"


@pytest.fixture
def dummy_creds(monkeypatch):
    """Mock HF credentials so no real secrets are needed."""
    monkeypatch.setattr(f"{MODULE_PATH}.get_credentials", lambda provider: {"HF_TOKEN": "dummy-token"})


@pytest.fixture
def loguru_capture():
    buf = io.StringIO()
    handler_id = logger.add(buf, format="{message}")
    yield buf
    logger.remove(handler_id)


def test_estimate_hf_repo_size_with_sizes(dummy_creds, monkeypatch) -> None:
    fake_info = SimpleNamespace(
        siblings=[
            SimpleNamespace(rfilename="a.th", size=1024),
            SimpleNamespace(rfilename="b.th", size=2048),
        ]
    )
    monkeypatch.setattr(f"{MODULE_PATH}.HfApi.model_info", lambda *a, **kw: fake_info)
    size = huggingface.estimate_hf_repo_size("repo", "model")
    assert size == 1024 + 2048


def test_estimate_hf_repo_size_with_head_request(dummy_creds, monkeypatch) -> None:
    # One file has no size and triggers HEAD request
    fake_info = SimpleNamespace(
        siblings=[
            SimpleNamespace(rfilename="a.th", size=None),
        ]
    )

    monkeypatch.setattr(f"{MODULE_PATH}.HfApi.model_info", lambda *a, **kw: fake_info)

    class DummyResp:
        status_code = 200
        headers = {"Content-Length": "512"}

        def raise_for_status(self):
            pass

    monkeypatch.setattr(f"{MODULE_PATH}.requests.head", lambda *a, **kw: DummyResp())

    size = huggingface.estimate_hf_repo_size("repo", "model")
    assert size == 512


def test_fetch_huggingface_repo_snapshot(dummy_creds, monkeypatch, tmp_path: Path) -> None:
    called = {}

    def fake_snapshot_download(**kwargs):
        called["args"] = kwargs

    monkeypatch.setattr(f"{MODULE_PATH}.snapshot_download", fake_snapshot_download)

    huggingface.fetch_huggingface_repo_snapshot("repo", tmp_path)
    assert called["args"]["repo_id"] == "repo"
    assert str(tmp_path) in called["args"]["local_dir"]


def test_fetch_huggingface_file(dummy_creds, monkeypatch, tmp_path: Path) -> None:
    called = {}

    def fake_download(**kwargs):
        called["args"] = kwargs

    monkeypatch.setattr(f"{MODULE_PATH}.hf_hub_download", fake_download)

    huggingface.fetch_huggingface_file("repo", "file.txt", tmp_path)
    assert called["args"]["filename"] == "file.txt"
    assert str(tmp_path) in called["args"]["local_dir"]


def test_fetch_huggingface_by_extension_filters_and_downloads(dummy_creds, monkeypatch, tmp_path: Path) -> None:
    repo_files = ["keep1.th", "skip.txt", "keep2.th"]

    monkeypatch.setattr(f"{MODULE_PATH}.HfApi.list_repo_files", lambda *a, **kw: repo_files)

    downloaded = []

    def fake_download(**kwargs):
        downloaded.append(kwargs["filename"])

    monkeypatch.setattr(f"{MODULE_PATH}.hf_hub_download", fake_download)

    huggingface.fetch_huggingface_by_extension(
        repo_id="repo",
        extension=".th",
        local_dir=tmp_path,
        max_workers=2,
    )

    assert set(downloaded) == {"keep1.th", "keep2.th"}


def test_fetch_huggingface_by_extension_no_matches(dummy_creds, monkeypatch, tmp_path: Path, loguru_capture) -> None:
    monkeypatch.setattr(f"{MODULE_PATH}.HfApi.list_repo_files", lambda *a, **kw: ["file.txt", "other.json"])

    huggingface.fetch_huggingface_by_extension(
        repo_id="repo",
        extension=".th",
        local_dir=tmp_path,
    )

    logs = loguru_capture.getvalue()

    assert "No files with extension" in logs
