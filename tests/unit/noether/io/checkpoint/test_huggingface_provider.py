#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import pytest
import requests

from noether.io.checkpoint.providers.huggingface_provider import HFProvider

MODULE_PATH = "noether.io.checkpoint.providers.huggingface_provider"


class StubProgress:
    def __init__(self, label: str, total_bytes: int | None) -> None:
        self.label = label
        self.total_bytes = total_bytes
        self.updates: list[int] = []
        self.closed = False

    def update(self, n: int) -> None:
        self.updates.append(n)

    def close(self) -> None:
        self.closed = True


class FakeResponse:
    """Mimic a requests.Response subset needed by provider (HEAD/GET)."""

    def __init__(
        self,
        *,
        status: int = 200,
        headers: dict[str, Any] | None = None,
        chunks: list[bytes] | None = None,
    ) -> None:
        self.status_code = status
        self.headers = headers or {}
        self._chunks = chunks or []

    # Context manager support
    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # nothing special to clean
        return

    def raise_for_status(self) -> None:
        if not (200 <= self.status_code < 400):
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int):
        for c in self._chunks:
            if c:
                yield c


class StubRequests:
    """Capture HEAD/GET calls and return seeded FakeResponses."""

    def __init__(self) -> None:
        self.last_head: tuple[tuple, dict] | None = None
        self.last_get: tuple[tuple, dict] | None = None
        self._head_responses: list[FakeResponse] = []
        self._get_responses: list[FakeResponse] = []

    def seed_head(self, response: FakeResponse) -> None:
        self._head_responses.append(response)

    def seed_get(self, response: FakeResponse) -> None:
        self._get_responses.append(response)

    def head(self, *args, **kwargs) -> FakeResponse:
        self.last_head = (args, kwargs)
        return self._head_responses.pop(0) if self._head_responses else FakeResponse(status=404)

    def get(self, *args, **kwargs) -> FakeResponse:
        self.last_get = (args, kwargs)
        return self._get_responses.pop(0) if self._get_responses else FakeResponse(status=404)


def mk_hf_uri(repo: str = "org/repo", filename: str = "model.bin", rev: str | None = None) -> str:
    # hf://org/repo[@rev]?filename=FILENAME
    if rev:
        return f"hf://{repo}@{rev}?filename={filename}"
    return f"hf://{repo}?filename={filename}"


# --- Basic scheme tests:


def test_can_handle_hf():
    p = HFProvider()
    assert p.can_handle("hf://org/repo?filename=m.bin")
    assert not p.can_handle("s3://bucket/key")
    assert not p.can_handle("/local/file.bin")


def test_bad_uri_rejected_missing_filename(monkeypatch):
    """Missing ?filename should raise before any network call."""
    provider = HFProvider()
    # Prevent accidental network
    stub = StubRequests()
    monkeypatch.setattr(f"{MODULE_PATH}.requests", stub)
    with pytest.raises(ValueError):
        provider.fetch("hf://org/repo", Path("/tmp"))


def test_bad_uri_rejected_empty_repo(monkeypatch):
    """Empty repo part should raise before any network call."""
    provider = HFProvider()
    stub = StubRequests()
    monkeypatch.setattr(f"{MODULE_PATH}.requests", stub)
    with pytest.raises(ValueError):
        provider.fetch("hf://?filename=a.bin", Path("/tmp"))


# --- estimate_size (HEAD):

# def test_estimate_size_ok(monkeypatch):
#     provider = HFProvider()
#     stub = StubRequests()
#     stub.seed_head(FakeResponse(status=200, headers={"Content-Length": "123456"}))
#
#     # monkeypatch requests and progress class
#     monkeypatch.setattr(
#         "emmi_data_management.checkpoint.providers.huggingface_provider.requests",
#         stub,
#     )
#     # no token
#     monkeypatch.setattr(
#         "emmi_data_management.checkpoint.providers.huggingface_provider.get_credentials",
#         lambda _provider: {},
#     )
#
#     size = provider.estimate_size(mk_hf_uri("EmmiAI/AB-UPT", "file.th", rev="main"))
#     assert size == 123456
#
#     # the URL should be built as /resolve/<rev>/<filename>
#     (args, kwargs) = stub.last_head
#     url = args[0]
#     assert "/EmmiAI/AB-UPT/resolve/main/file.th" in url
#     # no auth header without token
#     assert "headers" not in kwargs or "authorization" not in {k.lower() for k in kwargs.get("headers", {}).keys()}


def test_estimate_size_head_error_returns_none(monkeypatch):
    provider = HFProvider()
    stub = StubRequests()
    stub.seed_head(FakeResponse(status=403))  # forbidden -> treat as unknown

    monkeypatch.setattr(
        f"{MODULE_PATH}.requests",
        stub,
    )
    monkeypatch.setattr(
        f"{MODULE_PATH}.get_credentials",
        lambda _provider: {},
    )

    size = provider.estimate_size(mk_hf_uri("org/repo", "m.pt"))
    assert size is None


# --- fetch (GET streaming + atomic writer):


def test_fetch_streams_and_hashes(tmp_path: Path, monkeypatch):
    provider = HFProvider()

    chunks = [b"hello ", b"world"]
    total = sum(len(c) for c in chunks)

    stub = StubRequests()
    stub.seed_head(FakeResponse(status=200, headers={"Content-Length": str(total)}))
    stub.seed_get(FakeResponse(status=200, chunks=chunks))

    monkeypatch.setattr(
        f"{MODULE_PATH}.requests",
        stub,
    )
    monkeypatch.setattr(
        f"{MODULE_PATH}.LogProgress",
        StubProgress,
    )
    # no token
    monkeypatch.setattr(
        f"{MODULE_PATH}.get_credentials",
        lambda _provider: {},
    )

    out_path, sha = provider.fetch(
        mk_hf_uri("org/repo", "weights.bin", rev="main"),
        tmp_path,
        compute_hash=True,
    )
    assert out_path == tmp_path / "weights.bin"
    assert out_path.read_bytes() == b"hello world"
    # hash should be the sha256 of the content
    import hashlib

    assert sha == hashlib.sha256(b"hello world").hexdigest()


def test_fetch_without_head_still_downloads(tmp_path: Path, monkeypatch):
    provider = HFProvider()

    chunks = [b"a" * 10, b"b" * 5]

    stub = StubRequests()
    stub.seed_head(FakeResponse(status=403))  # HEAD forbidden
    stub.seed_get(FakeResponse(status=200, chunks=chunks))  # but GET works

    monkeypatch.setattr(
        f"{MODULE_PATH}.requests",
        stub,
    )
    monkeypatch.setattr(
        f"{MODULE_PATH}.LogProgress",
        StubProgress,
    )
    monkeypatch.setattr(
        f"{MODULE_PATH}.get_credentials",
        lambda _provider: {},
    )

    out_path, sha = provider.fetch(mk_hf_uri("org/repo", "theta.pt"), tmp_path, compute_hash=False)
    assert out_path.exists()
    assert out_path.read_bytes() == (b"a" * 10 + b"b" * 5)
    assert sha is None  # compute_hash=False returns None


def _auth_header(kwargs: dict) -> str:
    headers = kwargs.get("headers", {}) or {}
    for k, v in headers.items():
        if k.lower() == "authorization":
            return v or ""
    return ""


def test_fetch_injects_hf_token(monkeypatch, tmp_path: Path):
    p = HFProvider()
    chunks = [b"x"]

    stub = StubRequests()
    # HEAD may or may not be used by the provider
    stub.seed_head(FakeResponse(status=200, headers={"Content-Length": "1"}))
    stub.seed_get(FakeResponse(status=200, chunks=chunks))

    monkeypatch.setattr(f"{MODULE_PATH}.requests", stub)
    monkeypatch.setattr(f"{MODULE_PATH}.LogProgress", StubProgress)
    monkeypatch.setattr(
        f"{MODULE_PATH}.get_credentials",
        lambda provider: {"HF_TOKEN": "secret-token"},
    )

    result = p.fetch(mk_hf_uri("org/repo", "m.bin"), tmp_path, compute_hash=False)
    out = result[0] if isinstance(result, tuple) else result
    assert out.exists()

    # GET must happen; validate Authorization header (case-insensitive)
    assert stub.last_get is not None
    _, get_kwargs = stub.last_get
    auth = _auth_header(get_kwargs)
    assert auth.lower().startswith("bearer ")
    assert auth.split()[-1] == "secret-token"

    # If HEAD happened, it should also carry the token
    if stub.last_head is not None:
        _, head_kwargs = stub.last_head
        auth_head = _auth_header(head_kwargs)
        assert auth_head.lower().startswith("bearer ")
        assert auth_head.split()[-1] == "secret-token"


def test_fetch_http_error_propagates(monkeypatch, tmp_path: Path):
    provider = HFProvider()
    stub = StubRequests()
    stub.seed_head(FakeResponse(status=200, headers={"Content-Length": "10"}))
    stub.seed_get(FakeResponse(status=404))  # GET fails

    monkeypatch.setattr(
        f"{MODULE_PATH}.requests",
        stub,
    )
    monkeypatch.setattr(
        f"{MODULE_PATH}.get_credentials",
        lambda _provider: {},
    )

    with pytest.raises(requests.HTTPError):
        provider.fetch(mk_hf_uri("org/repo", "file.bin"), tmp_path)


def test_fetch_basename_written(tmp_path: Path, monkeypatch):
    provider = HFProvider()

    stub = StubRequests()
    stub.seed_head(FakeResponse(status=200, headers={"Content-Length": "4"}))
    stub.seed_get(FakeResponse(status=200, chunks=[b"data"]))

    monkeypatch.setattr(
        f"{MODULE_PATH}.requests",
        stub,
    )
    monkeypatch.setattr(
        f"{MODULE_PATH}.LogProgress",
        StubProgress,
    )
    monkeypatch.setattr(
        f"{MODULE_PATH}.get_credentials",
        lambda _provider: {},
    )

    out, _ = provider.fetch(
        mk_hf_uri("org/repo", "model.safetensors", rev="v1"),
        tmp_path,
        compute_hash=False,
    )
    assert out.name == "model.safetensors"
    assert out.read_bytes() == b"data"
