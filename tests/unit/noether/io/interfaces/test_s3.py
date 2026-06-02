#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import noether.io.interfaces.s3 as s3


class _FakeStreamingBody:
    def __init__(self, data: bytes):
        self._bio = io.BytesIO(data)

    def iter_chunks(self, chunk_size: int = 1024) -> Iterator[bytes]:
        while True:
            chunk = self._bio.read(chunk_size)
            if not chunk:
                break
            yield chunk


class _FakePaginator:
    def __init__(self, pages: list[dict[str, Any]]):
        self._pages = pages

    def paginate(self, **_: Any) -> Iterator[dict[str, Any]]:
        # Return the list of page dicts directly
        yield from self._pages


class _FakeS3Client:
    """
    Minimal fake client with the methods our s3.py calls.
    - get_paginator("list_objects_v2")
    - head_object
    - get_object
    - download_file
    """

    def __init__(
        self,
        pages: list[dict[str, Any]],
        data_by_key: dict[str, bytes] | None = None,
        fail_on_download: set[str] | None = None,
        etag_by_key: dict[str, str] | None = None,
    ):
        self._pages = pages
        self._data = data_by_key or {}
        self._fail = fail_on_download or set()
        self._etag = etag_by_key or {}

    def get_paginator(self, name: str) -> _FakePaginator:
        assert name == "list_objects_v2"
        return _FakePaginator(self._pages)

    # Head: return ContentLength & ETag (with quotes like AWS does)
    def head_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        data = self._data.get(Key, b"")
        etag = self._etag.get(Key, "dummyetag")
        return {"ContentLength": len(data), "ETag": f'"{etag}"'}

    # Get: return a StreamingBody-like object
    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        data = self._data.get(Key, b"")
        return {"Body": _FakeStreamingBody(data)}

    # Download: write to path, allow forced failures for certain keys
    def download_file(self, Bucket: str, Key: str, Filename: str, Config=None) -> None:
        if Key in self._fail:
            raise RuntimeError("forced download error")
        p = Path(Filename)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self._data.get(Key, b""))


@pytest.fixture(autouse=True)
def clear_client_cache():
    # _s3_client is lru_cached; clear between tests
    s3.get_s3_client.cache_clear()
    yield
    s3.get_s3_client.cache_clear()


def test_list_s3_objects_filters_and_strips_etag(monkeypatch):
    pages = [
        {
            "Contents": [
                {"Key": "a/file1.bin", "Size": 3, "ETag": '"abc"'},
                {"Key": "a/dir/", "Size": 0, "ETag": '"dir"'},
                {"Key": "a/file2.txt", "Size": 5, "ETag": '"def"'},
            ]
        }
    ]
    fake = _FakeS3Client(pages)
    monkeypatch.setattr(s3, "get_s3_client", lambda: fake)

    # no extension filter:
    objs = s3.list_s3_objects("buck", "a/")
    keys = [o["key"] for o in objs]
    assert keys == ["a/file1.bin", "a/file2.txt"]  # "a/dir/" is skipped
    assert objs[0]["etag"] == "abc"

    # with extension filter:
    objs2 = s3.list_s3_objects("buck", "a/", extension=".bin")
    assert [o["key"] for o in objs2] == ["a/file1.bin"]


def test_estimate_s3_size(monkeypatch):
    pages = [{"Contents": [{"Key": "x/1.bin", "Size": 2}, {"Key": "x/2.bin", "Size": 5}]}]
    fake = _FakeS3Client(pages)
    monkeypatch.setattr(s3, "get_s3_client", lambda: fake)

    total, count = s3.estimate_s3_size("buck", "x/")
    assert total == 7
    assert count == 2


def test_fetch_s3_file_writes_and_preserves_subpath(monkeypatch, tmp_path: Path):
    data = b"hello"
    pages = [{"Contents": [{"Key": "p/f.bin", "Size": len(data)}]}]
    fake = _FakeS3Client(pages, data_by_key={"p/f.bin": data})
    monkeypatch.setattr(s3, "get_s3_client", lambda: fake)

    out = s3.fetch_s3_file("buck", "p/f.bin", tmp_path)
    assert out == tmp_path / "p" / "f.bin"
    assert out.read_bytes() == data


def test_iter_s3_object_chunks_yields_all_bytes(monkeypatch):
    data = b"abcdefghij"
    pages = [{"Contents": [{"Key": "k.dat", "Size": len(data)}]}]
    fake = _FakeS3Client(pages, data_by_key={"k.dat": data})
    monkeypatch.setattr(s3, "get_s3_client", lambda: fake)

    got = b"".join(s3.iter_s3_object_chunks("buck", "k.dat", chunk_size=3))
    assert got == data


def test_head_s3_object_returns_size_and_stripped_etag(monkeypatch):
    data = b"xyz"
    pages = [{"Contents": [{"Key": "h.bin", "Size": len(data)}]}]
    fake = _FakeS3Client(pages, data_by_key={"h.bin": data}, etag_by_key={"h.bin": "abc123"})
    monkeypatch.setattr(s3, "get_s3_client", lambda: fake)

    size, etag = s3.head_s3_object("buck", "h.bin")
    assert size == 3
    assert etag == "abc123"  # quotes stripped


def test_fetch_s3_prefix_downloads_all_and_fixes_dir_marker(monkeypatch, tmp_path: Path):
    # Two files under x/, and a folder marker in listing:
    pages = [
        {
            "Contents": [
                {"Key": "x/", "Size": 0},
                {"Key": "x/one.bin", "Size": 3},
                {"Key": "x/two.bin", "Size": 4},
            ]
        }
    ]
    data = {"x/one.bin": b"111", "x/two.bin": b"2222"}
    fake = _FakeS3Client(pages, data_by_key=data)
    monkeypatch.setattr(s3, "get_s3_client", lambda: fake)

    # Create a conflicting *file* at the directory path "x" (zero-byte), to check preflight fix:
    conflict = tmp_path / "x"
    conflict.write_bytes(b"")

    rels = s3.fetch_s3_prefix("buck", "x/", tmp_path)
    assert sorted(rels) == ["x/one.bin", "x/two.bin"]
    assert (tmp_path / "x" / "one.bin").read_bytes() == b"111"
    assert (tmp_path / "x" / "two.bin").read_bytes() == b"2222"


def test_fetch_s3_prefix_extension_and_partial_failures(monkeypatch, tmp_path: Path):
    pages = [
        {
            "Contents": [
                {"Key": "p/a.txt", "Size": 1},
                {"Key": "p/b.txt", "Size": 1},
                {"Key": "p/c.bin", "Size": 1},
            ]
        }
    ]
    data = {"p/a.txt": b"A", "p/b.txt": b"B", "p/c.bin": b"C"}

    # Force b.txt to fail:
    failing = {"p/b.txt"}

    fake = _FakeS3Client(pages, data_by_key=data, fail_on_download=failing)
    monkeypatch.setattr(s3, "get_s3_client", lambda: fake)

    # Only .txt should be attempted; one will fail:
    rels = s3.fetch_s3_prefix("buck", "p/", tmp_path, extension=".txt", max_workers=3)

    # current implementation logs failures but does NOT raise:
    assert sorted(rels) in (["p/a.txt"], ["p/a.txt", "p/b.txt"])  # depending on timing, failed may not append
    # Verify successful file is present:
    assert (tmp_path / "p" / "a.txt").exists()


def test__s3_client_uses_unsigned_when_no_creds(monkeypatch):
    # Patch credentials to be empty/blank:
    monkeypatch.setattr(s3, "get_credentials", lambda provider: {})
    called = {}

    def _fake_boto3_client(service, **kwargs):
        called["service"] = service
        called["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(s3, "boto3", type("B3", (), {"client": _fake_boto3_client}))
    s3.get_s3_client.cache_clear()
    _ = s3.get_s3_client()

    assert called["service"] == "s3"
    # No explicit keys passed:
    assert "aws_access_key_id" not in called["kwargs"]
    # Config should have UNSIGNED signature; check attr where possible:
    cfg = called["kwargs"].get("config")
    # The botocore Config keeps signature in .signature_version:
    assert getattr(cfg, "signature_version", None) == s3.UNSIGNED


def test__s3_client_uses_credentialed_when_creds_present(monkeypatch):
    creds = {
        s3.AWSSecrets.AWS_ACCESS_KEY_ID: "AKIA...",
        s3.AWSSecrets.AWS_SECRET_ACCESS_KEY: "SECRET",
        s3.AWSSecrets.AWS_SESSION_TOKEN: "SESSION",
        s3.AWSSecrets.AWS_DEFAULT_REGION: "us-east-1",
        s3.AWSSecrets.AWS_ENDPOINT_URL: "https://minio.local",
    }
    monkeypatch.setattr(s3, "get_credentials", lambda provider: creds)
    called = {}

    def _fake_boto3_client(service, **kwargs):
        called["service"] = service
        called["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(s3, "boto3", type("B3", (), {"client": _fake_boto3_client}))
    s3.get_s3_client.cache_clear()
    _ = s3.get_s3_client()

    assert called["service"] == "s3"
    kw = called["kwargs"]
    assert kw["aws_access_key_id"] == "AKIA..."
    assert kw["aws_session_token"] == "SESSION"
    assert kw["endpoint_url"] == "https://minio.local"
    cfg = kw.get("config")
    # In credentialed path we still pass a Config (retries, pool size):
    assert cfg is not None
