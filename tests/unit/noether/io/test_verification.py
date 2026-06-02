#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

import noether.io.verification as ver


def write(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def test_build_manifest_creates_posix_paths_and_hash(tmp_path: Path) -> None:
    write(tmp_path / "a" / "b" / "x.txt", b"hello")
    write(tmp_path / "root.bin", b"\x00" * 3)

    manifest = ver.build_manifest(tmp_path, jobs=2, include_hash=True)
    # Keys are POSIX style and relative
    assert "a/b/x.txt" in manifest
    assert "root.bin" in manifest
    # Hash present when include_hash=True
    assert manifest["a/b/x.txt"].hash is not None
    assert manifest["root.bin"].hash is not None
    # Size matches
    assert manifest["a/b/x.txt"].size == 5
    assert manifest["root.bin"].size == 3


def test_save_and_load_manifest_roundtrip(tmp_path: Path) -> None:
    write(tmp_path / "f.txt", b"abc")
    man = ver.build_manifest(tmp_path, include_hash=True)
    out = tmp_path / "manifest.json"
    ver.save_manifest(man, out)

    # Load back
    man2 = ver.load_manifest(out)
    assert set(man2.keys()) == set(man.keys())
    assert man2["f.txt"].hash == man["f.txt"].hash
    assert man2["f.txt"].size == man["f.txt"].size


def test_verify_tree_all_ok_with_hash(tmp_path: Path) -> None:
    write(tmp_path / "ok.dat", b"data123")
    man = ver.build_manifest(tmp_path, include_hash=True)

    res = ver.verify_tree(tmp_path, man, jobs=2, require_hash=True)
    assert res.missing == []
    assert res.extra == []
    assert res.size_mismatch == []
    assert res.hash_mismatch == []
    assert res.ok == ["ok.dat"]


def test_verify_tree_missing_and_extra(tmp_path: Path) -> None:
    # Build with one file present
    write(tmp_path / "keep.txt", b"1")
    man = ver.build_manifest(tmp_path, include_hash=False)

    # Now delete keep.txt (becomes missing) and add extra.txt (becomes extra)
    (tmp_path / "keep.txt").unlink()
    write(tmp_path / "extra.txt", b"2")

    res = ver.verify_tree(tmp_path, man, require_hash=False)
    assert res.missing == ["keep.txt"]
    assert res.extra == ["extra.txt"]
    assert res.size_mismatch == []
    assert res.hash_mismatch == []


def test_verify_tree_size_mismatch_detected(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    write(p, b"\x00" * 4)
    man = ver.build_manifest(tmp_path, include_hash=False)

    # Change size only
    write(p, b"\x00" * 5)

    res = ver.verify_tree(tmp_path, man, require_hash=False)
    assert res.size_mismatch == ["f.bin"]
    assert res.hash_mismatch == []


def test_verify_tree_hash_mismatch_detected(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    write(p, b"AAAA")
    man = ver.build_manifest(tmp_path, include_hash=True)

    # Change content, same size
    write(p, b"AAAB")

    res = ver.verify_tree(tmp_path, man, require_hash=True)
    assert res.hash_mismatch == ["f.bin"]
    assert res.size_mismatch == []


def test_verify_tree_require_hash_false_allows_no_hash(tmp_path: Path) -> None:
    # Build without hashes
    write(tmp_path / "a.txt", b"a")
    man = ver.build_manifest(tmp_path, include_hash=False)

    # Verify without requiring hash → OK
    res = ver.verify_tree(tmp_path, man, require_hash=False)
    assert res.hash_mismatch == []
    assert res.size_mismatch == []
    assert res.missing == []
    assert res.extra == []
    assert res.ok == ["a.txt"]


def test_parallel_map_collect_errors_aggregates() -> None:
    items = [1, 2, 3, 4]

    def fn(x: int) -> int:
        if x in (2, 4):
            raise ValueError(f"bad {x}")
        return x * 2

    with pytest.raises(ver.ParallelErrors) as exc:
        ver.parallel_map_collect_errors(fn, items, max_workers=3)

    # Two failures (for 2 and 4), both recorded
    errs = exc.value.errors
    bad_inputs = sorted(x for x, _ in errs)
    assert bad_inputs == [2, 4]
