#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import os
import tempfile
import time

import filelock
import pytest
from fsspec import AbstractFileSystem

from noether.io.diskcache.lru_cache import LOCK_FILE_SUFFIX, LRUCacheFileSystem


class DummyFS(AbstractFileSystem):
    protocol = "dummy"

    def __init__(self):
        self.fs_attr = "fs_value"
        self.get_file_calls = []

    @classmethod
    def _strip_protocol(cls, path):
        return path

    def get_file(self, rpath, lpath, callback=None, outfile=None, **kwargs):
        self.get_file_calls.append((rpath, lpath))
        with open(lpath, "wb") as f:
            f.write(rpath.encode())


@pytest.fixture
def lru_cache_fs():
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = DummyFS()
        cache = LRUCacheFileSystem(fs=fs, storage=tmpdir, cache_size=1024)
        yield cache


def test_basic_caching(lru_cache_fs):
    """Test basic file caching behavior"""
    # First access should cache the file
    path = "/test/file.txt"
    with lru_cache_fs.open(path, "rb") as f:
        content = f.read()

    # Check content is correct
    assert content == b"/test/file.txt"

    # Check that underlying fs was called once
    assert len(lru_cache_fs.fs.get_file_calls) == 1

    # Second access should use cache
    with lru_cache_fs.open(path, "rb") as f:
        cached_content = f.read()

    # Content should be the same
    assert cached_content == content

    # Check that underlying fs wasn't called again
    assert len(lru_cache_fs.fs.get_file_calls) == 1


def test_cache_eviction(lru_cache_fs):
    """Test that files are evicted when cache exceeds size limit"""
    # Set small cache size for testing
    lru_cache_fs.cache_storage_size = 100

    # Create files that will exceed the cache size
    files = [f"/test/file{i}.txt" for i in range(10)]

    # Access files in order
    for path in files:
        with lru_cache_fs.open(path, "rb") as f:
            f.read()

    # Force cache cleanup
    lru_cache_fs._last_enforce_time = 0
    lru_cache_fs._enforce_cache_size()

    # Check that early files are evicted (no longer in cache)
    # We can't check precisely which files are gone due to file system specifics
    # but we can check that some files were removed
    cached_files = sum(
        1 for entry in os.scandir(lru_cache_fs.storage) if entry.is_file() and not entry.name.endswith(LOCK_FILE_SUFFIX)
    )
    assert cached_files < 10


def test_update_access_time(lru_cache_fs):
    """Test that access times are updated"""
    # Access a file to cache it
    path = "/test/access_time.txt"
    with lru_cache_fs.open(path, "rb") as f:
        f.read()

    # Get cache file path
    sha = lru_cache_fs._mapper(path)
    fn = os.path.join(lru_cache_fs.storage, sha)

    # Get initial mtime
    initial_mtime = os.path.getmtime(fn)

    # Sleep to ensure mtime can change
    time.sleep(0.05)

    # Access file again
    with lru_cache_fs.open(path, "rb") as f:
        f.read()

    # Check that mtime was updated
    new_mtime = os.path.getmtime(fn)
    assert new_mtime > initial_mtime


def test_write_not_supported(lru_cache_fs):
    """Test that writes are not supported"""
    with pytest.raises(NotImplementedError):
        lru_cache_fs.open("/test/file.txt", "wb")


def test_cache_cleanup_lock(lru_cache_fs):
    """Test that cache cleanup is protected by a lock"""
    # Create a file to simulate another process holding the lock
    lock_file = os.path.join(lru_cache_fs.storage, ".cleanup_lock")
    with filelock.FileLock(lock_file):
        # Set small cache size to trigger cleanup
        lru_cache_fs.cache_storage_size = 10
        lru_cache_fs._last_enforce_time = 0

        # This should not raise an exception, just skip cleanup
        lru_cache_fs._enforce_cache_size()


def test_cat_ranges(lru_cache_fs):
    """Test cat_ranges method"""
    # Prepare test files
    paths = ["/1.txt", "/2.csv"]
    for path in paths:
        with lru_cache_fs.open(path, "rb"):
            pass  # Just cache the files

    # Assert files exist in cache directory
    for path in paths:
        sha = lru_cache_fs._mapper(path)
        cache_path = os.path.join(lru_cache_fs.storage, sha)
        assert os.path.exists(cache_path), f"Cache file for {path} does not exist"

    # Test cat_ranges
    starts = [0, 2]
    ends = [2, 4]
    result = lru_cache_fs.cat_ranges(paths, starts, ends)

    # Each result should be a slice of the path name bytes
    for start, end, path, res in zip(starts, ends, paths, result, strict=False):
        expected = path.encode()[start:end]
        assert res == expected, f"Expected {expected} but got {res}"


def test_cache_cleanup_watermarks(lru_cache_fs):
    """Test cache cleanup respects high and low watermarks"""
    # Set cache size and watermarks
    lru_cache_fs.cache_storage_size = 1000
    lru_cache_fs.cache_cleanup_high_watermark = 0.8
    lru_cache_fs.cache_cleanup_low_watermark = 0.6
    lru_cache_fs._last_enforce_time = 0

    # Create files to fill cache beyond high watermark
    # Each file is approximately 100 bytes
    for i in range(10):
        path = f"/test/watermark_{i}.txt"
        with lru_cache_fs.open(path, "rb") as f:
            f.read()
        time.sleep(0.01)  # Ensure different mtimes

    # Force cleanup
    lru_cache_fs._enforce_cache_size()

    # Calculate actual cache size
    total_size = sum(
        os.path.getsize(os.path.join(lru_cache_fs.storage, entry.name))
        for entry in os.scandir(lru_cache_fs.storage)
        if entry.is_file() and not entry.name.endswith(LOCK_FILE_SUFFIX)
    )

    # Cache should be cleaned to below high watermark
    assert total_size <= lru_cache_fs.cache_storage_size * lru_cache_fs.cache_cleanup_high_watermark


def test_attribute_delegation():
    """Test that attributes are properly delegated to underlying filesystem"""
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = DummyFS()
        cache = LRUCacheFileSystem(fs=fs, storage=tmpdir, cache_size=1024)

        # Should access the underlying fs's attribute
        assert cache.fs_attr == "fs_value"


def test_cache_size_none_no_eviction():
    """Test that cache_size=None means no eviction"""
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = DummyFS()
        cache = LRUCacheFileSystem(fs=fs, storage=tmpdir, cache_size=None)

        # Create many files
        for i in range(20):
            path = f"/test/file_{i}.txt"
            with cache.open(path, "rb") as f:
                f.read()

        # Force cleanup attempt
        cache._last_enforce_time = 0
        cache._enforce_cache_size()

        # All files should still be present
        cached_files = sum(
            1 for entry in os.scandir(cache.storage) if entry.is_file() and not entry.name.endswith(LOCK_FILE_SUFFIX)
        )
        assert cached_files == 20


def test_enforce_size_throttling(lru_cache_fs):
    """Test that cache size enforcement is throttled"""
    lru_cache_fs.enforce_size_every_seconds = 10
    lru_cache_fs._last_enforce_time = time.time()

    initial_time = lru_cache_fs._last_enforce_time

    # Access a file
    with lru_cache_fs.open("/test/throttle.txt", "rb") as f:
        f.read()

    # Last enforce time should not change (cleanup was skipped)
    assert lru_cache_fs._last_enforce_time == initial_time


def test_missing_cached_file_handling(lru_cache_fs):
    """Test handling when cached file is deleted externally"""
    path = "/test/deleted.txt"

    # Cache the file
    with lru_cache_fs.open(path, "rb") as f:
        f.read()

    # Manually delete the cached file
    sha = lru_cache_fs._mapper(path)
    fn = os.path.join(lru_cache_fs.storage, sha)
    os.remove(fn)

    # Should re-download the file
    initial_calls = len(lru_cache_fs.fs.get_file_calls)
    with lru_cache_fs.open(path, "rb") as f:
        content = f.read()

    assert content == path.encode()
    assert len(lru_cache_fs.fs.get_file_calls) == initial_calls + 1


def test_cat_ranges_mixed_cached_uncached():
    """Test cat_ranges with mix of cached and uncached files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = DummyFS()
        cache = LRUCacheFileSystem(fs=fs, storage=tmpdir, cache_size=1024)

        # Cache only the first file
        with cache.open("/cached.txt", "rb"):
            pass

        # Request ranges for both cached and uncached files
        paths = ["/cached.txt", "/uncached.txt"]
        starts = [0, 1]
        ends = [3, 5]

        result = cache.cat_ranges(paths, starts, ends)

        assert result[0] == b"/ca"
        assert result[1] == b"unca"


def test_file_permissions(lru_cache_fs):
    """Test that cached files have correct permissions"""
    path = "/test/permissions.txt"

    with lru_cache_fs.open(path, "rb") as f:
        f.read()

    sha = lru_cache_fs._mapper(path)
    fn = os.path.join(lru_cache_fs.storage, sha)

    # Check file permissions
    stat_info = os.stat(fn)
    mode = stat_info.st_mode & 0o777
    assert mode == lru_cache_fs.cache_storage_mode


def test_cleanup_with_locked_files(lru_cache_fs):
    """Test that cleanup skips files that are locked"""
    lru_cache_fs.cache_storage_size = 100
    lru_cache_fs._last_enforce_time = 0

    # Create and cache a file
    path = "/test/locked.txt"
    with lru_cache_fs.open(path, "rb") as f:
        f.read()

    sha = lru_cache_fs._mapper(path)
    fn = os.path.join(lru_cache_fs.storage, sha)

    # Lock the file
    lock = filelock.FileLock(fn + LOCK_FILE_SUFFIX)
    lock.acquire()

    try:
        # Create more files to trigger cleanup
        for i in range(10):
            with lru_cache_fs.open(f"/test/file_{i}.txt", "rb") as f:
                f.read()

        # The locked file should still exist
        assert os.path.exists(fn)
    finally:
        lock.release()


def test_double_check_file_creation(lru_cache_fs):
    """Test that file is not downloaded twice if created by another process"""
    path = "/test/double_check.txt"
    sha = lru_cache_fs._mapper(path)
    fn = os.path.join(lru_cache_fs.storage, sha)

    original_get_file = lru_cache_fs.fs.get_file
    call_count = 0

    def mock_get_file(rpath, lpath, **kwargs):
        nonlocal call_count
        call_count += 1
        # Simulate another process creating the file
        if call_count == 1:
            with open(lpath, "wb") as f:
                f.write(b"already_exists")
        else:
            original_get_file(rpath, lpath, **kwargs)

    lru_cache_fs.fs.get_file = mock_get_file

    with lru_cache_fs.open(path, "rb") as f:
        content = f.read()

    # Should use the file created by "another process"
    assert content == b"already_exists"
    assert call_count == 1
