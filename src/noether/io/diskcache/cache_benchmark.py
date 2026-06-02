#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

# Use the patched filesystem

import functools
import logging
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import fsspec  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
from tqdm import tqdm

from noether.io.diskcache.lru_cache import LRUCacheFileSystem, SqliteLRUCacheFileSystem

USE_OCI = False

_logger = logging.getLogger(__name__)


def underlying_fs(cache_size=0):
    """
    This function returns the underlying filesystem, either S3 or OCI, based on the USE_OCI flag.
    1. If USE_OCI is False, it returns an S3 filesystem pointing to a MinIO server.
    You have to run a local MinIO server for this to work from https://www.min.io/download?platform=linux&arch=amd64
    then run `MINIO_BROWSER=off ./minio server --address ":9123" /mnt/localdisk/benchmark/minio`

    """
    global USE_OCI
    if not USE_OCI:
        return fsspec.filesystem(
            "s3",
            key="minioadmin",
            secret="minioadmin",
            client_kwargs={"endpoint_url": "http://localhost:9123"},
        )
    return fsspec.filesystem("oci", config="~/.oci/config")


def naive_fs(cache_size: int):
    return LRUCacheFileSystem(
        fs=underlying_fs(),
        storage="/mnt/localdisk/benchmark/cache/naive",
        cache_size=cache_size,
    )


def naive_sqlite_fs(cache_size: int):
    return SqliteLRUCacheFileSystem(
        fs=underlying_fs(),
        storage="/mnt/localdisk/benchmark/cache/naive_sqlite",
        cache_size=cache_size,
    )


def local_fs(cache_size: int):
    return fsspec.filesystem("file")


local = threading.local()


def initialize_worker(fs_factory: Callable[[], fsspec.AbstractFileSystem]):
    """
    This is the initializer function for each worker process.
    It creates an instance of HeavyObject and assigns it to a global variable.
    """
    global local
    # local the process-local object
    local.fs = fs_factory()


def read_file(path):
    global local
    start = time.time()
    with local.fs.open(path, "rb") as f:
        data = f.read()
    return len(data), time.time() - start


import random
import shutil


def benchmark(
    fs_factory: Callable[[int], fsspec.AbstractFileSystem],
    name: str,
    cache_size: int,
    num_workers,
    prefix: str,
    max_files=100,
    warmup_fraction=0.0,
    executor=ThreadPoolExecutor,
):
    fs = fs_factory(cache_size)

    # Clean up the cache directory before each benchmark
    storage = getattr(fs, "storage", None)
    storage = storage[-1] if isinstance(storage, list) else storage
    if storage:
        _logger.info(f"Removing cache directory: {storage}")
        shutil.rmtree(storage)
        os.makedirs(storage, exist_ok=True)

    files: list[str] = fs.glob(prefix + "*")[:max_files]

    if not files:
        _logger.info(f"No files found with prefix: {prefix}")
        return {}

    latencies = []
    total_bytes = 0
    pool = executor(
        max_workers=num_workers,
        initializer=initialize_worker,
        initargs=(functools.partial(fs_factory, cache_size),),
    )

    with pool as executor:
        warmup_files = random.sample(files, int(len(files) * warmup_fraction))

        for _ in tqdm(
            as_completed({executor.submit(read_file, file): file for file in warmup_files}),
            total=len(warmup_files),
            desc="Warming up cache",
        ):
            pass

        start = time.time()

        futures = {executor.submit(read_file, file): file for file in files}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Benchmarking {name} filesystem",
        ):
            nbytes, latency = future.result()
            latencies.append(latency)
            total_bytes += nbytes
    end = time.time()

    total_time = end - start
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    throughput = total_bytes / total_time if total_time > 0 else 0
    _logger.info(f"Benchmark results for {name} filesystem:")
    _logger.info(f"Workers: {num_workers}, Files: {len(files)}")
    _logger.info(f"Total bytes: {total_bytes}, Total time: {total_time:.2f}s")
    _logger.info(f"Avg latency per file: {avg_latency:.4f}s")
    _logger.info(f"Throughput: {throughput / 1024 / 1024:.2f} MB/s")
    return {
        "name": name,
        "workers": num_workers,
        "files": len(files),
        "total_bytes": total_bytes,
        "total_time": total_time,
        "avg_latency": avg_latency,
        "throughput": throughput,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    impls = {
        "naive_sqlite": naive_sqlite_fs,
        "naive": naive_fs,
        "underlying": underlying_fs,
        "local": local_fs,
        # "nfs": local_fs,
    }
    hit_rates = [0.5, 0.95, 0.99, 1.0]

    results = []

    """
    Model different scenarios:
    - cache can hold 0.25x the working set
    - cache can hold 1x the working set
    - cache can hold 2x the working set
    """
    relative_cache_capacities = [0.5, 1.0]
    NUM_WORKERS = 32

    for name, fs in impls.items():
        prefix = "benchmark@frwnorq7ern2/1MB" if USE_OCI else "benchmark/1MB/"
        if name == "local":
            prefix = "/mnt/localdisk/benchmark/testdata/1MB/"
        if name == "nfs":
            prefix = "/nfs-gpu/tmp/benchmark/testdata/1MB/"

        for executor in [ProcessPoolExecutor, ThreadPoolExecutor]:
            for cache_capacity in relative_cache_capacities:
                for intended_hit_rate in hit_rates:
                    if cache_capacity < intended_hit_rate:
                        continue  # skip impossible scenarios

                    FILE_SIZE = 1024**2
                    N_FILES = 5_000 if USE_OCI else 10_000

                    cache_files = int(N_FILES / cache_capacity) + 1
                    cache_size = int(cache_files * FILE_SIZE)

                    random.seed(42)
                    res = benchmark(
                        fs,  # type: ignore[arg-type]
                        cache_size=cache_size,
                        name=name,
                        num_workers=NUM_WORKERS,
                        prefix=prefix,
                        max_files=N_FILES,
                        warmup_fraction=0 if name == "underlying" else intended_hit_rate,
                        executor=executor,
                    )
                    res["intended_hit_rate"] = intended_hit_rate
                    res["cache_fill_rate"] = min(1.0, cache_capacity)
                    res["executor"] = "process" if executor == ProcessPoolExecutor else "thread"
                    results.append(res)
                    if name in ["underlying", "local", "nfs"]:
                        break  # only run once for underlying fs
                if name in ["underlying", "local", "nfs"]:
                    break  # only run once for underlying fs
            if name in ["underlying", "local", "nfs"]:
                break  # only run once for underlying fs

    df = pd.DataFrame(results)

    num = 32
    out_file = f"benchmark_results_oci_{num}.csv" if USE_OCI else f"benchmark_results_{num}.csv"
    df.to_csv(out_file, index=False, mode="a", header=not os.path.exists(out_file))
