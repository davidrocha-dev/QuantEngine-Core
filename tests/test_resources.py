import os

import pandas as pd

from app.engine import resources
from app.engine.resources import limit_worker_threads, max_parallel_workers


def make_df(rows: int = 1000) -> pd.DataFrame:
    return pd.DataFrame({"open": range(rows), "high": range(rows), "low": range(rows), "close": range(rows)})


GB = 1024 * 1024 * 1024


def test_worker_count_is_capped_by_available_memory(monkeypatch):
    # 4 GB available, 1 GB held back for the parent and the OS, 60% of the
    # remaining 3 GB at ~250 MB a worker -> 7, not 12.
    monkeypatch.setattr(resources, "available_memory_bytes", lambda: 4 * GB)
    assert max_parallel_workers(12, make_df()) == 7


def test_the_host_keeps_its_reserve(monkeypatch):
    """The parent grows to roughly 450 MB over a campaign - it holds every
    candidate's stats, the equity curves and the checkpoint - and the
    machine has to keep running too. A fraction of a large number still
    leaves nothing for either when the real constraint is elsewhere."""
    monkeypatch.setattr(resources, "available_memory_bytes", lambda: resources.HOST_RESERVE_BYTES)
    assert max_parallel_workers(12, make_df()) == 1


def test_worker_count_never_drops_below_one(monkeypatch):
    monkeypatch.setattr(resources, "available_memory_bytes", lambda: 1024 * 1024)  # 1 MB free
    assert max_parallel_workers(12, make_df()) == 1


def test_worker_count_respects_the_requested_ceiling(monkeypatch):
    monkeypatch.setattr(resources, "available_memory_bytes", lambda: 64 * 1024 * 1024 * 1024)
    assert max_parallel_workers(4, make_df()) == 4


def test_falls_back_to_the_requested_count_when_memory_is_unknown(monkeypatch):
    monkeypatch.setattr(resources, "available_memory_bytes", lambda: None)
    assert max_parallel_workers(8, make_df()) == 8


def test_a_large_dataframe_shrinks_the_worker_count(monkeypatch):
    monkeypatch.setattr(resources, "available_memory_bytes", lambda: 4 * 1024 * 1024 * 1024)
    small = max_parallel_workers(12, make_df(1_000))
    large = max_parallel_workers(12, make_df(5_000_000))
    assert large < small


def test_limit_worker_threads_pins_numeric_backends(monkeypatch):
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    limit_worker_threads()
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_windows_reads_the_commit_limit_not_just_free_ram(monkeypatch):
    """A process is bounded by the system commit limit — RAM plus page
    file — and Windows refuses the allocation when the commit charge is
    exhausted however much physical memory is free.

    Observed on the owner's machine: 6.44 GB of free RAM against 4.29 GB
    of commit left, the guard sizing a pool against the first number, and
    workers dying inside `import socket` with MemoryError before they
    reached any of our code.
    """
    import ctypes
    import sys

    if sys.platform != "win32":
        return

    captured = {}

    class FakeKernel32:
        @staticmethod
        def GlobalMemoryStatusEx(pointer):
            status = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_ulonglong))
            # Layout: dwLength, dwMemoryLoad (2 x c_ulong = 1 x c_ulonglong),
            # then ullTotalPhys, ullAvailPhys, ullTotalPageFile, ullAvailPageFile.
            status[1] = 17 * 1024**3  # total phys
            status[2] = 6 * 1024**3  # avail phys — the roomy number
            status[3] = 20 * 1024**3  # total page file
            status[4] = 2 * 1024**3  # avail page file — the binding one
            captured["called"] = True
            return 1

    monkeypatch.setattr(ctypes, "windll", type("W", (), {"kernel32": FakeKernel32})())

    available = resources.available_memory_bytes()

    assert captured.get("called")
    assert available == 2 * 1024**3, "the smaller of the two has to win"
