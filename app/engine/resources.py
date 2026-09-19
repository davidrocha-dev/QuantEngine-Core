"""Host resource limits for the process pool.

Parallel evaluation spawns one Python process per worker, and each one
loads numpy/pandas/numba and holds its own copy of the bar data. On a
machine that is already under memory pressure, `os.cpu_count()` workers is
how you turn a campaign into `MemoryError` / `OpenBLAS: Memory allocation
failed` — observed in practice on a 17 GB machine with 1.4 GB free.

So the worker count is capped by *available* memory as well as by cores.
"""

from __future__ import annotations

import ctypes
import os
import sys

import pandas as pd

# Measured floor for a worker that has imported numpy + pandas + numba.
WORKER_BASE_MEMORY_BYTES = 250 * 1024 * 1024
# Never commit more than this share of free memory to workers.
MEMORY_HEADROOM_FRACTION = 0.6


def available_memory_bytes() -> int | None:
    """Free physical memory, or None when it cannot be determined (in which
    case callers fall back to the core count alone)."""
    if sys.platform == "win32":
        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullAvailPhys)

    try:  # Linux / macOS
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def dataframe_memory_bytes(df: pd.DataFrame) -> int:
    try:
        return int(df.memory_usage(deep=True).sum())
    except Exception:
        return 0


def max_parallel_workers(requested: int, df: pd.DataFrame | None = None) -> int:
    """Worker count that fits both the core budget and free memory."""
    workers = max(1, requested)
    available = available_memory_bytes()
    if available is None:
        return workers

    per_worker = WORKER_BASE_MEMORY_BYTES + (dataframe_memory_bytes(df) if df is not None else 0)
    affordable = int(available * MEMORY_HEADROOM_FRACTION // per_worker)
    return max(1, min(workers, affordable))


def limit_worker_threads() -> None:
    """Pin every numeric backend to a single thread inside a worker.

    Each worker is already one of N parallel processes; letting BLAS spawn
    its own thread pool per process oversubscribes the CPU and multiplies
    memory use for no gain.
    """
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
