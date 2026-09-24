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

# Measured peak for a worker that has imported numpy + pandas + numba and
# evaluated genomes over a 40,000-bar series: 153 MB, sampled through the
# spawn as well as the steady state. Kept above that, because the spawn is
# where it fails and a worker that cannot start takes the campaign with it.
WORKER_BASE_MEMORY_BYTES = 250 * 1024 * 1024
# Never commit more than this share of what is left to workers.
MEMORY_HEADROOM_FRACTION = 0.6
# Held back before any of that arithmetic, for the parent process and the
# operating system. The parent grows to roughly 450 MB over a campaign -
# it holds every candidate's stats, the equity curves and the checkpoint -
# and a fraction of a large number still leaves nothing for it when the
# real constraint is somewhere else.
HOST_RESERVE_BYTES = 1024 * 1024 * 1024


def available_memory_bytes() -> int | None:
    """Memory a new process can actually get, or None when it cannot be
    determined (in which case callers fall back to the core count alone).

    On Windows that is not free physical RAM. A process is bounded by the
    system commit limit - RAM plus page file - and Windows refuses the
    allocation when the commit charge is exhausted however much physical
    memory is free. Observed on the owner's machine: 6.44 GB of free RAM
    and 4.29 GB of commit left, the guard sizing a pool against the first
    number, and workers dying inside `import socket` with MemoryError
    before they reached any of this code. The binding constraint is
    whichever is smaller.
    """
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
        return min(int(status.ullAvailPhys), int(status.ullAvailPageFile))

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

    # The reserve comes off the top: the parent has to keep running, and
    # the machine has to keep running too.
    spendable = max(0, available - HOST_RESERVE_BYTES)
    per_worker = WORKER_BASE_MEMORY_BYTES + (dataframe_memory_bytes(df) if df is not None else 0)
    affordable = int(spendable * MEMORY_HEADROOM_FRACTION // per_worker)
    return max(1, min(workers, affordable))


def limit_worker_threads() -> None:
    """Pin every numeric backend to a single thread inside a worker.

    Each worker is already one of N parallel processes; letting BLAS spawn
    its own thread pool per process oversubscribes the CPU and multiplies
    memory use for no gain.
    """
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
