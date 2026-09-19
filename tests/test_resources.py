import os

import pandas as pd

from app.engine import resources
from app.engine.resources import limit_worker_threads, max_parallel_workers


def make_df(rows: int = 1000) -> pd.DataFrame:
    return pd.DataFrame({"open": range(rows), "high": range(rows), "low": range(rows), "close": range(rows)})


def test_worker_count_is_capped_by_available_memory(monkeypatch):
    # 1 GB free, 60% headroom, ~250 MB per worker -> 2 workers, not 12.
    monkeypatch.setattr(resources, "available_memory_bytes", lambda: 1024 * 1024 * 1024)
    assert max_parallel_workers(12, make_df()) == 2


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
