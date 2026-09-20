from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from app.config import EngineConfig
from app.data.cache import LocalCache
from app.data.dukascopy_client import TickHour
from app.data.sync import DeltaSyncRunner


class FakeClient:
    """Deterministic stand-in for DukascopyClient — avoids real network
    calls, which are unreachable from this sandbox."""

    def __init__(self):
        self.calls: list[datetime] = []

    async def fetch_hour(self, symbol: str, hour_start: datetime) -> TickHour:
        self.calls.append(hour_start)
        ticks = pd.DataFrame(
            {
                "timestamp": [hour_start],
                "bid": [1.0800],
                "ask": [1.0802],
                "bid_volume": [1.0],
                "ask_volume": [1.0],
            }
        )
        return TickHour(symbol, hour_start, ticks)


def make_cache(tmp_path: Path) -> LocalCache:
    config = EngineConfig(data_dir=tmp_path / "data", bundled_data_dir=tmp_path / "bundled")
    return LocalCache(config)


@pytest.mark.asyncio
async def test_sync_symbol_fetches_only_missing_hours(tmp_path):
    cache = make_cache(tmp_path)
    client = FakeClient()
    runner = DeltaSyncRunner(cache=cache, client=client)

    now = datetime.now(timezone.utc)
    last_synced = (now - timedelta(hours=4)).replace(minute=0, second=0, microsecond=0)
    cache.set_last_synced_hour("EURUSD", last_synced)

    hours_synced = await runner.sync_symbol("EURUSD")

    assert hours_synced == len(client.calls)
    assert all(call > last_synced for call in client.calls)
    assert cache.get_last_synced_hour("EURUSD") is not None


def test_sync_start_is_bounded_by_configured_history_depth(tmp_path):
    # A first sync must not walk back to the 2003 epoch — that is ~190,000
    # hourly files per symbol. It starts `history_years` back instead.
    cache = make_cache(tmp_path)
    runner = DeltaSyncRunner(cache=cache, client=FakeClient(), history_years=3)

    start = runner._sync_start("EURUSD")

    expected = datetime.now(timezone.utc) - timedelta(days=365 * 3)
    assert abs((start - expected).total_seconds()) < 60
    assert start > datetime(2003, 5, 1, tzinfo=timezone.utc)


def test_sync_start_never_precedes_the_dukascopy_epoch(tmp_path):
    cache = make_cache(tmp_path)
    runner = DeltaSyncRunner(cache=cache, client=FakeClient(), history_years=100)

    assert runner._sync_start("EURUSD") == datetime(2003, 5, 1, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_sync_all_isolates_failures_per_symbol(tmp_path):
    cache = make_cache(tmp_path)

    class FlakyClient(FakeClient):
        async def fetch_hour(self, symbol: str, hour_start: datetime) -> TickHour:
            if symbol == "GBPUSD":
                raise RuntimeError("simulated network failure")
            return await super().fetch_hour(symbol, hour_start)

    now = datetime.now(timezone.utc)
    recent = (now - timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    cache.set_last_synced_hour("EURUSD", recent)
    cache.set_last_synced_hour("GBPUSD", recent)

    runner = DeltaSyncRunner(cache=cache, client=FlakyClient())
    results = await runner.sync_all(["EURUSD", "GBPUSD"])

    assert results["GBPUSD"] == 0
    assert "EURUSD" in results


class SlowCache(LocalCache):
    """A cache whose writes take as long as a real one on a large file."""

    WRITE_SECONDS = 0.15

    def append_bars(self, symbol: str, timeframe: str, bars: pd.DataFrame) -> None:
        import time

        time.sleep(self.WRITE_SECONDS)
        super().append_bars(symbol, timeframe, bars)


@pytest.mark.asyncio
async def test_a_running_sync_does_not_freeze_the_rest_of_the_engine(tmp_path):
    """The sync's heavy work - tick decoding and rewriting the parquet
    cache - has to happen off the event loop.

    When it ran on the loop, every HTTP request the app made during a
    sync waited behind it: opening the app on a machine with months of
    catching up to do looked like a frozen window, because /status could
    not answer for a minute at a time. This test does not inspect where
    the work runs; it asks the only question that matters, which is
    whether anything else still gets a turn while a sync is going.
    """
    import asyncio

    cache = SlowCache(EngineConfig(data_dir=tmp_path / "data", bundled_data_dir=tmp_path / "bundled"))
    runner = DeltaSyncRunner(cache=cache, client=FakeClient())
    now = datetime.now(timezone.utc)
    cache.set_last_synced_hour("EURUSD", (now - timedelta(hours=8)).replace(minute=0, second=0, microsecond=0))

    heartbeats = 0

    async def pretend_to_be_the_app():
        nonlocal heartbeats
        while True:
            await asyncio.sleep(0.01)
            heartbeats += 1

    poller = asyncio.create_task(pretend_to_be_the_app())
    await runner.sync_symbol("EURUSD")
    poller.cancel()

    # At least one write happened, so the loop had something to be blocked
    # by; it answered many times anyway.
    assert heartbeats > 5
