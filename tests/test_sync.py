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


# --- Backfill -----------------------------------------------------------
#
# Raising HISTORY DEPTH after a symbol had been synced used to do nothing:
# sync resumed from the forward watermark and only ever walked towards now.
# The setting said three years, the cache held one, and the campaign
# trained on the one without saying so.


def seed_bars(cache: LocalCache, symbol: str, first: datetime, hours: int) -> None:
    stamps = [first + timedelta(hours=i) for i in range(hours)]
    cache.append_bars(
        symbol,
        "M1",
        pd.DataFrame(
            {
                "timestamp": stamps,
                "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
            }
        ),
    )


def test_backfill_asks_for_the_years_the_cache_is_missing(tmp_path):
    cache = make_cache(tmp_path)
    runner = DeltaSyncRunner(cache=cache, client=FakeClient(), history_years=3)

    one_year_ago = (datetime.now(timezone.utc) - timedelta(days=365)).replace(minute=0, second=0, microsecond=0)
    cache.set_first_synced_hour("XAUUSD", one_year_ago)

    hours = runner._backfill_hours("XAUUSD", "M1")

    assert hours, "raising the depth must ask for the older years"
    assert max(hours) < one_year_ago
    # Three years back, minus the year already held.
    assert 700 * 24 < len(hours) < 740 * 24


def test_no_backfill_when_the_cache_already_reaches_the_configured_depth(tmp_path):
    cache = make_cache(tmp_path)
    runner = DeltaSyncRunner(cache=cache, client=FakeClient(), history_years=1)

    four_years_ago = (datetime.now(timezone.utc) - timedelta(days=4 * 365)).replace(minute=0, second=0, microsecond=0)
    cache.set_first_synced_hour("XAUUSD", four_years_ago)

    assert runner._backfill_hours("XAUUSD", "M1") == []


def test_backfill_falls_back_to_the_bars_on_disk(tmp_path):
    # A manifest written before `first` existed - which is every cache the
    # released build produced.
    cache = make_cache(tmp_path)
    runner = DeltaSyncRunner(cache=cache, client=FakeClient(), history_years=3)

    start = (datetime.now(timezone.utc) - timedelta(days=200)).replace(minute=0, second=0, microsecond=0)
    seed_bars(cache, "XAUUSD", start, 48)
    cache.set_last_synced_hour("XAUUSD", datetime.now(timezone.utc) - timedelta(hours=1))
    assert cache.get_first_synced_hour("XAUUSD") is None

    hours = runner._backfill_hours("XAUUSD", "M1")

    assert hours and max(hours) < start


def test_legacy_string_manifest_still_reads(tmp_path):
    cache = make_cache(tmp_path)
    hour = datetime(2026, 1, 2, 3, tzinfo=timezone.utc)
    cache.write_manifest({"EURUSD": hour.isoformat()})

    assert cache.get_last_synced_hour("EURUSD") == hour
    assert cache.get_first_synced_hour("EURUSD") is None


def test_first_synced_hour_only_ever_retreats(tmp_path):
    cache = make_cache(tmp_path)
    early = datetime(2024, 1, 1, tzinfo=timezone.utc)
    cache.set_first_synced_hour("EURUSD", early)
    cache.set_first_synced_hour("EURUSD", datetime(2025, 1, 1, tzinfo=timezone.utc))

    assert cache.get_first_synced_hour("EURUSD") == early


@pytest.mark.asyncio
async def test_backfill_commits_only_whole_blocks(tmp_path):
    """A hole in the middle of the cached range is undetectable downstream:
    read_bars returns a frame, the backtester resamples it, and the missing
    weeks simply are not there."""
    cache = make_cache(tmp_path)
    runner = DeltaSyncRunner(cache=cache, client=FakeClient())

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    hours = [start + timedelta(hours=i) for i in range(10)]
    progress = __import__("app.data.sync", fromlist=["SyncProgress"]).SyncProgress(symbol="XAUUSD")

    calls: list[datetime] = []

    async def fetch(hour):
        calls.append(hour)
        if hour == hours[3]:
            raise RuntimeError("feed threw")
        return pd.DataFrame(
            {"timestamp": [hour], "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}
        )

    import app.data.sync as sync_module

    original = sync_module.RETRY_BACKOFF_SECONDS
    sync_module.RETRY_BACKOFF_SECONDS = 0
    try:
        synced = await runner._backfill("XAUUSD", "M1", hours, progress, fetch)
    finally:
        sync_module.RETRY_BACKOFF_SECONDS = original

    # The whole run is one block, and one hour in it fails, so nothing is
    # committed rather than nine tenths of a range claiming to be whole.
    assert synced == 0
    assert cache.get_first_synced_hour("XAUUSD") is None
    assert progress.error


@pytest.mark.asyncio
async def test_backfill_progress_moves_before_a_block_commits(tmp_path):
    """A block is WRITE_BATCH_HOURS of hourly files.

    Counting progress only when one commits left a seventeen-thousand-hour
    backfill reading 0.0% for the first ten minutes and then stepping by
    whole percent, which looks exactly like a sync that has hung.
    """
    import app.data.sync as sync_module

    cache = make_cache(tmp_path)
    runner = DeltaSyncRunner(cache=cache, client=FakeClient())
    progress = sync_module.SyncProgress(symbol="XAUUSD", hours_total=10)

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    hours = [start + timedelta(hours=i) for i in range(10)]
    seen: list[int] = []

    async def fetch(hour):
        seen.append(progress.hours_done)
        return pd.DataFrame(
            {"timestamp": [hour], "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}
        )

    await runner._backfill("XAUUSD", "M1", hours, progress, fetch)

    assert progress.hours_done == 10
    # Some hour saw a count raised by an earlier hour in the same block,
    # rather than every one of them seeing zero.
    assert max(seen) > 0, f"progress never moved inside the block: {seen}"


@pytest.mark.asyncio
async def test_a_retried_block_does_not_count_its_hours_twice(tmp_path):
    import app.data.sync as sync_module

    cache = make_cache(tmp_path)
    runner = DeltaSyncRunner(cache=cache, client=FakeClient())
    progress = sync_module.SyncProgress(symbol="XAUUSD", hours_total=4)

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    hours = [start + timedelta(hours=i) for i in range(4)]
    attempt = {"n": 0}

    async def fetch(hour):
        if hour == hours[2] and attempt["n"] == 0:
            attempt["n"] += 1
            raise RuntimeError("feed threw once")
        return pd.DataFrame(
            {"timestamp": [hour], "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}
        )

    original = sync_module.RETRY_BACKOFF_SECONDS
    sync_module.RETRY_BACKOFF_SECONDS = 0
    try:
        synced = await runner._backfill("XAUUSD", "M1", hours, progress, fetch)
    finally:
        sync_module.RETRY_BACKOFF_SECONDS = original

    assert synced == 4
    assert progress.hours_done == 4, "a retried block must not inflate the count"
