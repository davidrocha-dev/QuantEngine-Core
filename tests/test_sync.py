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


def test_sync_start_uses_dukascopy_epoch_when_never_synced(tmp_path):
    cache = make_cache(tmp_path)
    runner = DeltaSyncRunner(cache=cache, client=FakeClient())

    start = runner._sync_start("EURUSD")
    assert start == datetime(2003, 5, 1, tzinfo=timezone.utc)


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
