"""Raw tick access for verification — design.md Part C (tick mode).

The sync keeps M1 bars and throws the ticks away, which is the right
trade for breeding: a year of EURUSD ticks is tens of millions of rows
and the genetic search would spend all its time reading them.

Verifying a finished strategy is a different problem. It needs ticks, but
only for the hours its own trades were open in - typically a few hundred
hours out of the year, not all 8,760. So this fetches by the hour, caches
each hour it fetched, and never asks for an hour twice.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from app.config import EngineConfig, get_config
from app.data.dukascopy_client import DukascopyClient

EMPTY_TICKS = pd.DataFrame(columns=["timestamp", "bid", "ask", "bid_volume", "ask_volume"])


def hours_covering(windows: list[tuple[datetime, datetime]]) -> list[datetime]:
    """Every UTC hour touched by any of these [start, end] windows.

    Deduplicated and sorted, because two trades overlapping the same hour
    must not download it twice.
    """
    needed: set[datetime] = set()
    for start, end in windows:
        if end < start:
            start, end = end, start
        hour = start.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        last = end.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        while hour <= last:
            needed.add(hour)
            hour += timedelta(hours=1)
    return sorted(needed)


class TickStore:
    """Hourly ticks, on disk, fetched once.

    The cache is not an optimisation here so much as a courtesy: the feed
    throttles hard (see DukascopyClient), and re-verifying the same
    strategy twice should cost nothing the second time.
    """

    def __init__(self, config: EngineConfig | None = None, client: DukascopyClient | None = None) -> None:
        self._config = config or get_config()
        self._client = client or DukascopyClient()

    @property
    def _root(self) -> Path:
        return self._config.cache_dir / "ticks"

    def _path(self, symbol: str, hour: datetime) -> Path:
        return self._root / symbol / f"{hour:%Y-%m-%d_%H}.parquet"

    def cached(self, symbol: str, hour: datetime) -> bool:
        return self._path(symbol, hour).exists()

    def read(self, symbol: str, hour: datetime) -> pd.DataFrame:
        path = self._path(symbol, hour)
        if not path.exists():
            return EMPTY_TICKS.copy()
        return pd.read_parquet(path)

    def write(self, symbol: str, hour: datetime, ticks: pd.DataFrame) -> None:
        path = self._path(symbol, hour)
        path.parent.mkdir(parents=True, exist_ok=True)
        # An hour with no ticks (a weekend, a holiday) is still a fact
        # worth remembering, or every verification re-asks for it.
        ticks.to_parquet(path, index=False)

    async def ensure(self, symbol: str, hours: list[datetime], on_progress=None) -> dict[datetime, pd.DataFrame]:
        """Returns the ticks for these hours, downloading what is missing.

        Sequential on purpose. The feed rate-limits aggressively and this
        runs while a user waits: a verification that finishes slowly is
        better than one that trips a cooldown and finishes much later.
        """
        loaded: dict[datetime, pd.DataFrame] = {}
        for index, hour in enumerate(hours):
            if self.cached(symbol, hour):
                loaded[hour] = self.read(symbol, hour)
            else:
                tick_hour = await self._client.fetch_hour(symbol, hour)
                await asyncio.to_thread(self.write, symbol, hour, tick_hour.ticks)
                loaded[hour] = tick_hour.ticks
            if on_progress is not None:
                on_progress(index + 1, len(hours))
        return loaded

    async def aclose(self) -> None:
        await self._client.aclose()


def concat_hours(hours: dict[datetime, pd.DataFrame]) -> pd.DataFrame:
    """One frame, sorted by time, ready to slice per trade."""
    frames = [frame for frame in hours.values() if not frame.empty]
    if not frames:
        return EMPTY_TICKS.copy()
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("timestamp").reset_index(drop=True)
