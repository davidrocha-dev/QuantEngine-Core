"""Incremental delta sync on app startup — REQ-DATA-1.5.

Downloads only the hours missing since the last successful sync per
symbol/timeframe, instead of re-fetching the full history each launch.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from app.data.cache import LocalCache
from app.data.dukascopy_client import DukascopyClient, resample_ticks_to_ohlc, utc_hour_range
from app.data.symbols import DUKASCOPY_EPOCH_MONTH, DUKASCOPY_EPOCH_YEAR

logger = logging.getLogger(__name__)

# Never sync the current, still-forming hour.
SYNC_LAG = timedelta(hours=1)
DEFAULT_TIMEFRAME = "M1"


class DeltaSyncRunner:
    def __init__(self, cache: LocalCache | None = None, client: DukascopyClient | None = None) -> None:
        self._cache = cache or LocalCache()
        self._client = client or DukascopyClient()

    def _sync_start(self, symbol: str) -> datetime:
        last_synced = self._cache.get_last_synced_hour(symbol)
        if last_synced is not None:
            return last_synced + timedelta(hours=1)
        return datetime(DUKASCOPY_EPOCH_YEAR, DUKASCOPY_EPOCH_MONTH, 1, tzinfo=timezone.utc)

    async def sync_symbol(self, symbol: str, timeframe: str = DEFAULT_TIMEFRAME) -> int:
        """Sync one symbol; returns the number of hours fetched."""
        start = self._sync_start(symbol)
        end = datetime.now(timezone.utc) - SYNC_LAG
        hours_synced = 0

        for hour_start in utc_hour_range(start, end):
            tick_hour = await self._client.fetch_hour(symbol, hour_start)
            if not tick_hour.ticks.empty:
                bars = resample_ticks_to_ohlc(tick_hour.ticks, timeframe)
                self._cache.append_bars(symbol, timeframe, bars)
            self._cache.set_last_synced_hour(symbol, hour_start)
            hours_synced += 1

        return hours_synced

    async def sync_all(self, symbols: list[str], timeframe: str = DEFAULT_TIMEFRAME) -> dict[str, int]:
        results: dict[str, int] = {}
        for symbol in symbols:
            try:
                results[symbol] = await self.sync_symbol(symbol, timeframe)
            except Exception:
                logger.exception("Delta sync failed for %s", symbol)
                results[symbol] = 0
        return results
