"""Incremental delta sync on app startup — REQ-DATA-1.5.

Downloads only the hours missing since the last successful sync per symbol,
instead of re-fetching the full history each launch. Three properties make
this usable rather than merely correct:

  * **Bounded start.** A first sync starts `history_depth_years` back (the
    Settings → Campaign value), not at the 2003 Dukascopy epoch — 22 years
    of hourly tick files is ~190,000 requests per symbol.
  * **Concurrent fetches.** Hours are fetched in parallel batches; the feed
    rate-limits, so concurrency stays modest and the client's own retry /
    backoff handles 429s.
  * **Batched writes.** Bars are buffered and appended once per batch.
    Appending per hour rewrote the whole Parquet file every time, which is
    quadratic in the size of the cache.

Progress is observable (`SyncProgress`) so the UI can show what is
happening during what is, unavoidably, a long first download.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

from app.data.cache import LocalCache
from app.data.dukascopy_client import DukascopyClient, resample_ticks_to_ohlc, utc_hour_range
from app.data.symbols import DUKASCOPY_EPOCH_MONTH, DUKASCOPY_EPOCH_YEAR

logger = logging.getLogger(__name__)

# Never sync the current, still-forming hour.
SYNC_LAG = timedelta(hours=1)
DEFAULT_TIMEFRAME = "M1"
DEFAULT_HISTORY_YEARS = 3
# Hours fetched concurrently. The feed 429s readily, so this stays low —
# the win over serial fetching is still roughly an order of magnitude.
CONCURRENT_HOURS = 6
# Bars are flushed to Parquet once per batch of this many hours.
WRITE_BATCH_HOURS = 24 * 7
# How many times a batch is retried after a throttled/failed hour before the
# symbol is left for the next run (progress so far is always kept).
MAX_BATCH_RETRIES = 5
RETRY_BACKOFF_SECONDS = 30


@dataclass
class SyncProgress:
    symbol: str
    hours_total: int = 0
    hours_done: int = 0
    finished: bool = False
    error: str | None = None

    @property
    def pct(self) -> float:
        return 100.0 if self.finished else (100.0 * self.hours_done / self.hours_total if self.hours_total else 0.0)


@dataclass
class SyncState:
    """Live view of the running sync, polled by GET /data/sync/status."""

    running: bool = False
    per_symbol: dict[str, SyncProgress] = field(default_factory=dict)

    def snapshot(self) -> dict:
        return {
            "running": self.running,
            "symbols": {
                symbol: {
                    "hours_total": p.hours_total, "hours_done": p.hours_done,
                    "pct": round(p.pct, 1), "finished": p.finished, "error": p.error,
                }
                for symbol, p in self.per_symbol.items()
            },
        }


class DeltaSyncRunner:
    def __init__(
        self,
        cache: LocalCache | None = None,
        client: DukascopyClient | None = None,
        history_years: int = DEFAULT_HISTORY_YEARS,
        state: SyncState | None = None,
        concurrency: int = CONCURRENT_HOURS,
    ) -> None:
        self._cache = cache or LocalCache()
        self._client = client or DukascopyClient()
        self._history_years = history_years
        self._state = state or SyncState()
        self._concurrency = max(1, concurrency)

    @property
    def state(self) -> SyncState:
        return self._state

    def _depth_start(self) -> datetime:
        depth_start = datetime.now(timezone.utc) - timedelta(days=365 * self._history_years)
        epoch = datetime(DUKASCOPY_EPOCH_YEAR, DUKASCOPY_EPOCH_MONTH, 1, tzinfo=timezone.utc)
        return max(depth_start, epoch)

    def _sync_start(self, symbol: str) -> datetime:
        last_synced = self._cache.get_last_synced_hour(symbol)
        if last_synced is not None:
            return last_synced + timedelta(hours=1)
        return self._depth_start()

    def _backfill_hours(self, symbol: str, timeframe: str) -> list[datetime]:
        """Hours older than the cache that the configured depth asks for.

        A single forward watermark cannot express this. Sync resumed from
        it and only ever walked towards now, so raising HISTORY DEPTH
        after a symbol had been synced did nothing whatsoever - no error,
        no message, no download. The setting said three years, the cache
        held one, and the next campaign trained on the one without ever
        saying so. That is not a small difference: a strategy fitted to
        twelve months of a single regime scores well on the tail of its
        own training data and then meets the years it never saw.
        """
        first_synced = self._cache.get_first_synced_hour(symbol)
        if first_synced is None:
            # Written by a build that only tracked the forward edge; the
            # bars on disk still say where the cache begins.
            first_synced = self._cache.get_first_bar_hour(symbol, timeframe)
        if first_synced is None:
            return []  # Nothing cached: the forward pass starts at the depth anyway.
        depth_start = self._depth_start()
        if depth_start >= first_synced:
            return []
        return list(utc_hour_range(depth_start, first_synced))

    async def _fetch_bars(self, symbol: str, hour_start: datetime, timeframe: str) -> pd.DataFrame | None:
        tick_hour = await self._client.fetch_hour(symbol, hour_start)
        if tick_hour.ticks.empty:
            return None
        return await asyncio.to_thread(resample_ticks_to_ohlc, tick_hour.ticks, timeframe)

    async def sync_symbol(self, symbol: str, timeframe: str = DEFAULT_TIMEFRAME) -> int:
        """Sync one symbol; returns the number of hours fetched."""
        end = datetime.now(timezone.utc) - SYNC_LAG
        hours = list(utc_hour_range(self._sync_start(symbol), end))
        backfill = self._backfill_hours(symbol, timeframe)

        progress = SyncProgress(symbol=symbol, hours_total=len(hours) + len(backfill))
        self._state.per_symbol[symbol] = progress
        if not hours and not backfill:
            progress.finished = True
            return 0

        semaphore = asyncio.Semaphore(self._concurrency)

        async def fetch(hour: datetime) -> pd.DataFrame | None:
            async with semaphore:
                return await self._fetch_bars(symbol, hour, timeframe)

        hours_synced = 0
        batch_start = 0
        attempts = 0
        while batch_start < len(hours):
            batch = hours[batch_start : batch_start + WRITE_BATCH_HOURS]
            results = await asyncio.gather(*(fetch(hour) for hour in batch), return_exceptions=True)

            frames: list[pd.DataFrame] = []
            completed = 0
            failure: Exception | None = None
            for hour, result in zip(batch, results):
                if isinstance(result, Exception):
                    # The manifest must stay contiguous, so a failed hour
                    # ends this batch — everything before it is still saved.
                    failure = result
                    break
                if result is not None and not result.empty:
                    frames.append(result)
                completed += 1

            if frames:
                # append_bars rewrites the whole parquet file: read, concat,
                # dedupe, sort, write. On a cache with a year of M1 bars in
                # it that is far too long to hold the event loop for, and it
                # happens once per batch.
                await asyncio.to_thread(
                    self._cache.append_bars, symbol, timeframe, pd.concat(frames, ignore_index=True)
                )
            if completed:
                await asyncio.to_thread(self._cache.set_last_synced_hour, symbol, batch[completed - 1])
                hours_synced += completed
                progress.hours_done += completed
                batch_start += completed

            if failure is None:
                attempts = 0
                continue

            # A failure here is almost always the feed throttling us. Wait
            # longer each time rather than abandoning the whole symbol: the
            # alternative is a sync that quietly stops months short.
            attempts += 1
            if attempts > MAX_BATCH_RETRIES:
                progress.error = str(failure)
                logger.warning("Sync stopped for %s at %s: %s", symbol, batch[completed], failure)
                break
            backoff = RETRY_BACKOFF_SECONDS * attempts
            logger.info("Sync backing off %ss for %s (attempt %s)", backoff, symbol, attempts)
            await asyncio.sleep(backoff)

        # The forward pass goes first so the app is current within
        # minutes; the older years, which can be tens of thousands of
        # hours, arrive behind it.
        hours_synced += await self._backfill(symbol, timeframe, backfill, progress, fetch)

        progress.finished = True
        return hours_synced

    async def _backfill(
        self,
        symbol: str,
        timeframe: str,
        hours: list[datetime],
        progress: SyncProgress,
        fetch,
    ) -> int:
        """Fetches `hours` from the newest backwards, one contiguous block
        at a time.

        Unlike the forward pass, a partly-finished block is not committed.
        Forward, a short batch just means the cache stops earlier than it
        could and the next run resumes there. Backwards, it would leave a
        hole in the middle of the cached range that nothing downstream can
        detect: `read_bars` returns a frame, the backtester resamples it,
        and the missing weeks simply are not there.
        """
        synced = 0
        attempts = 0
        end = len(hours)
        while end > 0:
            start = max(0, end - WRITE_BATCH_HOURS)
            batch = hours[start:end]
            results = await asyncio.gather(*(fetch(hour) for hour in batch), return_exceptions=True)

            failure = next((r for r in results if isinstance(r, Exception)), None)
            if failure is not None:
                attempts += 1
                if attempts > MAX_BATCH_RETRIES:
                    progress.error = str(failure)
                    logger.warning("Backfill stopped for %s at %s: %s", symbol, batch[0], failure)
                    break
                backoff = RETRY_BACKOFF_SECONDS * attempts
                logger.info("Backfill backing off %ss for %s (attempt %s)", backoff, symbol, attempts)
                await asyncio.sleep(backoff)
                continue

            frames = [r for r in results if r is not None and not r.empty]
            if frames:
                await asyncio.to_thread(
                    self._cache.append_bars, symbol, timeframe, pd.concat(frames, ignore_index=True)
                )
            await asyncio.to_thread(self._cache.set_first_synced_hour, symbol, batch[0])
            synced += len(batch)
            progress.hours_done += len(batch)
            end = start
            attempts = 0
        return synced

    async def sync_all(self, symbols: list[str], timeframe: str = DEFAULT_TIMEFRAME) -> dict[str, int]:
        results: dict[str, int] = {}
        self._state.running = True
        try:
            for symbol in symbols:
                try:
                    results[symbol] = await self.sync_symbol(symbol, timeframe)
                except Exception as exc:
                    logger.exception("Delta sync failed for %s", symbol)
                    progress = self._state.per_symbol.setdefault(symbol, SyncProgress(symbol=symbol))
                    progress.error, progress.finished = str(exc), True
                    results[symbol] = 0
        finally:
            self._state.running = False
            close = getattr(self._client, "aclose", None)
            if close is not None:
                await close()
        return results
