"""Dukascopy Historical Data Feed client — REQ-DATA-1.1, design.md B.4.

Downloads hourly tick files (``.bi5``), decompresses them (LZMA_ALONE
container) and parses the fixed-width tick records into a DataFrame. This
is the sole source of breeding/backtest history (REQ-DATA-1.1/1.2) — the
MT5/broker connection never feeds this pipeline (see app/broker/mt5_bridge.py).
"""

from __future__ import annotations

import asyncio
import lzma
import struct
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd

from app.data.symbols import Instrument, get_instrument

FEED_BASE_URL = "https://datafeed.dukascopy.com/datafeed"

# Each tick record is 20 bytes: time offset (ms since hour start, uint32),
# ask (int32), bid (int32), ask volume (float32), bid volume (float32) —
# all big-endian. This is the standard Dukascopy .bi5 record layout.
_TICK_STRUCT = struct.Struct(">IIIff")


class DukascopyDownloadError(RuntimeError):
    """Raised when an hourly tick file cannot be fetched or is malformed."""


@dataclass(frozen=True)
class TickHour:
    symbol: str
    hour_start: datetime  # UTC, naive-free
    ticks: pd.DataFrame  # columns: timestamp, bid, ask, bid_volume, ask_volume


def build_hour_url(dukascopy_code: str, hour_start: datetime) -> str:
    return (
        f"{FEED_BASE_URL}/{dukascopy_code}/{hour_start.year:04d}/"
        f"{hour_start.month - 1:02d}/{hour_start.day:02d}/{hour_start.hour:02d}h_ticks.bi5"
    )


def decompress_bi5(raw: bytes) -> bytes:
    if not raw:
        return b""
    try:
        return lzma.decompress(raw, format=lzma.FORMAT_ALONE)
    except lzma.LZMAError as exc:
        raise DukascopyDownloadError("corrupt or non-LZMA .bi5 payload") from exc


def parse_ticks(decompressed: bytes, instrument: Instrument, hour_start: datetime) -> pd.DataFrame:
    if len(decompressed) % _TICK_STRUCT.size != 0:
        raise DukascopyDownloadError(
            f"tick payload size {len(decompressed)} is not a multiple of {_TICK_STRUCT.size}"
        )

    records = list(_TICK_STRUCT.iter_unpack(decompressed))
    if not records:
        return pd.DataFrame(columns=["timestamp", "bid", "ask", "bid_volume", "ask_volume"])

    divisor = instrument.point_divisor
    rows = {
        "timestamp": [hour_start + timedelta(milliseconds=r[0]) for r in records],
        "ask": [r[1] / divisor for r in records],
        "bid": [r[2] / divisor for r in records],
        "ask_volume": [r[3] for r in records],
        "bid_volume": [r[4] for r in records],
    }
    return pd.DataFrame(rows)[["timestamp", "bid", "ask", "bid_volume", "ask_volume"]]


class DukascopyClient:
    """A shared `httpx.AsyncClient` is reused across calls (connection
    keep-alive) instead of opening a fresh TLS connection per hour — that,
    combined with hammering the feed with no pacing at all, is what makes
    `429 Too Many Requests` trivial to hit on any multi-hour fetch (seen in
    practice: `python -m app.data.build_bundled_subset` started failing
    partway through its second symbol without either of these)."""

    def __init__(self, timeout_seconds: float = 20.0, max_retries: int = 8, request_delay_seconds: float = 0.4) -> None:
        self._client = httpx.AsyncClient(timeout=timeout_seconds)
        self._max_retries = max_retries
        self._request_delay_seconds = request_delay_seconds
        # Shared cooldown: a 429 means the *feed* is throttling this client,
        # not just this request, so every concurrent fetch backs off together.
        # Retrying per-request in isolation just burns the retry budget of
        # every in-flight hour at once (observed: a whole sync giving up).
        self._cooldown_until = 0.0
        self._cooldown_seconds = 0.0

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _await_cooldown(self) -> None:
        remaining = self._cooldown_until - asyncio.get_running_loop().time()
        if remaining > 0:
            await asyncio.sleep(remaining)

    def _enter_cooldown(self, seconds: float) -> None:
        self._cooldown_seconds = min(max(seconds, self._cooldown_seconds * 2, 5.0), 120.0)
        self._cooldown_until = asyncio.get_running_loop().time() + self._cooldown_seconds

    def _relax_cooldown(self) -> None:
        self._cooldown_seconds = max(0.0, self._cooldown_seconds / 2)

    async def fetch_hour(self, symbol: str, hour_start: datetime) -> TickHour:
        """Fetch and parse one hour of ticks. Returns an empty frame for
        hours with no trading activity (weekends, holidays) — Dukascopy
        serves an empty/absent file in that case, which is not an error."""
        instrument = get_instrument(symbol)
        url = build_hour_url(instrument.dukascopy_code, hour_start)

        for attempt in range(self._max_retries):
            await self._await_cooldown()
            try:
                response = await self._client.get(url)
            except httpx.TransportError as exc:
                # Beyond plain 429s, sustained request volume gets connections
                # refused/timed-out outright (observed in practice) — back off
                # and retry rather than letting one flaky hour kill the run.
                # Wrapped in our own error (not re-raised as-is) so callers
                # that catch DukascopyDownloadError — e.g. build_bundled_subset's
                # per-hour skip-and-continue — actually catch this too.
                if attempt == self._max_retries - 1:
                    raise DukascopyDownloadError(f"{type(exc).__name__} after {self._max_retries} retries: {url}") from exc
                await asyncio.sleep(min(2**attempt, 30))
                continue

            if response.status_code == 404:
                return TickHour(symbol, hour_start, parse_ticks(b"", instrument, hour_start))

            if response.status_code == 429:
                self._enter_cooldown(float(response.headers.get("retry-after", 0)))
                continue

            response.raise_for_status()
            self._relax_cooldown()
            if self._request_delay_seconds:
                await asyncio.sleep(self._request_delay_seconds)

            # LZMA decompression plus a struct unpack of several thousand
            # ticks is tens of milliseconds of pure CPU, and the sync runs
            # hundreds of these. Left on the event loop it starves every
            # HTTP request the app makes while a sync is running, which
            # reads as a frozen UI. Off the loop it is just work.
            ticks = await asyncio.to_thread(
                lambda payload: parse_ticks(decompress_bi5(payload), instrument, hour_start),
                response.content,
            )
            return TickHour(symbol, hour_start, ticks)

        raise DukascopyDownloadError(f"rate-limited after {self._max_retries} retries: {url}")


def resample_ticks_to_ohlc(ticks: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate mid-price ticks into OHLC bars. MVP consumes OHLC, not raw
    ticks — tick-mode backtesting is deferred post-MVP (design.md Part C)."""
    if ticks.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    freq_map = {
        "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
        "H1": "1h", "H2": "2h", "H4": "4h", "D1": "1D",
    }
    if timeframe not in freq_map:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    mid = (ticks["bid"] + ticks["ask"]) / 2
    series = pd.Series(mid.values, index=pd.DatetimeIndex(ticks["timestamp"]))
    ohlc = series.resample(freq_map[timeframe]).ohlc()
    volume = (ticks["bid_volume"] + ticks["ask_volume"])
    volume.index = series.index
    ohlc["volume"] = volume.resample(freq_map[timeframe]).sum()
    ohlc = ohlc.dropna(subset=["open"]).reset_index()
    ohlc = ohlc.rename(columns={"index": "timestamp"})
    return ohlc[["timestamp", "open", "high", "low", "close", "volume"]]


def utc_hour_range(start: datetime, end: datetime):
    """Yield each hour boundary in [start, end), both UTC."""
    current = start.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    end = end.astimezone(timezone.utc)
    while current < end:
        yield current
        current += timedelta(hours=1)
