"""Single entrypoint the GA/backtester uses to read historical bars.

Enforces REQ-DATA-1.6 (BUILT-IN DATA ONLY forces the bundled subset
regardless of full cache/broker availability) and REQ-DATA-1.2/1.3 (this
module — not the MT5 bridge — is the only historical data path; see
app/broker/mt5_bridge.py for the explicit live-only boundary).
"""

from __future__ import annotations

import pandas as pd

from app.config import EngineConfig, get_config
from app.data.broker_clock import UTC_CLOCK, BrokerClock, clock_from_settings
from app.data.bundled import read_bundled_bars
from app.data.cache import LocalCache


BASE_TIMEFRAME = "M1"

_RESAMPLE_FREQ = {
    "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
    "H1": "1h", "H2": "2h", "H4": "4h", "D1": "1D",
}


# How long one bar lasts. Anything that has to turn a bar index back into
# a wall-clock window - tick verification, for one - needs this, and a
# second copy of it elsewhere would be a second thing to get wrong.
TIMEFRAME_MINUTES = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H2": 120, "H4": 240, "D1": 1440,
}


def resample_bars(bars: pd.DataFrame, timeframe: str, clock: BrokerClock = UTC_CLOCK) -> pd.DataFrame:
    """Aggregates M1 OHLCV bars into `timeframe` bars on the broker's clock.

    Bar boundaries are the broker's: an H4 bar on a New York close server
    opens at 00:00, 04:00, ... server time, which is 22:00, 02:00, ... UTC
    in winter. `timestamp` stays the UTC instant the bar opened (like MT5,
    a bar is labelled by its open) and `broker_time` is the same instant
    on the server's clock. Empty periods (weekends, holidays) are dropped
    rather than forward-filled.
    """
    if timeframe not in _RESAMPLE_FREQ:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    if bars.empty:
        return bars
    utc = pd.DatetimeIndex(bars["timestamp"])
    utc = utc.tz_localize("UTC") if utc.tz is None else utc.tz_convert("UTC")
    local = clock.local(utc)
    if timeframe == BASE_TIMEFRAME:
        out = bars.copy()
        out["broker_time"] = local
        return out

    bucket = local.floor(_RESAMPLE_FREQ[timeframe])
    # The offset in force at each bar, so a bucket's broker-time boundary
    # can be turned back into the UTC instant it happened at. It only
    # changes at a daylight-saving switch, which falls on a weekend.
    offset = pd.Series(local - utc.tz_localize(None), index=bars.index)
    frame = bars[["open", "high", "low", "close", "volume"]].assign(_bucket=bucket, _offset=offset.to_numpy())
    grouped = frame.groupby("_bucket", sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), _offset=("_offset", "first"),
    )
    grouped = grouped.dropna(subset=["open"])
    broker_time = pd.DatetimeIndex(grouped.index)
    timestamp = (broker_time - pd.TimedeltaIndex(grouped["_offset"])).tz_localize("UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": grouped["open"].to_numpy(), "high": grouped["high"].to_numpy(),
            "low": grouped["low"].to_numpy(), "close": grouped["close"].to_numpy(),
            "volume": grouped["volume"].to_numpy(), "broker_time": broker_time,
        }
    )


def _read_with_resample(read, symbol: str, timeframe: str, clock: BrokerClock) -> pd.DataFrame:
    # Both the bundled subset and delta sync (app/data/sync.py) only store
    # M1 bars — every other timeframe is derived from M1 on read, on the
    # broker's clock. A stored higher-timeframe file, if one exists, was
    # cut on UTC boundaries and is only a fallback when there is no M1.
    m1 = read(symbol, BASE_TIMEFRAME)
    if not m1.empty:
        return resample_bars(m1, timeframe, clock)
    if timeframe == BASE_TIMEFRAME:
        return m1
    stored = read(symbol, timeframe)
    return resample_bars(stored, BASE_TIMEFRAME, clock) if not stored.empty else stored


def _configured_clock(config: EngineConfig) -> BrokerClock:
    from app.settings_store import load_settings

    try:
        return clock_from_settings(load_settings(config.settings_path))
    except Exception:  # an unreadable settings file must not stop data loading
        return clock_from_settings(None)


def get_historical_bars(
    symbol: str,
    timeframe: str,
    config: EngineConfig | None = None,
    cache: LocalCache | None = None,
    clock: BrokerClock | None = None,
) -> pd.DataFrame:
    """Bars for `symbol`/`timeframe`, with each bar's `broker_time` on the
    clock in Settings -> Broker (or `clock`, when given)."""
    config = config or get_config()
    clock = clock or _configured_clock(config)

    def read_bundled(sym: str, tf: str) -> pd.DataFrame:
        return read_bundled_bars(sym, tf, config)

    if config.built_in_data_only:
        return _read_with_resample(read_bundled, symbol, timeframe, clock)

    cache = cache or LocalCache(config)
    bars = _read_with_resample(cache.read_bars, symbol, timeframe, clock)
    if bars.empty:
        # Full cache not synced yet for this symbol — fall back to the
        # bundled subset so a campaign can still start (REQ-DATA-1.4).
        return _read_with_resample(read_bundled, symbol, timeframe, clock)
    return bars
