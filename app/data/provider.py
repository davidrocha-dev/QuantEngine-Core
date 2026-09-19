"""Single entrypoint the GA/backtester uses to read historical bars.

Enforces REQ-DATA-1.6 (BUILT-IN DATA ONLY forces the bundled subset
regardless of full cache/broker availability) and REQ-DATA-1.2/1.3 (this
module — not the MT5 bridge — is the only historical data path; see
app/broker/mt5_bridge.py for the explicit live-only boundary).
"""

from __future__ import annotations

import pandas as pd

from app.config import EngineConfig, get_config
from app.data.bundled import read_bundled_bars
from app.data.cache import LocalCache


BASE_TIMEFRAME = "M1"

_RESAMPLE_FREQ = {
    "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
    "H1": "1h", "H2": "2h", "H4": "4h", "D1": "1D",
}


def resample_bars(bars: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregates M1 OHLCV bars into `timeframe` bars (left-labelled, like
    MT5: a bar's timestamp is its open time). Empty periods (weekends,
    holidays) are dropped rather than forward-filled."""
    if timeframe not in _RESAMPLE_FREQ:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    if bars.empty or timeframe == BASE_TIMEFRAME:
        return bars
    indexed = bars.set_index(pd.DatetimeIndex(bars["timestamp"]))
    out = indexed.resample(_RESAMPLE_FREQ[timeframe], label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    out = out.dropna(subset=["open"]).reset_index(names="timestamp")
    return out[["timestamp", "open", "high", "low", "close", "volume"]]


def _read_with_resample(read, symbol: str, timeframe: str) -> pd.DataFrame:
    # Both the bundled subset and delta sync (app/data/sync.py) only store
    # M1 bars — every other timeframe is derived from M1 on read.
    bars = read(symbol, timeframe)
    if bars.empty and timeframe != BASE_TIMEFRAME:
        bars = resample_bars(read(symbol, BASE_TIMEFRAME), timeframe)
    return bars


def get_historical_bars(
    symbol: str,
    timeframe: str,
    config: EngineConfig | None = None,
    cache: LocalCache | None = None,
) -> pd.DataFrame:
    config = config or get_config()

    def read_bundled(sym: str, tf: str) -> pd.DataFrame:
        return read_bundled_bars(sym, tf, config)

    if config.built_in_data_only:
        return _read_with_resample(read_bundled, symbol, timeframe)

    cache = cache or LocalCache(config)
    bars = _read_with_resample(cache.read_bars, symbol, timeframe)
    if bars.empty:
        # Full cache not synced yet for this symbol — fall back to the
        # bundled subset so a campaign can still start (REQ-DATA-1.4).
        return _read_with_resample(read_bundled, symbol, timeframe)
    return bars
