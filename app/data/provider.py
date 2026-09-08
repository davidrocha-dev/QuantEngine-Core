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


def get_historical_bars(
    symbol: str,
    timeframe: str,
    config: EngineConfig | None = None,
    cache: LocalCache | None = None,
) -> pd.DataFrame:
    config = config or get_config()

    if config.built_in_data_only:
        return read_bundled_bars(symbol, timeframe, config)

    cache = cache or LocalCache(config)
    bars = cache.read_bars(symbol, timeframe)
    if bars.empty:
        # Full cache not synced yet for this symbol — fall back to the
        # bundled subset so a campaign can still start (REQ-DATA-1.4).
        return read_bundled_bars(symbol, timeframe, config)
    return bars
