from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


def make_synthetic_ohlc(n_bars: int = 500, seed: int = 42, drift: float = 0.0002, volatility: float = 0.001) -> pd.DataFrame:
    """Deterministic synthetic H1 OHLC series (geometric-ish random walk with
    drift) used across engine tests instead of live Dukascopy data."""
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(hours=i) for i in range(n_bars)]

    returns = rng.normal(drift, volatility, n_bars)
    close = 100 * np.cumprod(1 + returns)
    open_ = np.roll(close, 1)
    open_[0] = 100.0
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.0015, n_bars))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.0015, n_bars))
    volume = rng.uniform(50, 200, n_bars)

    return pd.DataFrame(
        {"timestamp": timestamps, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )
