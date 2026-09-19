from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from app.config import EngineConfig
from app.data.bundled import bundled_bar_path
from app.data.cache import LocalCache
from app.data.provider import get_historical_bars


def make_config(tmp_path: Path) -> EngineConfig:
    return EngineConfig(data_dir=tmp_path / "data", bundled_data_dir=tmp_path / "bundled")


def write_bundled_bars(config: EngineConfig, symbol: str, timeframe: str) -> pd.DataFrame:
    bars = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0],
        }
    )
    config.bundled_data_dir.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(bundled_bar_path(config.bundled_data_dir, symbol, timeframe), index=False)
    return bars


def test_built_in_data_only_forces_bundled_even_with_full_cache(tmp_path):
    config = make_config(tmp_path)
    config.built_in_data_only = True
    cache = LocalCache(config)

    full_cache_bars = pd.DataFrame(
        {
            "timestamp": [datetime(2020, 1, 1, tzinfo=timezone.utc)],
            "open": [2.0], "high": [2.0], "low": [2.0], "close": [2.0], "volume": [1.0],
        }
    )
    cache.append_bars("EURUSD", "M1", full_cache_bars)
    write_bundled_bars(config, "EURUSD", "M1")

    result = get_historical_bars("EURUSD", "M1", config, cache)

    assert result.iloc[0]["close"] == pytest.approx(1.0)  # bundled value, not the cached one


def test_falls_back_to_bundled_when_cache_empty(tmp_path):
    config = make_config(tmp_path)
    cache = LocalCache(config)
    write_bundled_bars(config, "EURUSD", "M1")

    result = get_historical_bars("EURUSD", "M1", config, cache)

    assert len(result) == 1
    assert result.iloc[0]["close"] == pytest.approx(1.0)


def test_uses_full_cache_when_available_and_not_built_in_only(tmp_path):
    config = make_config(tmp_path)
    cache = LocalCache(config)
    cached = pd.DataFrame(
        {
            "timestamp": [datetime(2020, 1, 1, tzinfo=timezone.utc)],
            "open": [3.0], "high": [3.0], "low": [3.0], "close": [3.0], "volume": [1.0],
        }
    )
    cache.append_bars("EURUSD", "M1", cached)

    result = get_historical_bars("EURUSD", "M1", config, cache)

    assert result.iloc[0]["close"] == pytest.approx(3.0)


def test_higher_timeframe_is_resampled_from_m1(tmp_path):
    # Bundled data and delta sync only store M1 — an H1 request must be
    # derived from it rather than coming back empty.
    config = make_config(tmp_path)
    config.built_in_data_only = True
    timestamps = pd.date_range("2024-01-01 10:00", periods=120, freq="1min", tz="UTC")
    m1 = pd.DataFrame(
        {"timestamp": timestamps, "open": range(120), "high": range(1, 121), "low": range(120), "close": range(120), "volume": [1.0] * 120}
    ).astype({"open": float, "high": float, "low": float, "close": float})
    config.bundled_data_dir.mkdir(parents=True, exist_ok=True)
    m1.to_parquet(bundled_bar_path(config.bundled_data_dir, "EURUSD", "M1"), index=False)

    h1 = get_historical_bars("EURUSD", "H1", config)

    assert len(h1) == 2
    first = h1.iloc[0]
    assert first["timestamp"] == timestamps[0]
    assert (first["open"], first["high"], first["low"], first["close"], first["volume"]) == (0.0, 60.0, 0.0, 59.0, 60.0)
