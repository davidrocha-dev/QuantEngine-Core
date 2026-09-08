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
