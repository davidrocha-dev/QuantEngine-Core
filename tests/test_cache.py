from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.config import EngineConfig
from app.data.cache import LocalCache


def make_config(tmp_path: Path) -> EngineConfig:
    return EngineConfig(data_dir=tmp_path / "data", bundled_data_dir=tmp_path / "bundled")


def test_read_bars_missing_file_returns_empty_frame(tmp_path):
    cache = LocalCache(make_config(tmp_path))
    bars = cache.read_bars("EURUSD", "M1")
    assert bars.empty
    assert list(bars.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_append_and_read_bars_dedupes_and_sorts(tmp_path):
    cache = LocalCache(make_config(tmp_path))
    batch_1 = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 1, 0, tzinfo=timezone.utc)],
            "open": [1.1, 1.0], "high": [1.1, 1.0], "low": [1.1, 1.0], "close": [1.1, 1.0], "volume": [1.0, 1.0],
        }
    )
    batch_2 = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 1, 2, tzinfo=timezone.utc)],
            "open": [9.9, 1.2], "high": [9.9, 1.2], "low": [9.9, 1.2], "close": [9.9, 1.2], "volume": [1.0, 1.0],
        }
    )
    cache.append_bars("EURUSD", "M1", batch_1)
    cache.append_bars("EURUSD", "M1", batch_2)

    bars = cache.read_bars("EURUSD", "M1")
    assert len(bars) == 3
    assert list(bars["timestamp"]) == sorted(bars["timestamp"])


def test_manifest_round_trip(tmp_path):
    cache = LocalCache(make_config(tmp_path))
    assert cache.get_last_synced_hour("EURUSD") is None

    hour = datetime(2024, 5, 1, 12, tzinfo=timezone.utc)
    cache.set_last_synced_hour("EURUSD", hour)

    assert cache.get_last_synced_hour("EURUSD") == hour
    # A second symbol's manifest entry must not clobber the first.
    cache.set_last_synced_hour("GBPUSD", hour)
    assert cache.get_last_synced_hour("EURUSD") == hour
