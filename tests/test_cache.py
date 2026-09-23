from unittest.mock import patch
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


def test_a_reader_never_sees_a_half_written_cache(tmp_path):
    """The sync rewrites the whole file each time a block lands, while
    /data/coverage polls it and campaigns load it. Written in place, the
    file is truncated for a couple of hundred milliseconds and a reader
    landing there gets an exception, not a short frame."""
    import pandas as pd

    from app.config import EngineConfig
    from app.data.cache import LocalCache

    cache = LocalCache(EngineConfig(data_dir=tmp_path / "data", bundled_data_dir=tmp_path / "b"))

    def frame(n: int, start: str) -> pd.DataFrame:
        stamps = pd.date_range(start, periods=n, freq="1min", tz="UTC")
        return pd.DataFrame(
            {"timestamp": stamps, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}
        )

    cache.append_bars("XAUUSD", "M1", frame(100, "2025-01-01"))
    assert len(cache.read_bars("XAUUSD", "M1")) == 100

    seen: list[int] = []
    original = pd.DataFrame.to_parquet

    def watching_to_parquet(self, path, *args, **kwargs):
        result = original(self, path, *args, **kwargs)
        # Mid-write: whatever a reader would get from the real path now.
        seen.append(len(cache.read_bars("XAUUSD", "M1")))
        return result

    with patch.object(pd.DataFrame, "to_parquet", watching_to_parquet):
        cache.append_bars("XAUUSD", "M1", frame(50, "2025-01-02"))

    assert seen == [100], f"a reader saw {seen} rows while the new file was being written"
    assert len(cache.read_bars("XAUUSD", "M1")) == 150
    assert not list((tmp_path / "data").rglob("*.tmp")), "temporary file left behind"
