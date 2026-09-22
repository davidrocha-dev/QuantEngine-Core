"""Local Dukascopy cache — design.md B.4.

OHLC bars are cached to Parquet, one file per symbol/timeframe, with a JSON
manifest tracking the last fully-synced UTC hour per symbol so startup sync
(REQ-DATA-1.5) only fetches the delta instead of re-downloading everything.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.config import EngineConfig, get_config


class LocalCache:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self._config = config or get_config()
        self._config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _bar_path(self, symbol: str, timeframe: str) -> Path:
        return self._config.cache_dir / f"{symbol}_{timeframe}.parquet"

    def read_bars(self, symbol: str, timeframe: str) -> pd.DataFrame:
        path = self._bar_path(symbol, timeframe)
        if not path.exists():
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        return pd.read_parquet(path)

    def append_bars(self, symbol: str, timeframe: str, bars: pd.DataFrame) -> None:
        if bars.empty:
            return
        existing = self.read_bars(symbol, timeframe)
        combined = pd.concat([existing, bars], ignore_index=True)
        combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp")
        combined.to_parquet(self._bar_path(symbol, timeframe), index=False)

    def read_manifest(self) -> dict[str, dict | str]:
        if not self._config.manifest_path.exists():
            return {}
        return json.loads(self._config.manifest_path.read_text())

    def write_manifest(self, manifest: dict[str, dict | str]) -> None:
        self._config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def _entry(self, symbol: str) -> dict[str, str]:
        """One symbol's synced range.

        The manifest used to hold a bare ISO string per symbol: the last
        synced hour, and nothing about where the cache began. That was
        enough while sync only ever moved forward, and became the reason
        it could only ever move forward. Older entries still read, so a
        cache written by an earlier build keeps working.
        """
        raw = self.read_manifest().get(symbol)
        if raw is None:
            return {}
        if isinstance(raw, str):
            return {"last": raw}
        return dict(raw)

    def _write_entry(self, symbol: str, **fields: str) -> None:
        manifest = self.read_manifest()
        entry = self._entry(symbol)
        entry.update(fields)
        manifest[symbol] = entry
        self.write_manifest(manifest)

    def get_last_synced_hour(self, symbol: str) -> datetime | None:
        raw = self._entry(symbol).get("last")
        return datetime.fromisoformat(raw) if raw else None

    def set_last_synced_hour(self, symbol: str, hour: datetime) -> None:
        self._write_entry(symbol, last=hour.astimezone(timezone.utc).isoformat())

    def get_first_synced_hour(self, symbol: str) -> datetime | None:
        raw = self._entry(symbol).get("first")
        return datetime.fromisoformat(raw) if raw else None

    def set_first_synced_hour(self, symbol: str, hour: datetime) -> None:
        """Moves the start of the cached range earlier, never later.

        Backfill walks towards the past, so this only ever retreats; a
        value that moved forward would mark hours as missing that are
        sitting in the Parquet file.
        """
        hour = hour.astimezone(timezone.utc)
        current = self.get_first_synced_hour(symbol)
        if current is not None and current <= hour:
            return
        self._write_entry(symbol, first=hour.isoformat())

    def get_first_bar_hour(self, symbol: str, timeframe: str) -> datetime | None:
        """The oldest bar actually on disk.

        The fallback for a cache whose manifest predates `first`: the bars
        themselves know where they start.
        """
        bars = self.read_bars(symbol, timeframe)
        if bars.empty:
            return None
        oldest = pd.DatetimeIndex(bars["timestamp"]).min()
        if oldest.tz is None:
            oldest = oldest.tz_localize(timezone.utc)
        return oldest.to_pydatetime().replace(minute=0, second=0, microsecond=0)
