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

    def read_manifest(self) -> dict[str, str]:
        if not self._config.manifest_path.exists():
            return {}
        return json.loads(self._config.manifest_path.read_text())

    def write_manifest(self, manifest: dict[str, str]) -> None:
        self._config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def get_last_synced_hour(self, symbol: str) -> datetime | None:
        manifest = self.read_manifest()
        raw = manifest.get(symbol)
        if raw is None:
            return None
        return datetime.fromisoformat(raw)

    def set_last_synced_hour(self, symbol: str, hour: datetime) -> None:
        manifest = self.read_manifest()
        manifest[symbol] = hour.astimezone(timezone.utc).isoformat()
        self.write_manifest(manifest)
