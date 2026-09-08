"""Pre-bundled Dukascopy subset — REQ-DATA-1.4.

A small OHLC dataset for BUNDLED_SYMBOLS ships inside the installer
(``engine/app/data/bundled/*.parquet``, pulled in by PyInstaller —
design.md B.9) so the app is usable before the first full sync completes.
Regenerate with ``python -m app.data.build_bundled_subset``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.config import EngineConfig, get_config
from app.data.symbols import BUNDLED_SYMBOLS


def bundled_bar_path(bundled_dir: Path, symbol: str, timeframe: str) -> Path:
    return bundled_dir / f"{symbol}_{timeframe}.parquet"


def read_bundled_bars(symbol: str, timeframe: str, config: EngineConfig | None = None) -> pd.DataFrame:
    config = config or get_config()
    path = bundled_bar_path(config.bundled_data_dir, symbol, timeframe)
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    return pd.read_parquet(path)


def is_bundled_symbol(symbol: str) -> bool:
    return symbol.upper() in BUNDLED_SYMBOLS
