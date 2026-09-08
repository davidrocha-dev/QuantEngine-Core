"""Build-time script: regenerates the pre-bundled Dukascopy subset shipped
in the installer (REQ-DATA-1.4). Not run at app runtime.

Usage: python -m app.data.build_bundled_subset [--days 30]
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone

from app.data.bundled import bundled_bar_path
from app.data.dukascopy_client import DukascopyClient, DukascopyDownloadError, resample_ticks_to_ohlc, utc_hour_range
from app.data.symbols import BUNDLED_SYMBOLS

BUNDLED_TIMEFRAME = "M1"


async def build_symbol(client: DukascopyClient, symbol: str, days: int, out_dir) -> None:
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)

    frames = []
    skipped = 0
    for hour_start in utc_hour_range(start, end):
        try:
            tick_hour = await client.fetch_hour(symbol, hour_start)
        except DukascopyDownloadError:
            # One bad hour (out of ~720) isn't worth losing the whole
            # symbol's run for — note it and move on.
            skipped += 1
            continue
        if not tick_hour.ticks.empty:
            frames.append(resample_ticks_to_ohlc(tick_hour.ticks, BUNDLED_TIMEFRAME))

    if skipped:
        print(f"{symbol}: skipped {skipped} hour(s) after exhausting retries")

    import pandas as pd

    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(bundled_bar_path(out_dir, symbol, BUNDLED_TIMEFRAME), index=False)
    print(f"{symbol}: {len(combined)} bars written")


async def main(days: int) -> None:
    from pathlib import Path

    out_dir = Path(__file__).resolve().parent / "bundled"
    client = DukascopyClient()
    try:
        for symbol in BUNDLED_SYMBOLS:
            await build_symbol(client, symbol, days, out_dir)
    finally:
        await client.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    asyncio.run(main(args.days))
