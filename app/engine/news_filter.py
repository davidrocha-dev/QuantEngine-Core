"""News Filter gene (EVENT) — REQ-ENGINE-2.

FTMO mode blocks positions 30 min before/after any High or Medium impact
event. Strict mode blocks 60 min before/after High impact events, and 30
min before / 15 min after Medium impact events. Events are sourced from
Forex Factory's JSON calendar with a local CSV fallback when the feed is
unavailable.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

Impact = Literal["high", "medium", "low"]
NewsMode = Literal["FTMO", "Strict"]


@dataclass(frozen=True)
class EconomicEvent:
    timestamp: datetime
    impact: Impact
    currency: str
    title: str


# (before_minutes, after_minutes) blocking window per mode/impact.
_WINDOWS: dict[NewsMode, dict[Impact, tuple[int, int]]] = {
    "FTMO": {"high": (30, 30), "medium": (30, 30), "low": (0, 0)},
    "Strict": {"high": (60, 60), "medium": (30, 15), "low": (0, 0)},
}


def parse_forex_factory_json(raw: str) -> list[EconomicEvent]:
    """Parses the Forex Factory calendar JSON export. Expected shape:
    [{"title": str, "country": str, "date": ISO8601 str, "impact": "High"|"Medium"|"Low", ...}, ...]
    """
    events = []
    for row in json.loads(raw):
        impact = str(row.get("impact", "")).lower()
        if impact not in ("high", "medium", "low"):
            continue
        events.append(
            EconomicEvent(
                timestamp=datetime.fromisoformat(row["date"]).astimezone(timezone.utc),
                impact=impact,  # type: ignore[arg-type]
                currency=row.get("country", row.get("currency", "")),
                title=row.get("title", ""),
            )
        )
    return events


def parse_csv_fallback(raw: str) -> list[EconomicEvent]:
    """Local CSV fallback. Expected columns: timestamp,impact,currency,title."""
    events = []
    for row in csv.DictReader(raw.splitlines()):
        impact = row["impact"].strip().lower()
        if impact not in ("high", "medium", "low"):
            continue
        events.append(
            EconomicEvent(
                timestamp=datetime.fromisoformat(row["timestamp"]).astimezone(timezone.utc),
                impact=impact,  # type: ignore[arg-type]
                currency=row.get("currency", ""),
                title=row.get("title", ""),
            )
        )
    return events


def load_calendar(json_path: Path | None, csv_fallback_path: Path | None) -> list[EconomicEvent]:
    if json_path is not None and json_path.exists():
        try:
            return parse_forex_factory_json(json_path.read_text())
        except (json.JSONDecodeError, KeyError, ValueError):
            pass  # fall through to CSV fallback — REQ-ENGINE-2.3
    if csv_fallback_path is not None and csv_fallback_path.exists():
        return parse_csv_fallback(csv_fallback_path.read_text())
    return []


def _merge_windows(windows: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    if not windows:
        return []
    windows = sorted(windows, key=lambda w: w[0])
    merged = [windows[0]]
    for start, end in windows[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def apply_news_filter(df: pd.DataFrame, events: list[EconomicEvent], mode: NewsMode) -> pd.Series:
    """Returns a boolean veto Series aligned to `df` — True blocks entry."""
    if not events:
        return pd.Series(False, index=df.index)

    windows_by_mode = _WINDOWS[mode]
    raw_windows = []
    for event in events:
        before_min, after_min = windows_by_mode.get(event.impact, (0, 0))
        if before_min == 0 and after_min == 0:
            continue
        raw_windows.append((event.timestamp - timedelta(minutes=before_min), event.timestamp + timedelta(minutes=after_min)))

    merged = _merge_windows(raw_windows)
    if not merged:
        return pd.Series(False, index=df.index)

    starts = np.array([w[0].replace(tzinfo=None) for w in merged], dtype="datetime64[ns]")
    ends = np.array([w[1].replace(tzinfo=None) for w in merged], dtype="datetime64[ns]")
    bar_times = pd.DatetimeIndex(df["timestamp"]).tz_localize(None).values.astype("datetime64[ns]")

    idx = np.searchsorted(starts, bar_times, side="right") - 1
    valid = idx >= 0
    veto = np.zeros(len(bar_times), dtype=bool)
    veto[valid] = bar_times[valid] <= ends[idx[valid]]
    return pd.Series(veto, index=df.index)
