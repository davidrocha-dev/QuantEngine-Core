"""The broker's server clock, and history expressed in it.

Dukascopy history is UTC. MetaTrader shows every bar in the broker's
server time, and nearly every retail broker runs that clock on "New York
close": UTC+2 in the northern winter, UTC+3 in summer, switching with US
daylight saving, so that the trading day ends at 17:00 New York and every
D1 bar is a whole session with no Sunday stub.

Everything that asks what day or what hour it is has to ask the broker's
clock, or the backtest and the EA disagree:

  - H2, H4 and D1 bars start at the broker's boundaries, not UTC's - an
    H4 bar on a UTC+2 server opens at 22:00 UTC, not 00:00;
  - "today" (ORB, VWAP, DAILY MID, ADR EXHAUST) begins at the broker's
    midnight;
  - hours and weekdays (EOD CLOSE, FRIDAY PROFIT, NO FRI OPEN) are the
    hours the EA reads off its own bars.

Timestamps stay UTC everywhere - the news calendar, tick verification and
trade logs are all UTC - and each bar carries its broker wall-clock time
alongside, in the `broker_time` column. Genes that need a calendar read
that column (`bar_clock`).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

NY_CLOSE = "NY_CLOSE"
FIXED = "FIXED"
SERVER_TIME_MODES = (NY_CLOSE, FIXED)

# New York close is 17:00 New York, which is broker midnight.
_NY_CLOSE_SHIFT = pd.Timedelta(hours=7)


@dataclass(frozen=True)
class BrokerClock:
    mode: str = NY_CLOSE
    offset_hours: float = 0.0  # FIXED only: the server's offset from UTC

    def local(self, timestamps) -> pd.DatetimeIndex:
        """Broker wall-clock times (tz-naive) for UTC `timestamps`."""
        utc = pd.DatetimeIndex(timestamps)
        utc = utc.tz_localize("UTC") if utc.tz is None else utc.tz_convert("UTC")
        if self.mode == NY_CLOSE:
            return utc.tz_convert("America/New_York").tz_localize(None) + _NY_CLOSE_SHIFT
        return utc.tz_localize(None) + pd.Timedelta(hours=self.offset_hours)

    def describe(self) -> str:
        if self.mode == NY_CLOSE:
            return "New York close (UTC+2 / UTC+3 with US daylight saving)"
        return f"fixed UTC{self.offset_hours:+g}"


UTC_CLOCK = BrokerClock(FIXED, 0.0)
DEFAULT_CLOCK = BrokerClock(NY_CLOSE)


def clock_from_settings(settings: dict | None) -> BrokerClock:
    caps = (settings or {}).get("broker_caps", {})
    mode = caps.get("server_time", NY_CLOSE)
    if mode not in SERVER_TIME_MODES:
        mode = NY_CLOSE
    return BrokerClock(mode, float(caps.get("server_gmt_offset_hours", 0.0) or 0.0))


def bar_clock(df: pd.DataFrame) -> pd.DatetimeIndex:
    """The time each bar of `df` opened, on the broker's clock.

    Frames that were not loaded through the provider (tests, synthetic
    series) have no `broker_time` and are read as UTC, which is what they
    are.
    """
    if "broker_time" in df:
        return pd.DatetimeIndex(df["broker_time"])
    stamps = pd.DatetimeIndex(df["timestamp"])
    return stamps.tz_convert("UTC").tz_localize(None) if stamps.tz is not None else stamps


def broker_dates(df: pd.DataFrame) -> pd.Series:
    """The broker's trading day of every bar, as a Series aligned to `df`."""
    return pd.Series(bar_clock(df).normalize(), index=df.index)
