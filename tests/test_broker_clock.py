"""History on the broker's clock - the clock MetaTrader shows the EA."""

import pandas as pd
import pytest

from app.data.broker_clock import DEFAULT_CLOCK, UTC_CLOCK, BrokerClock, bar_clock, clock_from_settings
from app.data.provider import resample_bars


def _m1(start: str, minutes: int) -> pd.DataFrame:
    stamps = pd.date_range(start, periods=minutes, freq="1min", tz="UTC")
    n = len(stamps)
    return pd.DataFrame({
        "timestamp": stamps, "open": [1.0] * n, "high": [1.1] * n, "low": [0.9] * n,
        "close": [1.0] * n, "volume": [1.0] * n,
    })


def test_new_york_close_is_utc_plus_two_in_winter_and_three_in_summer():
    winter = DEFAULT_CLOCK.local(pd.DatetimeIndex(["2024-01-15 22:00"], tz="UTC"))[0]
    summer = DEFAULT_CLOCK.local(pd.DatetimeIndex(["2024-07-15 21:00"], tz="UTC"))[0]
    assert winter == pd.Timestamp("2024-01-16 00:00")
    assert summer == pd.Timestamp("2024-07-16 00:00")


def test_h4_bars_open_on_the_brokers_boundaries():
    # Winter, New York close: server 00:00 is 22:00 UTC the day before.
    h4 = resample_bars(_m1("2024-01-15 20:00", 8 * 60), "H4", DEFAULT_CLOCK)
    assert list(h4["broker_time"].dt.hour) == [20, 0, 4]
    assert h4["timestamp"].iloc[1] == pd.Timestamp("2024-01-15 22:00", tz="UTC")


def test_d1_has_no_sunday_stub_on_a_new_york_close_feed():
    # FX reopens Sunday 22:00 UTC in winter: on UTC days that is a two-hour
    # "Sunday"; on the broker's clock it is the start of Monday.
    sunday_open = _m1("2024-01-14 22:00", 24 * 60)
    utc_days = resample_bars(sunday_open, "D1", UTC_CLOCK)
    broker_days = resample_bars(sunday_open, "D1", DEFAULT_CLOCK)
    assert pd.DatetimeIndex(utc_days["timestamp"]).dayofweek[0] == 6
    assert list(pd.DatetimeIndex(broker_days["broker_time"]).dayofweek) == [0]


def test_m1_bars_carry_their_broker_time():
    m1 = resample_bars(_m1("2024-07-01 12:00", 3), "M1", BrokerClock("FIXED", 3))
    assert m1["broker_time"].iloc[0] == pd.Timestamp("2024-07-01 15:00")
    assert bar_clock(m1)[0].hour == 15


def test_frames_without_a_broker_time_read_as_utc():
    frame = _m1("2024-07-01 12:00", 1)
    assert bar_clock(frame)[0] == pd.Timestamp("2024-07-01 12:00")


def test_settings_choose_the_clock():
    assert clock_from_settings({"broker_caps": {"server_time": "NY_CLOSE"}}) == DEFAULT_CLOCK
    fixed = clock_from_settings({"broker_caps": {"server_time": "FIXED", "server_gmt_offset_hours": 3}})
    assert fixed.mode == "FIXED" and fixed.offset_hours == pytest.approx(3)


def test_daily_mid_uses_the_previous_broker_day():
    from app.genome.genes import BIAS_GENES

    # Two broker days (New York close, winter): day one trades at 1.0-2.0,
    # day two sits at 1.6 - above day one's mid of 1.5.
    day_one = _m1("2024-01-15 22:00", 24 * 60)
    day_one[["low", "high"]] = [1.0, 2.0]
    day_two = _m1("2024-01-16 22:00", 60)
    day_two[["open", "high", "low", "close"]] = 1.6
    h1 = resample_bars(pd.concat([day_one, day_two], ignore_index=True), "H1", DEFAULT_CLOCK)
    bias = BIAS_GENES["DAILY MID"].compute(h1, None, {})
    assert bias.iloc[-1] == 1
