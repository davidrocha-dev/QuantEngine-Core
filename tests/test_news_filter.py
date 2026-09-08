import json
from datetime import datetime, timedelta, timezone

from app.engine.news_filter import (
    EconomicEvent,
    apply_news_filter,
    parse_csv_fallback,
    parse_forex_factory_json,
)
from tests.helpers import make_synthetic_ohlc


def test_parse_forex_factory_json():
    raw = json.dumps([
        {"title": "NFP", "country": "USD", "date": "2024-01-05T13:30:00+00:00", "impact": "High"},
        {"title": "Minor thing", "country": "EUR", "date": "2024-01-05T09:00:00+00:00", "impact": "Low"},
    ])
    events = parse_forex_factory_json(raw)
    assert len(events) == 2
    assert events[0].impact == "high"


def test_parse_csv_fallback():
    raw = "timestamp,impact,currency,title\n2024-01-05T13:30:00+00:00,High,USD,NFP\n"
    events = parse_csv_fallback(raw)
    assert len(events) == 1
    assert events[0].currency == "USD"


def test_ftmo_mode_blocks_30_min_around_high_and_medium():
    df = make_synthetic_ohlc(48, seed=1)  # hourly bars
    event_time = df["timestamp"].iloc[24]
    events = [EconomicEvent(event_time, "high", "USD", "Test Event")]

    veto = apply_news_filter(df, events, "FTMO")

    assert veto.iloc[24] == True  # exactly at the event
    # 2 hours away is well outside the 30-min window
    assert veto.iloc[22] == False
    assert veto.iloc[26] == False


def test_strict_mode_wider_window_for_high_impact():
    df = make_synthetic_ohlc(48, seed=1)
    event_time = df["timestamp"].iloc[24]
    events = [EconomicEvent(event_time - timedelta(minutes=45), "high", "USD", "Test Event")]

    ftmo_veto = apply_news_filter(df, events, "FTMO")
    strict_veto = apply_news_filter(df, events, "Strict")

    # FTMO's 30-min window doesn't reach a bar 45 min before the event;
    # Strict's 60-min window does.
    assert ftmo_veto.iloc[24] == False
    assert strict_veto.iloc[24] == True


def test_low_impact_events_never_veto():
    df = make_synthetic_ohlc(48, seed=1)
    events = [EconomicEvent(df["timestamp"].iloc[24], "low", "USD", "Minor")]
    veto = apply_news_filter(df, events, "Strict")
    assert not veto.any()


def test_no_events_returns_all_false():
    df = make_synthetic_ohlc(20)
    veto = apply_news_filter(df, [], "FTMO")
    assert not veto.any()


def test_overlapping_event_windows_merge_correctly():
    df = make_synthetic_ohlc(48, seed=1)
    t0 = df["timestamp"].iloc[20]
    events = [
        EconomicEvent(t0, "high", "USD", "Event 1"),
        EconomicEvent(t0 + timedelta(minutes=50), "high", "EUR", "Event 2"),
    ]
    veto = apply_news_filter(df, events, "Strict")
    # Strict = 60 min window; the two events' windows overlap and should
    # merge into continuous coverage between them.
    assert veto.iloc[20] == True
    assert veto.iloc[21] == True
