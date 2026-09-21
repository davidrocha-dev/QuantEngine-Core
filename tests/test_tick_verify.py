"""Tick-resolution verification — design.md Part C.

The bar backtester assumes the stop was hit first whenever one bar spans
both levels. These tests are about what happens when that assumption is
wrong, and about never letting the tick pass be more optimistic than the
ticks actually justify.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app.data.ticks import concat_hours, hours_covering
from app.engine.backtest import Trade
from app.engine.tick_verify import bar_is_ambiguous, resolve_on_ticks, verify_trades

M15 = timedelta(minutes=15)
START = datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)


def make_bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """(open, high, low, close) per M15 bar, starting at START."""
    return pd.DataFrame(
        {
            "timestamp": [START + index * M15 for index in range(len(rows))],
            "open": [row[0] for row in rows],
            "high": [row[1] for row in rows],
            "low": [row[2] for row in rows],
            "close": [row[3] for row in rows],
            "volume": [1.0] * len(rows),
        }
    )


def make_ticks(prices: list[tuple[int, float]], spread: float = 0.00002) -> pd.DataFrame:
    """(seconds after START, bid) - the ask sits one spread above."""
    return pd.DataFrame(
        {
            "timestamp": [START + timedelta(seconds=offset) for offset, _ in prices],
            "bid": [bid for _, bid in prices],
            "ask": [bid + spread for _, bid in prices],
            "bid_volume": [1.0] * len(prices),
            "ask_volume": [1.0] * len(prices),
        }
    )


def long_trade(entry=1.0800, stop=1.0790, target=1.0820, exit_reason="stop", r=-1.0) -> Trade:
    return Trade(
        entry_idx=0, exit_idx=1, direction=1, entry_price=entry,
        exit_price=stop if exit_reason == "stop" else target,
        stop_price=stop, target_price=target, r_multiple=r, pnl=r * 100,
        exit_reason=exit_reason,
    )


# --- the ambiguity the whole feature exists for ---------------------------


def test_a_bar_spanning_both_levels_is_flagged_as_a_guess():
    bars = make_bars([(1.0800, 1.0805, 1.0798, 1.0802), (1.0802, 1.0825, 1.0785, 1.0790)])

    assert bar_is_ambiguous(bars, 1, stop=1.0790, target=1.0820) is True
    # A bar that only reached the target leaves nothing to guess about.
    assert bar_is_ambiguous(bars, 0, stop=1.0790, target=1.0820) is False


def test_the_ticks_can_overturn_the_conservative_guess():
    # The bar says stop, because its range covered both. The ticks say the
    # price went up to the target first and only then came back down.
    bars = make_bars([(1.0800, 1.0805, 1.0798, 1.0802), (1.0802, 1.0825, 1.0785, 1.0790)])
    ticks = make_ticks([(0, 1.0800), (900, 1.0810), (960, 1.0821), (1100, 1.0789)])

    result = verify_trades([long_trade()], bars, ticks, bar_duration=M15)

    [resolution] = result.resolutions
    assert resolution.bar_exit == "stop"
    assert resolution.tick_exit == "target"
    assert resolution.changed is True
    assert result.return_pct_ticks > result.return_pct_bars
    assert "conservative" in result.detail


def test_the_ticks_can_confirm_it_too():
    bars = make_bars([(1.0800, 1.0805, 1.0798, 1.0802), (1.0802, 1.0825, 1.0785, 1.0790)])
    ticks = make_ticks([(0, 1.0800), (800, 1.0789), (900, 1.0821)])  # stop really was first

    result = verify_trades([long_trade()], bars, ticks, bar_duration=M15)

    assert result.changed_trades == 0
    assert result.ambiguous_trades == 1
    assert result.agreement_pct == 100.0
    assert "agreed with every one" in result.detail


def test_a_stop_that_gaps_costs_more_than_one_r():
    # The part of the risk a bar backtest cannot see. The stop was at
    # 1.0790 - one R below a 1.0800 entry - and the next tick printed at
    # 1.0700, which is where the position actually closed.
    bars = make_bars([(1.0800, 1.0805, 1.0798, 1.0802), (1.0802, 1.0803, 1.0690, 1.0700)])
    ticks = make_ticks([(0, 1.0800), (900, 1.0700)])

    result = verify_trades([long_trade()], bars, ticks, bar_duration=M15)

    [resolution] = result.resolutions
    assert resolution.tick_exit == "stop"
    assert resolution.r_multiple_bar == -1.0
    assert resolution.r_multiple_tick == pytest.approx(-10.0, abs=0.1)
    assert result.return_pct_ticks < result.return_pct_bars


def test_a_single_tick_cannot_reach_a_target_it_gapped_past():
    # For a long the target is above and the stop below, so one tick can
    # never satisfy both. Whichever it breached is what happened.
    ticks = make_ticks([(0, 1.0800), (60, 1.0700)])

    reason, _, fill = resolve_on_ticks(ticks, direction=1, stop_price=1.0790, target_price=1.0820)

    assert reason == "stop"
    assert fill == 1.0700


# --- honesty about what the ticks do not say ------------------------------


def test_no_ticks_for_a_trade_leaves_its_bar_verdict_alone():
    # A feed gap is not evidence. The alternative - defaulting to the
    # target - would turn missing data into profit.
    bars = make_bars([(1.0800, 1.0805, 1.0798, 1.0802), (1.0802, 1.0825, 1.0785, 1.0790)])

    result = verify_trades([long_trade()], bars, pd.DataFrame(columns=["timestamp", "bid", "ask"]), bar_duration=M15)

    assert result.resolutions[0].tick_exit == "stop"
    assert result.changed_trades == 0
    assert result.return_pct_ticks == result.return_pct_bars


def test_ticks_outside_the_trade_window_are_ignored():
    # Price hit the target an hour after the trade had already closed.
    bars = make_bars([(1.0800, 1.0805, 1.0798, 1.0802), (1.0802, 1.0825, 1.0785, 1.0790)])
    ticks = make_ticks([(0, 1.0800), (800, 1.0789), (7200, 1.0900)])

    result = verify_trades([long_trade()], bars, ticks, bar_duration=M15)

    assert result.resolutions[0].tick_exit == "stop"


def test_a_short_is_resolved_on_the_ask():
    # Exiting a short means buying, and you buy at the ask. Checking the
    # bid would hand back the spread on every exit.
    short = Trade(
        entry_idx=0, exit_idx=1, direction=-1, entry_price=1.0800, exit_price=1.0810,
        stop_price=1.0810, target_price=1.0780, r_multiple=-1.0, pnl=-100.0, exit_reason="stop",
    )
    bars = make_bars([(1.0800, 1.0805, 1.0798, 1.0802), (1.0802, 1.0815, 1.0775, 1.0810)])
    # Bid reaches 1.07805, but the ask never gets to the 1.0780 target.
    ticks = make_ticks([(0, 1.0800), (600, 1.07805), (900, 1.08105)], spread=0.00002)

    result = verify_trades([short], bars, ticks, bar_duration=M15)

    assert result.resolutions[0].tick_exit == "stop"


def test_an_unambiguous_run_says_so_plainly():
    bars = make_bars([(1.0800, 1.0805, 1.0798, 1.0802), (1.0802, 1.0825, 1.0801, 1.0820)])
    trade = long_trade(exit_reason="target", r=2.0)
    ticks = make_ticks([(0, 1.0800), (900, 1.0821)])

    result = verify_trades([trade], bars, ticks, bar_duration=M15)

    assert result.ambiguous_trades == 0
    assert result.changed_trades == 0
    assert "the real one" in result.detail


def test_nothing_to_verify_is_not_an_error():
    result = verify_trades([], make_bars([(1.08, 1.08, 1.08, 1.08)]), pd.DataFrame(), bar_duration=M15)

    assert result.trades_checked == 0
    assert result.agreement_pct == 100.0


# --- fetching only what is needed ----------------------------------------


def test_only_the_hours_the_trades_touch_are_requested():
    # The point of the whole design: a year has 8,760 hours and a
    # strategy's trades are open in a few hundred of them.
    windows = [
        (datetime(2026, 3, 2, 9, 30, tzinfo=timezone.utc), datetime(2026, 3, 2, 11, 5, tzinfo=timezone.utc)),
        (datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc), datetime(2026, 3, 2, 10, 30, tzinfo=timezone.utc)),
    ]

    hours = hours_covering(windows)

    assert hours == [
        datetime(2026, 3, 2, 9, tzinfo=timezone.utc),
        datetime(2026, 3, 2, 10, tzinfo=timezone.utc),
        datetime(2026, 3, 2, 11, tzinfo=timezone.utc),
    ]


def test_a_backwards_window_is_still_covered():
    windows = [(datetime(2026, 3, 2, 12, tzinfo=timezone.utc), datetime(2026, 3, 2, 10, tzinfo=timezone.utc))]

    assert len(hours_covering(windows)) == 3


@pytest.mark.asyncio
async def test_an_hour_is_never_downloaded_twice(tmp_path, monkeypatch):
    from app.config import EngineConfig
    from app.data.ticks import TickStore

    calls: list[datetime] = []

    class CountingClient:
        async def fetch_hour(self, symbol, hour):
            from app.data.dukascopy_client import TickHour

            calls.append(hour)
            return TickHour(symbol, hour, make_ticks([(0, 1.08)]))

        async def aclose(self):
            return None

    store = TickStore(EngineConfig(data_dir=tmp_path / "data"), client=CountingClient())
    hour = datetime(2026, 3, 2, 9, tzinfo=timezone.utc)

    first = await store.ensure("EURUSD", [hour])
    second = await store.ensure("EURUSD", [hour])

    assert calls == [hour]  # the second call came off the disk
    assert len(first[hour]) == len(second[hour]) == 1


@pytest.mark.asyncio
async def test_an_empty_hour_is_remembered_as_empty(tmp_path):
    # Weekends and holidays have no ticks. Not caching that would mean
    # re-asking the feed for them on every verification.
    from app.config import EngineConfig
    from app.data.dukascopy_client import TickHour
    from app.data.ticks import EMPTY_TICKS, TickStore

    calls = []

    class EmptyClient:
        async def fetch_hour(self, symbol, hour):
            calls.append(hour)
            return TickHour(symbol, hour, EMPTY_TICKS.copy())

        async def aclose(self):
            return None

    store = TickStore(EngineConfig(data_dir=tmp_path / "data"), client=EmptyClient())
    hour = datetime(2026, 3, 7, 3, tzinfo=timezone.utc)  # a Saturday

    await store.ensure("EURUSD", [hour])
    await store.ensure("EURUSD", [hour])

    assert len(calls) == 1
    assert store.cached("EURUSD", hour) is True


def test_hours_are_stitched_into_one_ordered_frame():
    hours = {
        datetime(2026, 3, 2, 10, tzinfo=timezone.utc): make_ticks([(3600, 1.0810)]),
        datetime(2026, 3, 2, 9, tzinfo=timezone.utc): make_ticks([(0, 1.0800)]),
    }

    combined = concat_hours(hours)

    assert list(combined["bid"]) == [1.0800, 1.0810]
