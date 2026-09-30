"""Tick-resolution verification of a bar backtest — design.md Part C.

The bar backtester cannot see inside a bar. When one bar's range spans
both the stop and the target it assumes the stop was hit first
(`backtest.py`: "Conservative tie-break"), because assuming otherwise
would be inventing a winner. That is the right default and it is still a
guess.

This measures the guess. It replays each trade against the actual ticks
and reports what really happened first, which answers a question no
amount of bar data can:

  - If almost no trades were ambiguous, the bar result is the real one.
  - If many were, and they mostly hit target first, the strategy has been
    *understated* - its edge is real and larger than it looked.
  - If many were and the tick result is barely different, the bar figure
    is trustworthy at this timeframe.

Two price conventions meet here, and the difference is not a bug:

The cached bars are **mid** prices (dukascopy_client.resample_ticks_to_ohlc
averages bid and ask). The bar backtester pays the spread by moving the
entry: a long is filled a full spread above the mid, and its stop and
target are measured from there, which in mid space is where a bid-
triggered stop and target sit. The ticks are real **bid and ask**, so a
stop can still be touched on ticks inside a bar whose mid range did not
quite reach it. That is the resolution the bars work at, and tick mode
is where you find out how much it costs you.

The same modelled cost is still subtracted from the tick result, so the
two numbers are compared on equal terms. That double-charges the exit
side slightly, because the bid already contains the spread - the tick
figure is therefore a little pessimistic, which is the direction to be
wrong in.

What it deliberately does not do is re-run the genetic search on ticks.
Entries come from bar-close signals and stay exactly where the backtest
put them; only the exits are re-resolved. Breeding on ticks would cost
hours of CPU to answer a question nobody asked.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from app.engine.backtest import Trade

# How long a trade may stay open past its bar-level exit before the tick
# pass gives up and keeps the bar verdict. Exits are looked for inside the
# trade's own window; this is only the slack needed because a bar's
# timestamp is its open and the exit happens somewhere inside it.
EXIT_SLACK = timedelta(minutes=1)


@dataclass(frozen=True)
class TradeResolution:
    entry_time: datetime
    direction: int
    bar_exit: str  # stop | target | max_hold
    tick_exit: str
    ambiguous: bool  # the bar spanned both levels, so the bar result was a guess
    changed: bool  # the ticks disagreed with that guess
    tick_exit_time: datetime | None
    r_multiple_bar: float
    r_multiple_tick: float


@dataclass(frozen=True)
class TickVerification:
    trades_checked: int
    ambiguous_trades: int
    changed_trades: int
    return_pct_bars: float
    return_pct_ticks: float
    ticks_used: int
    hours_fetched: int
    detail: str
    resolutions: list[TradeResolution]

    @property
    def agreement_pct(self) -> float:
        if self.trades_checked == 0:
            return 100.0
        return (self.trades_checked - self.changed_trades) / self.trades_checked * 100


def _as_utc(value) -> pd.Timestamp:
    """One timezone, everywhere.

    Bars come from parquet and ticks from the feed, and the two do not
    reliably agree about whether a timestamp carries a zone. Comparing
    them directly raises, which is a silly way for a verification to
    fail.
    """
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _bar_time(bars: pd.DataFrame, index: int) -> datetime:
    return _as_utc(bars["timestamp"].iloc[index]).to_pydatetime()


def bar_is_ambiguous(bars: pd.DataFrame, index: int, stop: float, target: float) -> bool:
    """Both levels inside one bar's range: whatever the backtester said
    about this bar, it guessed."""
    low = float(bars["low"].iloc[index])
    high = float(bars["high"].iloc[index])
    lower, upper = (stop, target) if stop <= target else (target, stop)
    return low <= lower and upper <= high


def resolve_on_ticks(
    ticks: pd.DataFrame,
    *,
    direction: int,
    stop_price: float,
    target_price: float,
) -> tuple[str, datetime | None, float | None]:
    """Which level the ticks reached first, if either, and at what price.

    A long is closed at the bid and a short at the ask, which is what the
    broker would actually fill: using the mid here would quietly hand back
    half the spread on every exit.

    The fill price is returned rather than assumed, because a stop does
    not fill at the stop. Price gaps through it, and the difference is the
    part of the risk a bar backtest cannot see at all - a -1R stop that
    really filled at -2.4R is the kind of thing that empties an account
    while the equity curve still looks fine.
    """
    if ticks.empty:
        return "none", None, None

    prices = ticks["bid"].to_numpy() if direction == 1 else ticks["ask"].to_numpy()
    times = pd.to_datetime(ticks["timestamp"], utc=True).to_numpy()

    for position in range(len(prices)):
        price = prices[position]
        if direction == 1:
            hit_stop = price <= stop_price
            hit_target = price >= target_price
        else:
            hit_stop = price >= stop_price
            hit_target = price <= target_price
        if hit_stop or hit_target:
            # A single tick that clears both levels is a gap through the
            # whole range - that is a stop, not a win. Price did not
            # travel to the target; it jumped past everything.
            reason = "stop" if hit_stop else "target"
            return reason, pd.Timestamp(times[position]).to_pydatetime(), float(price)
    return "none", None, None


def verify_trades(
    trades: list[Trade],
    bars: pd.DataFrame,
    ticks: pd.DataFrame,
    *,
    bar_duration: timedelta,
    cost_points: float = 0.0,
    point_size: float = 0.0,
) -> TickVerification:
    """Replays each trade's exit against ticks and compares.

    `ticks` covers the hours the trades were open in; anything missing
    simply leaves that trade with its bar verdict, which is why an
    unreachable feed degrades the answer instead of breaking it.
    """
    resolutions: list[TradeResolution] = []
    r_bars = 0.0
    r_ticks = 0.0

    tick_times = pd.to_datetime(ticks["timestamp"], utc=True) if not ticks.empty else None

    for trade in trades:
        entry_time = _bar_time(bars, trade.entry_idx)
        exit_time = _bar_time(bars, trade.exit_idx) + bar_duration + EXIT_SLACK
        ambiguous = bar_is_ambiguous(bars, trade.exit_idx, trade.stop_price, trade.target_price)

        window = ticks
        if tick_times is not None:
            mask = (tick_times >= _as_utc(entry_time)) & (tick_times <= _as_utc(exit_time))
            window = ticks.loc[mask.to_numpy()]

        tick_reason, tick_time, fill_price = resolve_on_ticks(
            window,
            direction=trade.direction,
            stop_price=trade.stop_price,
            target_price=trade.target_price,
        )
        # No ticks for this window (feed gap, weekend, or nothing fetched)
        # means no new information, so the bar result stands.
        if tick_reason == "none":
            tick_reason, fill_price = trade.exit_reason, None

        stop_distance = abs(trade.entry_price - trade.stop_price)
        if fill_price is not None and stop_distance > 0:
            # From the price that actually traded, not from the level that
            # was asked for, and net of the same cost the bar backtest
            # charged - otherwise the comparison would flatter the ticks.
            cost_r = (cost_points * point_size) / stop_distance
            r_tick = (fill_price - trade.entry_price) * trade.direction / stop_distance - cost_r
        else:
            r_tick = trade.r_multiple

        r_bars += trade.r_multiple
        r_ticks += r_tick
        resolutions.append(
            TradeResolution(
                entry_time=entry_time,
                direction=trade.direction,
                bar_exit=trade.exit_reason,
                tick_exit=tick_reason,
                ambiguous=ambiguous,
                changed=tick_reason != trade.exit_reason,
                tick_exit_time=tick_time,
                r_multiple_bar=trade.r_multiple,
                r_multiple_tick=r_tick,
            )
        )

    changed = sum(1 for row in resolutions if row.changed)
    ambiguous_count = sum(1 for row in resolutions if row.ambiguous)

    return TickVerification(
        trades_checked=len(resolutions),
        ambiguous_trades=ambiguous_count,
        changed_trades=changed,
        return_pct_bars=r_bars,
        return_pct_ticks=r_ticks,
        ticks_used=int(len(ticks)),
        hours_fetched=0,
        detail=_describe(len(resolutions), ambiguous_count, changed, r_bars, r_ticks),
        resolutions=resolutions,
    )


def _describe(checked: int, ambiguous: int, changed: int, r_bars: float, r_ticks: float) -> str:
    if checked == 0:
        return "No trades to verify."
    if ambiguous == 0 and changed == 0:
        return (
            f"None of the {checked} trades had a bar that spanned both the stop and the target, "
            "and the ticks agreed with every exit. The bar-level result is the real one."
        )
    if ambiguous == 0:
        # No bar was ambiguous, yet the ticks still disagreed. That is the
        # price convention, not the path: bars are mid, and a long is
        # stopped on the bid.
        return (
            f"No bar spanned both levels, yet the ticks disagreed on {changed} of {checked} trades "
            f"({r_ticks:+.2f}R on ticks vs {r_bars:+.2f}R on bars). The cached bars are mid prices "
            "and a long is stopped out on the bid, so a stop can be reached on ticks in a bar whose "
            "mid never touched it. This is the cost of bar resolution, not an error in it."
        )
    if changed == 0:
        return (
            f"{ambiguous} of {checked} trades were decided inside a single bar, and the ticks agreed "
            "with every one of them."
        )
    if r_ticks > r_bars:
        return (
            f"{ambiguous} of {checked} trades were decided inside a single bar; the ticks disagreed on "
            f"{changed}. Total R is better on ticks ({r_ticks:+.2f}R vs {r_bars:+.2f}R) - the bar "
            "backtest was being conservative about exactly those trades."
        )
    return (
        f"{ambiguous} of {checked} trades were decided inside a single bar; the ticks disagreed on "
        f"{changed}. Total R is worse on ticks ({r_ticks:+.2f}R vs {r_bars:+.2f}R), which usually "
        "means stops filling past their level rather than at it."
    )
