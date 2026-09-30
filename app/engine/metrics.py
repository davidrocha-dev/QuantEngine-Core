"""Backtest quality metrics — feeds Phase 0 screening and the F1/F2 fitness
formulas (REQ-ENGINE-3), plus ICIR/AVG HOLD (REQ-ENGINE-3.9).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.data.broker_clock import bar_clock
from app.engine.backtest import BacktestResult, Trade

TRADING_DAYS_PER_YEAR = 252
MAX_REPORTED_CAGR_PCT = 1000.0
# Below this much history, annualising is extrapolation dressed as a
# measurement. Half a year is already generous for the claim "per year".
MIN_YEARS_TO_ANNUALISE = 0.5
# A perfectly steady edge divides by a spread of zero. The ratio is
# unbounded, not absent, and "unbounded" has to sort somewhere: at the
# end it belongs, which is the top for a winner and the bottom for a
# loser. Same treatment an infinite profit factor gets in fitness.py.
MAX_REPORTED_ICIR = 10.0
BARS_PER_YEAR_BY_TIMEFRAME = {
    "M1": 252 * 24 * 60, "M5": 252 * 24 * 12, "M15": 252 * 24 * 4, "M30": 252 * 24 * 2,
    "H1": 252 * 24, "H2": 252 * 12, "H4": 252 * 6, "D1": 252,
}


@dataclass
class TradeStats:
    trade_count: int
    profit_factor: float
    win_rate: float
    sharpe_ratio: float
    max_dd_pct: float
    net_profit_pct: float
    cagr_pct: float | None
    payoff_ratio: float
    icir: float
    avg_hold_hours: float
    # The worst single day. A different rule from max_dd_pct, and the one
    # a prop firm closes an account over. Last and defaulted so callers
    # that build a TradeStats positionally keep working.
    max_daily_dd_pct: float = 0.0
    # System Quality Number: mean R over its spread, scaled by the square
    # root of the sample (capped at 100 trades). An edge per trade, and
    # how much evidence there is for it - what win rate was standing in
    # for, without rewarding a 1R stop against a 0.38R target.
    sqn: float = 0.0


def profit_factor(trades: list[Trade]) -> float:
    gains = sum(t.pnl for t in trades if t.pnl > 0)
    losses = -sum(t.pnl for t in trades if t.pnl < 0)
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def win_rate(trades: list[Trade]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.pnl > 0)
    return wins / len(trades)


def payoff_ratio(trades: list[Trade]) -> float:
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [-t.pnl for t in trades if t.pnl < 0]
    if not wins or not losses:
        return 0.0
    return (sum(wins) / len(wins)) / (sum(losses) / len(losses))


def max_drawdown_pct(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    return float(-drawdown.min() * 100)


def max_daily_drawdown_pct(equity_curve: np.ndarray, timestamps: pd.DatetimeIndex | None) -> float:
    """The worst single day, peak-to-trough, as a share of that day's start.

    Total drawdown and daily drawdown are different rules and a strategy
    can pass one while failing the other. Prop firms enforce both — a
    typical account allows 10% overall and 5% in any one day — and this
    engine only ever measured the first, so a strategy that loses 6% in
    an afternoon and recovers it looked flawless right up to the moment
    the account was closed.

    Measured against the day's opening equity rather than the running
    peak, which is how the daily rule is usually written: the allowance
    resets at the session boundary.
    """
    if timestamps is None or len(timestamps) != len(equity_curve) or len(equity_curve) < 2:
        return 0.0
    series = pd.Series(equity_curve, index=pd.DatetimeIndex(timestamps))
    worst = 0.0
    for _, day in series.groupby(series.index.date):
        if len(day) < 2 or day.iloc[0] <= 0:
            continue
        trough = float(day.min())
        drop = (day.iloc[0] - trough) / day.iloc[0] * 100
        worst = max(worst, float(drop))
    return worst


def net_profit_pct(equity_curve: np.ndarray, initial_balance: float) -> float:
    if len(equity_curve) == 0:
        return 0.0
    return float((equity_curve[-1] - initial_balance) / initial_balance * 100)


def cagr_pct(
    equity_curve: np.ndarray,
    initial_balance: float,
    bar_count: int,
    timeframe: str,
    timestamps: pd.DatetimeIndex | None = None,
) -> float | None:
    """Annualised growth rate, or None when the window is too short to
    annualise honestly.

    Compounding a six-week result up to a year is arithmetic, not
    forecasting: a good month becomes four figures a year and a great one
    becomes millions of percent. Clamping that to a large finite number
    was worse than useless - it turned nonsense into something that
    looked like a measurement and sorted like one.

    So a window under `MIN_YEARS_TO_ANNUALISE` reports nothing at all.
    "We cannot say" is a fact; "+27,929,074,494,351%/yr" is not.
    """
    if len(equity_curve) == 0 or bar_count == 0:
        return 0.0
    # A blown account used to land in the guard above and report 0.0% -
    # the same figure as a strategy that did nothing, and a better one
    # than a strategy that lost half. It also made the -100.0 branch
    # further down unreachable, because nothing else can drive the final
    # equity to zero or below. Losing everything is not an extrapolation,
    # so it is reported whatever the window length.
    if equity_curve[-1] <= 0:
        return -100.0
    # Elapsed calendar time, because a year is 365 days however many of
    # them the market was open. The bar-count fallback below assumes
    # 252 x 24 hours of trading a year, which on a market that closes at
    # weekends undercounts the span it is standing in for - measured on a
    # real out-of-sample window of 183 days, 2026-03-25 to 2026-09-24, it
    # reported 0.4939 years against the calendar's 0.5014 and hid a CAGR
    # that had earned the right to be shown by six days.
    if timestamps is not None and len(timestamps) >= 2:
        elapsed_days = (timestamps.max() - timestamps.min()).total_seconds() / 86_400
        years = elapsed_days / 365.0
    else:
        bars_per_year = BARS_PER_YEAR_BY_TIMEFRAME.get(timeframe, TRADING_DAYS_PER_YEAR)
        years = bar_count / bars_per_year
    if years <= 0:
        return 0.0
    if years < MIN_YEARS_TO_ANNUALISE:
        return None
    total_return = equity_curve[-1] / initial_balance
    if total_return <= 0:
        return -100.0
    # Even over a long enough window, a runaway compounding figure helps
    # nobody read a table; the cap keeps it sortable.
    return float(min(max((total_return ** (1 / years) - 1) * 100, -100.0), MAX_REPORTED_CAGR_PCT))


def sharpe_ratio(equity_curve: np.ndarray, timestamps: pd.DatetimeIndex | None = None) -> float:
    """Annualised Sharpe on **daily** returns.

    This used to run on bar-by-bar returns annualised by bars-per-year.
    That is defensible arithmetic and it is not the number anyone else
    means by "Sharpe": a strategy holding one position at a time leaves
    the equity curve flat through most bars, so the series being measured
    was mostly zeros, and the figure was not comparable to a Sharpe quoted
    anywhere else in the industry.

    Daily closes, annualised by sqrt(252), is the convention. It is also
    what `portfolio.daily_returns` already used for correlation, so the
    two halves of the product now measure risk the same way.
    """
    if len(equity_curve) < 3:
        return 0.0

    if timestamps is None or len(timestamps) != len(equity_curve):
        # No calendar to resample against: fall back to per-observation
        # returns rather than inventing dates.
        returns = np.diff(equity_curve) / equity_curve[:-1]
    else:
        # Equity is a step function between trades, so the last value of
        # each day is that day's close.
        daily = pd.Series(equity_curve, index=pd.DatetimeIndex(timestamps)).resample("1D").last().dropna()
        if len(daily) < 3:
            return 0.0
        returns = daily.pct_change().dropna().to_numpy()

    returns = returns[np.isfinite(returns)]
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def information_coefficient_information_ratio(trades: list[Trade]) -> float:
    """ICIR — how much edge per trade, and how steadily.

    The mean of a rolling window of realised R divided by the standard
    deviation of those windows: positive when the average trade makes
    money, and larger when it does so consistently rather than through a
    lucky few. That is what "high and stable across trades beats a lucky
    few big winners" was always meant to say.

    It used to reduce each trade to the *sign* of its R before averaging,
    which makes the statistic (2 x win_rate - 1) / spread - a number that
    cannot be positive unless the strategy wins more often than it loses.
    Measured across a real Vault: 25 strategies, 25 negative ICIRs, a
    perfect monotone in win rate (15.8% -> -4.56, 40.1% -> -0.94) and no
    relation to whether the strategy made money. These are high-payoff,
    low-frequency-of-winning strategies; one returning +123.9% was
    reported at -4.56 and the Promotion Verdict said "ICIR computed",
    which is true and is not a test.
    """
    if len(trades) < 5:
        return 0.0
    r_multiples = np.array([t.r_multiple for t in trades], dtype=float)
    r_multiples = r_multiples[np.isfinite(r_multiples)]
    if len(r_multiples) < 5:
        return 0.0
    window = min(20, len(r_multiples))
    rolling = pd.Series(r_multiples).rolling(window).mean().dropna()
    if len(rolling) < 2:
        return 0.0
    spread = float(rolling.std())
    mean = float(rolling.mean())
    if spread == 0:
        # Every window identical: no variability to divide by. Zero would
        # rank the steadiest edge alongside no edge at all.
        return 0.0 if mean == 0 else math.copysign(MAX_REPORTED_ICIR, mean)
    return float(max(-MAX_REPORTED_ICIR, min(MAX_REPORTED_ICIR, mean / spread)))


SQN_SAMPLE_CAP = 100
MAX_REPORTED_SQN = 10.0


def system_quality_number(trades: list[Trade]) -> float:
    """Van Tharp's SQN: mean(R) / std(R) x sqrt(min(N, 100)).

    Rewards a steady edge per trade over a lucky few big ones, and a large
    sample over a small one, and is negative for a strategy that loses on
    average. The sample is capped so a strategy cannot score by trading
    more rather than better.
    """
    r = np.array([t.r_multiple for t in trades], dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    spread = float(r.std(ddof=1))
    mean = float(r.mean())
    if spread == 0:
        return 0.0 if mean == 0 else math.copysign(MAX_REPORTED_SQN, mean)
    value = mean / spread * math.sqrt(min(len(r), SQN_SAMPLE_CAP))
    return float(max(-MAX_REPORTED_SQN, min(MAX_REPORTED_SQN, value)))


def avg_hold_hours(trades: list[Trade], df: pd.DataFrame) -> float:
    if not trades:
        return 0.0
    timestamps = pd.DatetimeIndex(df["timestamp"])
    durations = [
        (timestamps[t.exit_idx] - timestamps[t.entry_idx]).total_seconds() / 3600
        for t in trades
        if t.exit_idx < len(timestamps)
    ]
    return float(np.mean(durations)) if durations else 0.0


def compute_trade_stats(result: BacktestResult, df: pd.DataFrame, initial_balance: float, timeframe: str) -> TradeStats:
    bars_per_year = BARS_PER_YEAR_BY_TIMEFRAME.get(timeframe, TRADING_DAYS_PER_YEAR)
    return TradeStats(
        trade_count=len(result.trades),
        profit_factor=profit_factor(result.trades),
        win_rate=win_rate(result.trades),
        sharpe_ratio=sharpe_ratio(result.equity_curve, pd.DatetimeIndex(df["timestamp"]) if "timestamp" in df else None),
        max_dd_pct=max_drawdown_pct(result.equity_curve),
        # The broker's day: prop firms reset the daily allowance at server
        # midnight, not UTC's.
        max_daily_dd_pct=max_daily_drawdown_pct(
            result.equity_curve, bar_clock(df) if "timestamp" in df else None
        ),
        net_profit_pct=net_profit_pct(result.equity_curve, initial_balance),
        cagr_pct=cagr_pct(
            result.equity_curve, initial_balance, len(df), timeframe,
            pd.DatetimeIndex(df["timestamp"]) if "timestamp" in df else None,
        ),
        payoff_ratio=payoff_ratio(result.trades),
        icir=information_coefficient_information_ratio(result.trades),
        avg_hold_hours=avg_hold_hours(result.trades, df),
        sqn=system_quality_number(result.trades),
    )
