"""Backtest quality metrics — feeds Phase 0 screening and the F1/F2 fitness
formulas (REQ-ENGINE-3), plus ICIR/AVG HOLD (REQ-ENGINE-3.9).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.engine.backtest import BacktestResult, Trade

TRADING_DAYS_PER_YEAR = 252
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
    cagr_pct: float
    payoff_ratio: float
    icir: float
    avg_hold_hours: float


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


def net_profit_pct(equity_curve: np.ndarray, initial_balance: float) -> float:
    if len(equity_curve) == 0:
        return 0.0
    return float((equity_curve[-1] - initial_balance) / initial_balance * 100)


def cagr_pct(equity_curve: np.ndarray, initial_balance: float, bar_count: int, timeframe: str) -> float:
    if len(equity_curve) == 0 or equity_curve[-1] <= 0 or bar_count == 0:
        return 0.0
    bars_per_year = BARS_PER_YEAR_BY_TIMEFRAME.get(timeframe, TRADING_DAYS_PER_YEAR)
    years = bar_count / bars_per_year
    if years <= 0:
        return 0.0
    total_return = equity_curve[-1] / initial_balance
    if total_return <= 0:
        return -100.0
    return float((total_return ** (1 / years) - 1) * 100)


def sharpe_ratio(equity_curve: np.ndarray, bars_per_year: float) -> float:
    if len(equity_curve) < 3:
        return 0.0
    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(bars_per_year))


def information_coefficient_information_ratio(trades: list[Trade]) -> float:
    """ICIR — Information Coefficient Information Ratio: the mean of the
    per-trade information coefficient (correlation proxy: sign match between
    predicted direction and realized R-multiple) divided by its stdev
    across a rolling window of trades. A standard robustness-of-edge stat:
    high & stable IC across trades beats a lucky few big winners."""
    if len(trades) < 5:
        return 0.0
    ic_series = np.array([1.0 if t.r_multiple > 0 else (-1.0 if t.r_multiple < 0 else 0.0) for t in trades])
    window = min(20, len(ic_series))
    rolling_ic = pd.Series(ic_series).rolling(window).mean().dropna()
    if len(rolling_ic) < 2 or rolling_ic.std() == 0:
        return 0.0
    return float(rolling_ic.mean() / rolling_ic.std())


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
        sharpe_ratio=sharpe_ratio(result.equity_curve, bars_per_year),
        max_dd_pct=max_drawdown_pct(result.equity_curve),
        net_profit_pct=net_profit_pct(result.equity_curve, initial_balance),
        cagr_pct=cagr_pct(result.equity_curve, initial_balance, len(df), timeframe),
        payoff_ratio=payoff_ratio(result.trades),
        icir=information_coefficient_information_ratio(result.trades),
        avg_hold_hours=avg_hold_hours(result.trades, df),
    )
