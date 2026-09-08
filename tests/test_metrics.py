import numpy as np

from app.engine.backtest import BacktestResult, Trade
from app.engine.metrics import (
    avg_hold_hours,
    cagr_pct,
    information_coefficient_information_ratio,
    max_drawdown_pct,
    net_profit_pct,
    payoff_ratio,
    profit_factor,
    win_rate,
)
from tests.helpers import make_synthetic_ohlc


def make_trade(entry_idx, exit_idx, pnl, r_multiple=None):
    return Trade(
        entry_idx=entry_idx, exit_idx=exit_idx, direction=1, entry_price=1.0, exit_price=1.0 + pnl / 1000,
        stop_price=0.99, r_multiple=r_multiple if r_multiple is not None else pnl / 100, pnl=pnl, exit_reason="target",
    )


def test_profit_factor_basic():
    trades = [make_trade(0, 1, 100), make_trade(1, 2, -50)]
    assert profit_factor(trades) == 2.0


def test_profit_factor_no_losses_is_infinite():
    trades = [make_trade(0, 1, 100)]
    assert profit_factor(trades) == float("inf")


def test_profit_factor_no_trades_is_zero():
    assert profit_factor([]) == 0.0


def test_win_rate():
    trades = [make_trade(0, 1, 100), make_trade(1, 2, -50), make_trade(2, 3, 30)]
    assert win_rate(trades) == 2 / 3


def test_payoff_ratio():
    trades = [make_trade(0, 1, 100), make_trade(1, 2, -50)]
    assert payoff_ratio(trades) == 2.0


def test_max_drawdown_pct():
    equity = np.array([100, 120, 90, 110])
    # Peak 120 -> trough 90 => -25%
    assert max_drawdown_pct(equity) == 25.0


def test_net_profit_pct():
    equity = np.array([100, 150])
    assert net_profit_pct(equity, 100) == 50.0


def test_cagr_pct_doubles_in_one_year_on_h1():
    bars_per_year = 252 * 24
    equity = np.array([100.0] * bars_per_year + [200.0])
    result = cagr_pct(equity, 100.0, bars_per_year, "H1")
    assert abs(result - 100.0) < 1.0  # ~100% CAGR for doubling in ~1 year


def test_icir_zero_with_too_few_trades():
    trades = [make_trade(0, 1, 10)]
    assert information_coefficient_information_ratio(trades) == 0.0


def test_icir_positive_for_consistent_winners():
    trades = [make_trade(i, i + 1, 10, r_multiple=1.0) for i in range(30)]
    assert information_coefficient_information_ratio(trades) >= 0.0


def test_avg_hold_hours():
    df = make_synthetic_ohlc(10)
    trades = [make_trade(0, 2, 10), make_trade(3, 7, 10)]
    result = avg_hold_hours(trades, df)
    assert result == (2 + 4) / 2  # hourly bars: 2h and 4h holds
