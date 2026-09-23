import numpy as np

from app.engine.backtest import BacktestResult, Trade
from app.engine.metrics import (
    BARS_PER_YEAR_BY_TIMEFRAME,
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
        stop_price=0.99, target_price=1.02, r_multiple=r_multiple if r_multiple is not None else pnl / 100, pnl=pnl, exit_reason="target",
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


def test_a_short_window_reports_no_cagr_rather_than_a_fantasy():
    """Compounding six weeks up to a year is arithmetic, not forecasting.

    The Vault had strategies showing 27,929,074,494,351%/yr because a
    two-month M1 backtest was annualised and then clamped. A clamped
    absurdity is worse than nothing: it looks like a measurement and it
    sorts like one.
    """
    from app.engine.metrics import MIN_YEARS_TO_ANNUALISE, cagr_pct

    six_weeks_of_m1 = int(BARS_PER_YEAR_BY_TIMEFRAME["M1"] * 0.12)
    curve = np.linspace(10_000, 13_000, six_weeks_of_m1)

    assert cagr_pct(curve, 10_000, six_weeks_of_m1, "M1") is None


def test_a_long_enough_window_still_reports_one():
    from app.engine.metrics import cagr_pct

    two_years = BARS_PER_YEAR_BY_TIMEFRAME["H1"] * 2
    curve = np.linspace(10_000, 14_400, two_years)  # +44% over two years

    result = cagr_pct(curve, 10_000, two_years, "H1")

    assert result is not None
    assert 15 < result < 25  # about 20%/yr


def test_the_cap_still_holds_on_a_long_window():
    from app.engine.metrics import MAX_REPORTED_CAGR_PCT, cagr_pct

    year = BARS_PER_YEAR_BY_TIMEFRAME["H1"]
    curve = np.linspace(10_000, 10_000_000, year)

    assert cagr_pct(curve, 10_000, year, "H1") == MAX_REPORTED_CAGR_PCT


def test_phase_2_fitness_survives_an_unannualisable_window():
    # F2 is CAGR x PF x payoff; None would crash it, and zero would flatten
    # every candidate in the tribe to the same score.
    from app.engine.fitness import fitness_f2_expansion
    from app.engine.metrics import TradeStats

    stats = TradeStats(
        trade_count=120, profit_factor=1.8, win_rate=0.5, sharpe_ratio=1.0, max_dd_pct=10.0,
        net_profit_pct=25.0, cagr_pct=None, payoff_ratio=1.5, icir=1.0, avg_hold_hours=4.0,
    )

    assert fitness_f2_expansion(stats) > 0


def test_a_blown_account_is_not_reported_as_flat():
    """`equity_curve[-1] <= 0` sat in the same guard as "no data" and
    returned 0.0 — the figure a strategy that never traded gets, and a
    better one than a strategy that lost half. It also made the -100.0
    branch below it unreachable, since nothing else drives final equity to
    zero."""
    import numpy as np

    from app.engine.metrics import cagr_pct

    two_years = 252 * 24 * 2

    assert cagr_pct(np.array([10_000.0, 0.0]), 10_000.0, two_years, "H1") == -100.0
    assert cagr_pct(np.array([10_000.0, -50.0]), 10_000.0, two_years, "H1") == -100.0

    # And it still sorts below a survivable loss, which was the point.
    halved = cagr_pct(np.array([10_000.0, 5_000.0]), 10_000.0, two_years, "H1")
    assert halved is not None and -100.0 < halved < 0.0

    # A window too short to annualise still reports the total loss: losing
    # everything is a fact, not an extrapolation.
    assert cagr_pct(np.array([10_000.0, 0.0]), 10_000.0, 100, "H1") == -100.0
