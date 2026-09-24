import pytest
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


def test_icir_is_positive_for_a_profitable_low_win_rate_strategy():
    """ICIR reduced each trade to the *sign* of its R before averaging,
    which makes the statistic (2 x win_rate - 1) / spread — a number that
    cannot be positive unless the strategy wins more often than it loses.

    Measured across a real Vault: 25 strategies, 25 negative ICIRs, a
    perfect monotone in win rate (15.8% -> -4.56, 40.1% -> -0.94) and no
    relation to whether the strategy made money. One returning +123.9%
    was reported at -4.56, under a Promotion Verdict line reading "ICIR
    computed", which is true and is not a test.
    """
    from app.engine.backtest import Trade
    from app.engine.metrics import information_coefficient_information_ratio as icir

    def run(r_multiples):
        return [Trade(i, i + 1, 1, 1.0, 1.0, 0.9, 1.2, r, r * 100, "target") for i, r in enumerate(r_multiples)]

    # One winner in five, paying 6R: loses four times as often as it wins
    # and makes money doing it.
    high_payoff = run([6.0 if i % 5 == 0 else -1.0 for i in range(60)])
    # Wins four times in five and gives it all back on the fifth.
    losing_but_frequent = run([0.2 if i % 5 else -1.5 for i in range(60)])

    assert sum(t.r_multiple for t in high_payoff) > 0
    assert sum(t.r_multiple for t in losing_but_frequent) < 0

    assert icir(high_payoff) > 0, "a profitable strategy cannot have a negative edge per trade"
    assert icir(losing_but_frequent) < 0, "and a losing one cannot have a positive one"
    assert icir(high_payoff) > icir(losing_but_frequent)


def test_the_worst_day_is_measured_against_that_days_opening_equity():
    """Total drawdown and daily drawdown are different rules, and a
    strategy can pass one while failing the other. Prop firms enforce
    both — typically 10% overall and 5% in any one day — and this engine
    only measured the first, so a strategy that lost 6% in an afternoon
    and recovered it looked flawless right up to the account being
    closed.
    """
    import numpy as np
    import pandas as pd

    from app.engine.metrics import max_daily_drawdown_pct, max_drawdown_pct

    # Two days. Day one dips 6% and recovers in full; day two drifts down
    # 3% and stays there. Overall peak-to-trough never exceeds 6%.
    stamps = pd.DatetimeIndex(
        [f"2026-01-01T{h:02d}:00" for h in range(6)] + [f"2026-01-02T{h:02d}:00" for h in range(6)]
    )
    equity = np.array(
        [10_000, 9_800, 9_400, 9_600, 9_900, 10_000, 10_000, 9_950, 9_900, 9_800, 9_750, 9_700],
        dtype=float,
    )

    assert max_daily_drawdown_pct(equity, stamps) == pytest.approx(6.0, abs=0.05)
    assert max_drawdown_pct(equity) < 7.0, "the overall figure never sees that day as a problem"

    # A 10% total limit passes this; a 5% daily limit does not.
    assert max_drawdown_pct(equity) < 10.0
    assert max_daily_drawdown_pct(equity, stamps) > 5.0


def test_the_worst_day_is_zero_without_timestamps_rather_than_wrong():
    import numpy as np

    from app.engine.metrics import max_daily_drawdown_pct

    assert max_daily_drawdown_pct(np.array([10_000.0, 9_000.0]), None) == 0.0
