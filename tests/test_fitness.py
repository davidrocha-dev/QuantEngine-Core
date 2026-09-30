from app.engine.backtest import BacktestResult
from app.engine.fitness import (
    PHASE_0_MIN_PROFIT_FACTOR,
    PHASE_0_MIN_TRADES,
    fitness_f1_stability,
    fitness_f2_expansion,
    passes_phase_0_screening,
)
from app.engine.metrics import TradeStats


def make_stats(**overrides) -> TradeStats:
    defaults = dict(
        trade_count=150, profit_factor=1.5, win_rate=0.55, sharpe_ratio=1.2,
        max_dd_pct=10.0, net_profit_pct=30.0, cagr_pct=15.0, payoff_ratio=1.8,
        icir=1.0, avg_hold_hours=4.0,
    )
    defaults.update(overrides)
    return TradeStats(**defaults)


def test_passes_phase_0_screening_happy_path():
    ok, reason = passes_phase_0_screening(make_stats(), BacktestResult())
    assert ok
    assert reason is None


def test_fails_on_low_profit_factor():
    ok, reason = passes_phase_0_screening(make_stats(profit_factor=1.0), BacktestResult())
    assert not ok
    assert "profit factor" in reason


def test_fails_on_too_few_trades():
    ok, reason = passes_phase_0_screening(make_stats(trade_count=50), BacktestResult())
    assert not ok
    assert "trades" in reason


def test_fails_on_margin_breach():
    ok, reason = passes_phase_0_screening(make_stats(), BacktestResult(margin_breach=True))
    assert not ok
    assert "margin" in reason


def test_boundary_values_pass():
    ok, _ = passes_phase_0_screening(
        make_stats(trade_count=PHASE_0_MIN_TRADES, profit_factor=PHASE_0_MIN_PROFIT_FACTOR), BacktestResult()
    )
    assert ok


def test_fitness_f1_formula():
    stats = make_stats(net_profit_pct=30.0, max_dd_pct=10.0, sqn=2.5)
    assert fitness_f1_stability(stats) == (30.0 / 10.0) * 2.5


def test_fitness_f1_no_longer_pays_for_win_rate():
    """Win rate paid for winning often, not for winning more than was lost,
    and bred small targets behind wide stops that MetaTrader's costs erase."""
    often = make_stats(win_rate=0.9, sqn=1.0)
    rarely = make_stats(win_rate=0.3, sqn=1.0)
    assert fitness_f1_stability(often) == fitness_f1_stability(rarely)


def test_fitness_f1_losing_strategy_ranks_below_winners():
    # Negative Net Profit x negative SQN must not multiply out positive.
    loser = make_stats(net_profit_pct=-30.0, max_dd_pct=10.0, sqn=-2.0)
    winner = make_stats(net_profit_pct=1.0, max_dd_pct=10.0, sqn=0.1)
    assert fitness_f1_stability(loser) < 0 < fitness_f1_stability(winner)


def test_fitness_f1_zero_drawdown_is_zero_not_infinite():
    stats = make_stats(max_dd_pct=0.0, sqn=2.0)
    assert fitness_f1_stability(stats) == 0.0


def test_fitness_f2_formula():
    stats = make_stats(cagr_pct=20.0, profit_factor=1.5, payoff_ratio=2.0)
    assert fitness_f2_expansion(stats) == 20.0 * 1.5


def test_fitness_f2_caps_the_profit_factor_and_ignores_payoff():
    from app.engine.fitness import FITNESS_MAX_PROFIT_FACTOR

    lucky = make_stats(cagr_pct=20.0, profit_factor=8.0, payoff_ratio=6.0)
    assert fitness_f2_expansion(lucky) == 20.0 * FITNESS_MAX_PROFIT_FACTOR


def test_a_candidate_with_no_losing_trades_ranks_at_the_top_not_the_bottom():
    from app.engine.fitness import FITNESS_MAX_PROFIT_FACTOR

    flawless = make_stats(cagr_pct=20.0, profit_factor=float("inf"), payoff_ratio=0.0)
    mediocre = make_stats(cagr_pct=8.0, profit_factor=1.3, payoff_ratio=1.1)

    assert fitness_f2_expansion(flawless) == 20.0 * FITNESS_MAX_PROFIT_FACTOR
    assert fitness_f2_expansion(flawless) > fitness_f2_expansion(mediocre)


def test_sqn_is_edge_per_trade_scaled_by_a_capped_sample():
    import math

    from app.engine.backtest import Trade
    from app.engine.metrics import system_quality_number

    def trades(rs):
        return [Trade(0, 1, 1, 1.0, 1.0, 0.9, 1.1, r, r * 100, "target") for r in rs]

    # mean 0.5, sample std 1.0 (alternating 1.5 / -0.5), 400 trades -> capped at 100
    rs = [1.5, -0.5] * 200
    expected = 0.5 / float(__import__("numpy").std(rs, ddof=1)) * math.sqrt(100)
    assert system_quality_number(trades(rs)) == __import__("pytest").approx(expected)
    assert system_quality_number(trades([-1.0, 0.2] * 50)) < 0
