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
    stats = make_stats(net_profit_pct=30.0, max_dd_pct=10.0, win_rate=0.5, sharpe_ratio=2.0)
    assert fitness_f1_stability(stats) == (30.0 / 10.0) * 0.5 * 2.0


def test_fitness_f1_losing_strategy_ranks_below_winners():
    # Negative Net Profit x negative Sharpe must not multiply out positive.
    loser = make_stats(net_profit_pct=-30.0, max_dd_pct=10.0, win_rate=0.4, sharpe_ratio=-2.0)
    winner = make_stats(net_profit_pct=1.0, max_dd_pct=10.0, win_rate=0.4, sharpe_ratio=0.1)
    assert fitness_f1_stability(loser) < 0 < fitness_f1_stability(winner)


def test_fitness_f1_zero_drawdown_is_zero_not_infinite():
    stats = make_stats(max_dd_pct=0.0)
    assert fitness_f1_stability(stats) == 0.0


def test_fitness_f2_formula():
    stats = make_stats(cagr_pct=20.0, profit_factor=1.5, payoff_ratio=2.0)
    assert fitness_f2_expansion(stats) == 20.0 * 1.5 * 2.0


def test_fitness_f2_infinite_profit_factor_treated_as_zero():
    stats = make_stats(cagr_pct=20.0, profit_factor=float("inf"), payoff_ratio=2.0)
    assert fitness_f2_expansion(stats) == 0.0
