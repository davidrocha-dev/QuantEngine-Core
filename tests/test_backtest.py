import numpy as np

from app.engine.backtest import EXIT_STOP, EXIT_TARGET, _scan_exit_numba, simulate_trades
from app.genome.gene_pool import GenePoolConfig
from app.genome.genome import GeneChoice, Genome
from tests.helpers import make_synthetic_ohlc


def test_scan_exit_numba_hits_target_first():
    highs = np.array([1.0, 1.01, 1.05, 1.02])
    lows = np.array([1.0, 0.99, 1.00, 1.00])
    idx, price, code = _scan_exit_numba(highs, lows, 0, 1, stop_price=0.90, target_price=1.05, max_hold=10)
    assert idx == 2
    assert price == 1.05
    assert code == EXIT_TARGET


def test_scan_exit_numba_hits_stop_first():
    highs = np.array([1.0, 1.01, 1.02, 1.03])
    lows = np.array([1.0, 0.89, 1.00, 1.00])
    idx, price, code = _scan_exit_numba(highs, lows, 0, 1, stop_price=0.90, target_price=1.20, max_hold=10)
    assert idx == 1
    assert price == 0.90
    assert code == EXIT_STOP


def test_scan_exit_numba_respects_max_hold():
    highs = np.array([1.0, 1.01, 1.01, 1.01, 1.01])
    lows = np.array([1.0, 0.99, 0.99, 0.99, 0.99])
    idx, price, code = _scan_exit_numba(highs, lows, 0, 1, stop_price=0.5, target_price=5.0, max_hold=2)
    assert idx == 2
    assert np.isnan(price)


def test_scan_exit_numba_short_direction():
    highs = np.array([1.0, 1.10, 1.01, 1.01])
    lows = np.array([1.0, 1.00, 0.95, 1.00])
    idx, price, code = _scan_exit_numba(highs, lows, 0, -1, stop_price=1.10, target_price=0.95, max_hold=10)
    assert idx == 1
    assert code == EXIT_STOP


def test_simulate_trades_produces_trades_on_trending_data():
    df = make_synthetic_ohlc(2000, seed=7, drift=0.001, volatility=0.002)
    pool = GenePoolConfig()
    genome = Genome(
        signal=GeneChoice("BREAKOUT", {"period": 20}),
        execution=GeneChoice("STOP", {"offset_atr": 0.3}),
    )
    result = simulate_trades(df, genome, pool, initial_balance=10_000, risk_pct=0.01)
    assert len(result.trades) > 0
    assert len(result.equity_curve) == len(df)


def test_simulate_trades_rejects_genome_using_disabled_gene():
    df = make_synthetic_ohlc(100)
    pool = GenePoolConfig()
    pool.set_enabled("signal", "BREAKOUT", False)
    genome = Genome(signal=GeneChoice("BREAKOUT", {}), execution=GeneChoice("MARKET2", {"confirm_bars": 1}))
    try:
        simulate_trades(df, genome, pool)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_simulate_trades_risk_sizing_bounds_single_trade_loss():
    df = make_synthetic_ohlc(2000, seed=3, drift=-0.0005, volatility=0.003)
    pool = GenePoolConfig()
    genome = Genome(
        signal=GeneChoice("MOM BREAK", {"period": 10, "threshold": 0.002}),
        execution=GeneChoice("MARKET2", {"confirm_bars": 1}),
    )
    result = simulate_trades(df, genome, pool, initial_balance=10_000, risk_pct=0.01)
    # No single stop-out should lose much more than 1% of balance at entry
    # (partial TP / management exits can only improve on this).
    for trade in result.trades:
        assert trade.pnl > -250  # generous bound vs. the nominal 1% (~100) risk


def test_market2_execution_waits_for_confirmation():
    df = make_synthetic_ohlc(300, seed=11)
    pool = GenePoolConfig()
    genome = Genome(
        signal=GeneChoice("BREAKOUT", {"period": 20}),
        execution=GeneChoice("MARKET2", {"confirm_bars": 3}),
    )
    result = simulate_trades(df, genome, pool)
    for trade in result.trades:
        assert trade.entry_idx >= 3
