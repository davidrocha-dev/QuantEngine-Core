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


# --- PARTIAL TP and the take-profit it has to respect --------------------


def _partial_tp_genome(r_multiple: float, target_atr_mult: float) -> Genome:
    return Genome(
        signal=GeneChoice("MACD", {"fast": 12, "slow": 26, "signal": 9}),
        bias=None,
        filters=[],
        execution=GeneChoice("MARKET2", {"confirm_bars": 1}),
        management=GeneChoice("PARTIAL TP", {"r_multiple": r_multiple, "close_fraction": 0.7}),
        stop_atr_mult=2.79,
        target_atr_mult=target_atr_mult,
    )


def test_partial_tp_cannot_fire_beyond_the_take_profit():
    """A take-profit is a real order in the book, and a position cannot run
    through it to reach a partial level beyond it.

    The scan used to stand `partial_price` in for the target, so a trade
    whose target sits at 0.39R kept going to 2.0R, banked most of the
    position there and closed the rest afterwards — reporting several
    times the R the geometry allows. Exported and run in MetaTrader, the
    same genome returned +23% where this said +366%.
    """
    from app.engine.backtest import _run_partial_tp

    # Target at 1.09 ATR against a 2.79 ATR stop is 0.39R; the partial
    # wants 2.0R. It can never be reached.
    genome = _partial_tp_genome(r_multiple=2.0026, target_atr_mult=1.09)
    highs = np.linspace(100.0, 140.0, 60)
    lows = highs - 0.5
    closes = highs - 0.25

    baseline = (7, 110.9, EXIT_TARGET)
    partial_r, exit_idx, exit_price, exit_code = _run_partial_tp(
        genome, highs, lows, closes,
        entry_idx=0, direction=1, entry_price=100.0,
        stop_price=100.0 - 2.79, target_price=100.0 + 1.09,
        exit_idx=baseline[0], exit_price=baseline[1], exit_code=baseline[2],
    )

    assert partial_r is None, "nothing can be banked past the take-profit"
    assert (exit_idx, exit_price, exit_code) == baseline, "the trade exits where it always would have"


def test_partial_tp_still_fires_when_the_target_is_further_out():
    from app.engine.backtest import _run_partial_tp

    # Target at 5.58 ATR against a 2.79 ATR stop is 2.0R, and the partial
    # wants 1.0R — reachable, so it banks.
    genome = _partial_tp_genome(r_multiple=1.0, target_atr_mult=5.58)
    highs = np.linspace(100.0, 140.0, 60)
    lows = highs - 0.5
    closes = highs - 0.25

    partial_r, _, _, _ = _run_partial_tp(
        genome, highs, lows, closes,
        entry_idx=0, direction=1, entry_price=100.0,
        stop_price=100.0 - 2.79, target_price=100.0 + 5.58,
        exit_idx=30, exit_price=105.58, exit_code=EXIT_TARGET,
    )

    assert partial_r == 1.0


def test_costs_change_which_barrier_is_reached_not_just_the_pnl():
    """Costs used to be a flat deduction from R after the fact.

    That leaves the stop and the target at their nominal distances, so it
    can make a trade less profitable but never make it lose: the win rate
    came out identical at any spread. In a mid-priced series a long is
    filled at the ask and both its barriers trigger on the bid, so the
    stop sits a spread nearer and the target a spread further. For a
    strategy near its break-even win rate that is the entire result.
    """
    df = make_synthetic_ohlc(6_000, seed=21, drift=0.0, volatility=0.004)
    pool = GenePoolConfig()
    genome = Genome(
        signal=GeneChoice("MACD", {"fast": 12, "slow": 26, "signal": 9}),
        bias=None,
        filters=[],
        execution=GeneChoice("MARKET2", {"confirm_bars": 1}),
        management=None,
        stop_atr_mult=1.0,
        target_atr_mult=1.0,
    )

    def win_rate(cost_points: float) -> tuple[float, int]:
        result = simulate_trades(
            df, genome, pool, initial_balance=10_000.0, risk_pct=0.01,
            point_size=0.0001, cost_points=cost_points, contract_size=100_000.0,
        )
        wins = sum(1 for t in result.trades if t.pnl > 0)
        return (wins / len(result.trades) if result.trades else 0.0), len(result.trades)

    # The effect is monotonic but small at realistic spreads - the point
    # is that it exists at all. Under the old model the win rate was
    # mathematically independent of cost, identical to the last decimal
    # at every spread, so a wide one is used here to make the assertion
    # robust rather than marginal.
    rates = [win_rate(cost)[0] for cost in (0.0, 100.0, 200.0, 400.0)]
    counts = [win_rate(cost)[1] for cost in (0.0, 400.0)]

    assert all(count > 30 for count in counts), "need trades for this to mean anything"
    assert rates == sorted(rates, reverse=True), f"win rate must fall as cost rises: {rates}"
    assert rates[0] - rates[-1] > 0.02, f"and by more than rounding: {rates}"


def test_a_stop_loss_costs_exactly_one_r():
    """The spread is paid getting in, so it is already inside every
    distance measured from the entry. Deducting it again at the exit
    double-charged it."""
    df = make_synthetic_ohlc(3_000, seed=22, drift=-0.002, volatility=0.006)
    pool = GenePoolConfig()
    genome = Genome(
        signal=GeneChoice("MACD", {"fast": 12, "slow": 26, "signal": 9}),
        bias=None,
        filters=[],
        execution=GeneChoice("MARKET2", {"confirm_bars": 1}),
        management=None,
        stop_atr_mult=1.5,
        target_atr_mult=3.0,
    )
    result = simulate_trades(
        df, genome, pool, initial_balance=10_000.0, risk_pct=0.01,
        point_size=0.0001, cost_points=30.0, commission_per_lot=0.0, contract_size=100_000.0,
    )

    stopped = [t for t in result.trades if t.exit_reason == "stop"]
    assert stopped, "this series is meant to produce stop-outs"
    for trade in stopped:
        assert abs(trade.r_multiple + 1.0) < 1e-9, trade.r_multiple
