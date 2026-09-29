import pytest
import numpy as np
import pandas as pd

from app.engine.backtest import EXIT_STOP, EXIT_TARGET, _scan_exit_numba, simulate_trades
from app.genome.gene_pool import GenePoolConfig
from app.genome.genome import GeneChoice, Genome
from tests.helpers import make_synthetic_ohlc


def test_scan_exit_numba_hits_target_first():
    highs = np.array([1.0, 1.01, 1.05, 1.02])
    lows = np.array([1.0, 0.99, 1.00, 1.00])
    idx, price, code = _scan_exit_numba(highs, lows, highs * 0 + 1.0, 0, 1, stop_price=0.90, target_price=1.05, max_hold=10)
    assert idx == 2
    assert price == 1.05
    assert code == EXIT_TARGET


def test_scan_exit_numba_hits_stop_first():
    highs = np.array([1.0, 1.01, 1.02, 1.03])
    lows = np.array([1.0, 0.89, 1.00, 1.00])
    idx, price, code = _scan_exit_numba(highs, lows, highs * 0 + 1.0, 0, 1, stop_price=0.90, target_price=1.20, max_hold=10)
    assert idx == 1
    assert price == 0.90
    assert code == EXIT_STOP


def test_scan_exit_numba_respects_max_hold():
    highs = np.array([1.0, 1.01, 1.01, 1.01, 1.01])
    lows = np.array([1.0, 0.99, 0.99, 0.99, 0.99])
    idx, price, code = _scan_exit_numba(highs, lows, highs * 0 + 1.0, 0, 1, stop_price=0.5, target_price=5.0, max_hold=2)
    assert idx == 2
    assert np.isnan(price)


def test_scan_exit_numba_short_direction():
    highs = np.array([1.0, 1.10, 1.01, 1.01])
    lows = np.array([1.0, 1.00, 0.95, 1.00])
    idx, price, code = _scan_exit_numba(highs, lows, highs * 0 + 1.0, 0, -1, stop_price=1.10, target_price=0.95, max_hold=10)
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
        genome, highs, lows, closes, closes,
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
        genome, highs, lows, closes, closes,
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


def test_a_resting_order_fills_only_when_price_trades_through_it():
    """Filling a buy limit the moment a bar's low reaches it means buying
    at the exact low of that bar — the one price in it nobody gets — and
    then measuring the trade from there.

    Measured on a real strategy: swapping only the execution gene, with
    signal, bias, filters and management untouched, moved the campaign
    return from +27,381% on LIMIT2 to -9.6% on MARKET2. Every one of the
    26 strategies in the Vault at the time used LIMIT or LIMIT2 — the
    search had found this and converged on it.
    """
    from app.engine.backtest import _wait_for_fill

    # A bar that reaches exactly 100.0 and no further.
    highs = np.array([101.0, 101.0, 101.0])
    lows = np.array([100.5, 100.0, 100.5])

    touched_exactly = _wait_for_fill(highs, lows, 0, 1, order_price=100.0, is_limit=True, margin=0.1)
    assert touched_exactly == (None, None), "a buy limit merely reached is not a buy limit filled"

    traded_through = _wait_for_fill(highs, lows, 0, 1, order_price=100.2, is_limit=True, margin=0.1)
    assert traded_through[0] == 1 and traded_through[1] == 100.2

    # And the same rule the other way for a sell limit, which rests above.
    sell_reached = _wait_for_fill(
        np.array([100.0, 100.0, 100.0]), np.array([99.0, 99.0, 99.0]), 0, -1,
        order_price=100.0, is_limit=True, margin=0.1,
    )
    assert sell_reached == (None, None)


def test_a_wider_spread_makes_a_resting_order_harder_to_fill():
    """The margin is the spread: a buy limit rests on the ask, and the
    quote has to cross it."""
    from app.engine.backtest import _wait_for_fill

    highs = np.array([101.0, 101.0])
    lows = np.array([100.5, 99.95])

    assert _wait_for_fill(highs, lows, 0, 1, 100.0, is_limit=True, margin=0.0)[0] == 1
    assert _wait_for_fill(highs, lows, 0, 1, 100.0, is_limit=True, margin=0.2) == (None, None)


def _gold_like_frame(n_bars: int = 600) -> pd.DataFrame:
    """The synthetic walk, scaled to gold's price. The volume ladder only
    bites when the requested size lands near the broker's minimum, and at
    a price of 100 with a 100-unit contract it never does."""
    df = make_synthetic_ohlc(n_bars=n_bars, seed=11, drift=0.0004, volatility=0.0025).copy()
    for column in ("open", "high", "low", "close"):
        df[column] = df[column] * 45.0
    return df


def _ladder_genome():
    return Genome(
        signal=GeneChoice("MOM BREAK", {"period": 5, "threshold": 0.001}),
        bias=None,
        filters=[],
        execution=GeneChoice("MARKET2", {"confirm_bars": 2}),
        management=None,
        stop_atr_mult=3.0,
        target_atr_mult=3.68,
    )


def test_the_broker_volume_ladder_breaks_fixed_fractional_risk():
    """Continuous sizing is a fiction on a small account.

    simulate_trades risked exactly risk_pct on every trade because it sized
    in fractions of a lot. A broker fills multiples of lot_step and never
    less than min_lot, so the requested 0.012 lots becomes 0.01 and a wide
    stop then risks *more* than the setting allows. Strategy #6F0A16 run in
    MetaTrader from a 10,000 account at 0.35% put all 344 of its trades on
    the 0.01 minimum: the median risked 27.20 of the 35.00 asked for and
    the worst risked 68.85, twice the setting. Modelling the ladder
    reproduced that - worst 69.64, max drawdown 4.94% against 5.01%.
    """
    df = _gold_like_frame()
    kwargs = dict(initial_balance=10_000.0, risk_pct=0.0035, point_size=0.001,
                  cost_points=0.0, commission_per_lot=0.0, contract_size=100.0)

    continuous = simulate_trades(df, _ladder_genome(), GenePoolConfig(), **kwargs)
    laddered = simulate_trades(df, _ladder_genome(), GenePoolConfig(), **kwargs,
                               lot_step=0.01, min_lot=0.01, max_lot=10.0)

    assert len(continuous.trades) == len(laddered.trades), "sizing must not change which trades happen"

    def risked(result):
        return [abs(t.pnl / t.r_multiple) for t in result.trades if t.r_multiple]

    smooth, stepped = risked(continuous), risked(laddered)
    assert smooth, "the fixture has to trade for this to mean anything"
    # Continuous sizing risks the same amount every time, bar compounding.
    assert max(smooth) / min(smooth) < 1.2
    # The ladder does not, and that spread is the whole point.
    assert max(stepped) / min(stepped) > 1.5
    # Here the requested size is under the minimum, so every trade is
    # clamped up and risks more than the setting - the case that costs a
    # prop-firm account its daily limit.
    assert min(stepped) > max(smooth)

    # Ask for enough size to clear the minimum and the ladder rounds the
    # other way, down to the step, and the account risks less than it set.
    rich = dict(kwargs, risk_pct=0.03)
    floored = risked(simulate_trades(df, _ladder_genome(), GenePoolConfig(), **rich,
                                     lot_step=0.01, min_lot=0.01, max_lot=10.0))
    exact = risked(simulate_trades(df, _ladder_genome(), GenePoolConfig(), **rich))
    assert max(floored) < max(exact)


def test_the_ladder_is_off_unless_a_step_is_given():
    """Every existing caller passes no ladder and must keep its numbers."""
    df = _gold_like_frame()
    kwargs = dict(initial_balance=10_000.0, risk_pct=0.0035, point_size=0.001,
                  cost_points=0.0, commission_per_lot=0.0, contract_size=100.0)

    default = simulate_trades(df, _ladder_genome(), GenePoolConfig(), **kwargs)
    explicit = simulate_trades(df, _ladder_genome(), GenePoolConfig(), **kwargs,
                               lot_step=0.0, min_lot=0.01, max_lot=10.0)

    assert default.final_balance == explicit.final_balance


# --- The backtest and the EA take the same trades -------------------------


def test_a_stop_gapped_through_fills_at_the_open_not_at_the_stop():
    highs = np.array([1.00, 1.00, 0.86])
    lows = np.array([1.00, 0.99, 0.80])
    opens = np.array([1.00, 1.00, 0.85])  # bar 2 opens below the 0.90 stop
    idx, price, code = _scan_exit_numba(highs, lows, opens, 0, 1, stop_price=0.90, target_price=1.20, max_hold=10)
    assert (idx, code) == (2, EXIT_STOP)
    assert price == 0.85


def test_a_stop_entry_gapped_through_fills_at_the_open():
    from app.engine.backtest import _wait_for_fill

    highs = np.array([100.0, 101.5])
    lows = np.array([99.5, 100.9])
    opens = np.array([100.0, 101.0])  # opens above the 100.5 buy stop
    assert _wait_for_fill(highs, lows, 0, 1, 100.5, is_limit=False, opens=opens) == (1, 101.0)


def _one_bar_series(n=40):
    """Flat bars with a range, so ATR is defined and nothing fills by accident."""
    from datetime import datetime, timedelta, timezone
    import pandas as pd

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return pd.DataFrame({
        "timestamp": [start + timedelta(hours=i) for i in range(n)],
        "open": [100.0] * n, "high": [100.1] * n, "low": [99.9] * n, "close": [100.0] * n,
        "volume": [1.0] * n,
    })


def test_an_unfilled_resting_order_blocks_signals_for_its_whole_window(monkeypatch):
    """The EA holds one pending order at a time. While it waits, no other
    signal is looked at; the next one evaluated is on the bar it expired
    after. The backtest used to try again on the very next bar."""
    import pandas as pd

    from app.engine import backtest
    from app.genome.gene_pool import GenePoolConfig
    from app.genome.genome import GeneChoice, Genome

    df = _one_bar_series()
    placed: list[int] = []

    def never_fills(df_, genome, signal_idx, *args, **kwargs):
        placed.append(signal_idx)
        return None, None

    monkeypatch.setattr(backtest, "_resolve_entry", never_fills)
    spec = backtest.ALL_GENES[("signal", "BREAKOUT")]
    monkeypatch.setitem(
        backtest.ALL_GENES, ("signal", "BREAKOUT"),
        type(spec)(spec.name, spec.layer, spec.default_params, lambda d, h, p: pd.Series(1, index=d.index)),
    )
    genome = Genome(signal=GeneChoice("BREAKOUT", {}), execution=GeneChoice("LIMIT", {"offset_atr": 0.5}))
    backtest.simulate_trades(df, genome, GenePoolConfig())

    gaps = np.diff(placed)
    assert (gaps == backtest.PENDING_ORDER_WINDOW_BARS).all()


def test_a_usdjpy_lot_is_sized_in_account_currency():
    """A JPY price move is worth 1/price dollars. Without converting, the
    backtest asked for about 1/150th of the real lot and the 0.01 minimum
    took over."""
    import pandas as pd

    from app.engine.backtest import simulate_trades
    from app.genome.gene_pool import GenePoolConfig
    from app.genome.genome import GeneChoice, Genome
    from tests.helpers import make_synthetic_ohlc

    df = make_synthetic_ohlc(600)
    df[["open", "high", "low", "close"]] *= 1.5  # ~150, a yen-like price
    genome = Genome(signal=GeneChoice("BREAKOUT", {"period": 20}), execution=GeneChoice("MARKET2", {"confirm_bars": 1}))
    rate = pd.Series(1 / df["close"].to_numpy(), index=pd.DatetimeIndex(df["timestamp"]))
    kwargs = dict(initial_balance=10_000, risk_pct=0.01, contract_size=100_000,
                  lot_step=0.01, min_lot=0.01, max_lot=100)

    converted = simulate_trades(df, genome, GenePoolConfig(), quote_to_account=rate, **kwargs)
    # With the lot sized in dollars, the ladder rounds a real lot and the
    # loss on a stop is the 1% of the balance at entry asked for, within
    # one lot step.
    balance, checked = 10_000.0, 0
    for trade in converted.trades:
        if trade.exit_reason == "stop":
            assert trade.pnl == pytest.approx(-0.01 * balance, rel=0.05)
            checked += 1
        balance += trade.pnl
    assert checked
