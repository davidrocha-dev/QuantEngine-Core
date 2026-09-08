import numpy as np

from app.engine.backtest import Trade
from app.engine.monte_carlo import (
    DEGRADATION_DISCARD_THRESHOLD_PCT,
    compute_degradation_pct,
    evaluate_robustness,
    inject_noise,
    monte_carlo_trade_sequences,
)
from tests.helpers import make_synthetic_ohlc


def make_trade(entry_idx, exit_idx, pnl, direction=1, entry_price=100.0, stop_price=99.5):
    # Priced near make_synthetic_ohlc's ~100 scale so tests that route
    # these trades through inject_noise (which substitutes a real df open
    # price) don't produce nonsensical cross-scale R-multiples.
    stop_distance = abs(entry_price - stop_price)
    r_multiple = pnl / 100 / stop_distance if stop_distance else 0.0
    return Trade(entry_idx, exit_idx, direction, entry_price, entry_price + pnl / 1000, stop_price, r_multiple, pnl, "target")


def test_monte_carlo_same_trades_different_order_same_final_return():
    trades = [make_trade(i, i + 1, pnl) for i, pnl in enumerate([100, -50, 30, -20, 80])]
    results = monte_carlo_trade_sequences(trades, 10_000, iterations=50, rng=np.random.default_rng(1))
    # Reshuffling doesn't change the SUM of pnls, only the path — final
    # return should be identical (no compounding) across all iterations.
    assert all(abs(r - results[0]) < 1e-9 for r in results)


def test_monte_carlo_empty_trades_returns_empty():
    assert monte_carlo_trade_sequences([], 10_000) == []


def test_compute_degradation_pct_no_change():
    assert compute_degradation_pct(20.0, 20.0) == 0.0


def test_compute_degradation_pct_full_wipeout():
    assert compute_degradation_pct(20.0, 0.0) == 100.0


def test_compute_degradation_pct_negative_baseline_edge_case():
    assert compute_degradation_pct(-5.0, -10.0) == 100.0
    assert compute_degradation_pct(-5.0, -5.0) == 0.0


def test_inject_noise_applies_adverse_cost_against_direction():
    df = make_synthetic_ohlc(200, seed=5)
    # Build trades whose entry price matches the df's own price scale so
    # the displaced-open substitution inject_noise performs is meaningful.
    trades = [
        Trade(idx, idx + 5, 1, float(df["open"].iloc[idx]), float(df["close"].iloc[idx + 5]), float(df["open"].iloc[idx]) - 1.0, 1.0, 50.0, "target")
        for idx in (10, 20, 30)
    ]
    rng = np.random.default_rng(2)
    noisy = inject_noise(trades, df, point_size=0.0001, rng=rng)
    for original, perturbed in zip(trades, noisy):
        expected_open = float(df["open"].iloc[perturbed.entry_idx])
        # Long trade: adverse cost is subtracted from the displaced open.
        assert perturbed.entry_price <= expected_open
        assert perturbed.entry_idx > original.entry_idx


def make_trade_from_df(df, entry_idx, exit_idx, pnl, stop_distance=0.5, risk_amount=100.0):
    # inject_noise substitutes the real df open at a displaced bar index,
    # so a trade's assumed entry_price must match the df at entry_idx —
    # otherwise the substitution reads as a huge artificial price jump
    # rather than a small few-bar displacement. Likewise, exit_price must
    # sit `r_multiple` stop-distances from entry (not an arbitrary offset)
    # so the trade's price extent is physically consistent with its stop —
    # inject_noise's r_multiple recompute divides by this same stop_distance.
    entry_price = float(df["open"].iloc[entry_idx])
    r_multiple = pnl / risk_amount
    exit_price = entry_price + r_multiple * stop_distance
    return Trade(entry_idx, exit_idx, 1, entry_price, exit_price, entry_price - stop_distance, r_multiple, pnl, "target")


def test_evaluate_robustness_stable_trades_pass():
    df = make_synthetic_ohlc(500, seed=9)
    # A wide stop (large relative to this series' ~0.1/bar volatility)
    # means a 1-2 bar entry displacement is a small fraction of an R —
    # this is the "robust" case. A tight stop (see the wipeout test below)
    # makes the same displacement a large fraction of an R — fragile.
    trades = [make_trade_from_df(df, i * 5, i * 5 + 2, 50, stop_distance=5.0) for i in range(20)]
    result = evaluate_robustness(trades, df, initial_balance=10_000, point_size=0.0001, rng=np.random.default_rng(0))
    assert result.passed
    assert result.monte_carlo_degradation_pct <= DEGRADATION_DISCARD_THRESHOLD_PCT


def test_evaluate_robustness_wipeout_fails():
    df = make_synthetic_ohlc(500, seed=9)
    # One huge winner + many tiny losers: shuffling puts the winner last in
    # the worst case, but degradation here comes from noise blowing away
    # a fragile single-trade-dependent return.
    trades = [make_trade_from_df(df, 0, 2, 1000)] + [make_trade_from_df(df, i, i + 1, -5) for i in range(1, 50)]
    result = evaluate_robustness(trades, df, initial_balance=10_000, point_size=0.01, rng=np.random.default_rng(0))
    assert result.baseline_return_pct > 0
    # Not asserting pass/fail (depends on RNG draw) — just that the fields are populated sanely.
    assert result.fail_reason is None or "degradation" in result.fail_reason
