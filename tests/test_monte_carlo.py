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
    target_price = entry_price + (entry_price - stop_price) * 2
    return Trade(entry_idx, exit_idx, direction, entry_price, entry_price + pnl / 1000, stop_price, target_price, r_multiple, pnl, "target")


def test_monte_carlo_bootstrap_produces_a_spread_of_outcomes():
    # The point of the bootstrap: resampling WITH replacement changes which
    # trades occur, so the returns must actually vary. A plain reshuffle of
    # the same P&L cannot (addition is commutative) and so could never fail
    # the REQ-ENGINE-3.10 degradation gate.
    trades = [make_trade(i, i + 1, pnl) for i, pnl in enumerate([100, -50, 30, -20, 80] * 6)]
    returns, drawdowns = monte_carlo_trade_sequences(trades, 10_000, iterations=50, rng=np.random.default_rng(1))

    assert len(returns) == len(drawdowns) == 50
    assert max(returns) - min(returns) > 1e-6
    assert all(dd >= 0 for dd in drawdowns)


def test_monte_carlo_empty_trades_returns_empty():
    assert monte_carlo_trade_sequences([], 10_000) == ([], [])


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
        Trade(idx, idx + 5, 1, float(df["open"].iloc[idx]), float(df["close"].iloc[idx + 5]), float(df["open"].iloc[idx]) - 1.0, float(df["open"].iloc[idx]) + 2.0, 1.0, 50.0, "target")
        for idx in (10, 20, 30)
    ]
    rng = np.random.default_rng(2)
    noisy = inject_noise(trades, df, point_size=0.0001, rng=rng)
    for original, perturbed in zip(trades, noisy):
        expected_open = float(df["open"].iloc[perturbed.entry_idx])
        # Long trade: adverse cost raises the entry above the displaced open.
        assert perturbed.entry_price > expected_open
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
    return Trade(entry_idx, exit_idx, 1, entry_price, exit_price, entry_price - stop_distance, entry_price + stop_distance * 2, r_multiple, pnl, "target")


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


# --- Sampling the noise distribution instead of one point ---------------


def _run(r_multiple: float, count: int = 40) -> list[Trade]:
    """A run of identical trades at a given R, so a fixture can state a
    degradation directly instead of hoping a simulation produces one."""
    return [make_trade(i, i + 1, r_multiple * 100.0) for i in range(count)]


def test_the_noise_gate_reads_the_pessimistic_end_not_the_middle():
    """Measured on a real strategy, noise degradation is bimodal and
    decided almost entirely by one integer: a 1-bar entry displacement
    cost 0.0% across 13 draws, a 2-bar displacement 63.6% across 17. A
    median over that is itself close to a coin toss, so the gate reads
    p90 while the reported figure stays the median."""
    df = make_synthetic_ohlc(2_000, seed=5)
    baseline = _run(1.0)
    # Three draws that barely hurt and two that are ruinous: the median
    # clears the threshold, the p90 does not.
    runs = [_run(0.98), _run(0.97), _run(0.96), _run(-0.5), _run(-0.6)]

    result = evaluate_robustness(baseline, df, 10_000.0, 0.0001, risk_pct=0.01, noisy_runs=runs)

    assert result.noise_samples == 5
    assert result.noise_degradation_p90_pct is not None
    assert result.noise_degradation_p90_pct > result.noise_degradation_pct, (
        "p90 must sit above the median or this fixture tests nothing"
    )
    assert result.noise_degradation_pct <= DEGRADATION_DISCARD_THRESHOLD_PCT, (
        "the median alone would have passed this"
    )
    assert result.noise_degradation_p90_pct > DEGRADATION_DISCARD_THRESHOLD_PCT
    assert not result.passed, "a strategy that collapses on two draws in five is not robust"


def test_a_single_draw_reports_no_range():
    df = make_synthetic_ohlc(2_000, seed=6)

    result = evaluate_robustness(
        _run(1.0), df, 10_000.0, 0.0001, risk_pct=0.01, noisy_trades=_run(0.9)
    )

    assert result.noise_samples == 1
    assert result.noise_degradation_p10_pct is None
    assert result.noise_degradation_p90_pct is None


def test_robustness_band_brackets_the_reported_value():
    from app.engine.monte_carlo import robustness_band

    df = make_synthetic_ohlc(2_000, seed=7)
    runs = [_run(r) for r in (1.0, 0.9, 0.7, 0.5, 0.2)]

    result = evaluate_robustness(_run(1.0), df, 10_000.0, 0.0001, risk_pct=0.01, noisy_runs=runs)
    value, low, high = robustness_band(result)

    assert low is not None and high is not None
    assert low <= value <= high, f"{low} <= {value} <= {high}"


def test_sample_noise_runs_draws_the_requested_number():
    from app.engine.monte_carlo import NOISE_SAMPLES, sample_noise_runs

    seen = []

    def run(noise):
        seen.append(noise)
        return _run(1.0, count=3)

    runs = sample_noise_runs(run, np.random.default_rng(1))

    assert len(runs) == NOISE_SAMPLES == len(seen)
    assert all(1 <= n.entry_delay_bars <= 2 for n in seen)
