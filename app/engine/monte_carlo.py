"""Phase 4 robustness perturbations — REQ-ENGINE-3.7/3.8/3.10.

Two independent stress tests run against the OOS trade sequence:
  1. Trade-sequence Monte Carlo: 100 randomized trade sequences, resampled
     WITH replacement (bootstrap) and compounded at the campaign's risk
     fraction. Resampling matters: a plain reshuffle of absolute P&L is
     mathematically incapable of changing the final return (addition is
     commutative), so it could never fail the REQ-ENGINE-3.10 gate. The
     bootstrap answers the question that gate is really asking — "how much
     of this return depends on the exact trades that happened to occur?".
  2. Noise injection: perturb each trade with a random spread (+1 to +5
     points), slippage (0-15ms), and entry displacement (1-2 bars), then
     recompute the return.

A candidate is discarded if EITHER perturbation degrades its return by
more than 25% relative to the clean OOS baseline (REQ-ENGINE-3.10).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.engine.backtest import EXIT_MAX_HOLD, MAX_HOLD_BARS, Trade, _scan_exit_numba

DEGRADATION_DISCARD_THRESHOLD_PCT = 25.0
MONTE_CARLO_ITERATIONS = 100
SPREAD_POINTS_RANGE = (1, 5)
SLIPPAGE_MS_RANGE = (0, 15)
DISPLACEMENT_BARS_RANGE = (1, 2)


@dataclass
class RobustnessResult:
    baseline_return_pct: float
    monte_carlo_median_return_pct: float
    monte_carlo_worst_return_pct: float  # 5th percentile of the bootstrap distribution
    monte_carlo_degradation_pct: float
    monte_carlo_worst_drawdown_pct: float
    noise_return_pct: float
    noise_degradation_pct: float
    passed: bool
    fail_reason: str | None = None
    # The noise test draws a random perturbation, and one draw is a
    # sample, not a measurement: `entry_delay_bars` alone is a coin flip
    # that moves every entry by one bar or two - half an hour a trade on
    # M15. Judging on a single draw made the same strategy, on the same
    # data, score anywhere from 64 to 80 between clicks, with robustness
    # from 36.6% to 100% and the pass/fail verdict flipping. These are
    # the 10th and 90th percentiles of the sampled degradations, so the
    # number can be shown with the range it came from.
    noise_degradation_p10_pct: float | None = None
    noise_degradation_p90_pct: float | None = None
    noise_samples: int = 1


def _return_pct_from_sequence(pnls: list[float], initial_balance: float) -> float:
    balance = initial_balance
    for pnl in pnls:
        balance += pnl
    return (balance - initial_balance) / initial_balance * 100


def _compound_r_sequence(r_multiples: np.ndarray, risk_pct: float) -> tuple[float, float]:
    """Compounds one sequence of R-multiples at a fixed risk fraction.
    Returns (total return %, max drawdown % along the path)."""
    equity = np.cumprod(1.0 + risk_pct * r_multiples)
    peak = np.maximum.accumulate(np.concatenate(([1.0], equity)))
    path = np.concatenate(([1.0], equity))
    drawdown = (path - peak) / peak
    return float((equity[-1] - 1.0) * 100), float(-drawdown.min() * 100)


def monte_carlo_trade_sequences(
    trades: list[Trade],
    initial_balance: float,
    iterations: int = MONTE_CARLO_ITERATIONS,
    rng: np.random.Generator | None = None,
    risk_pct: float = 0.01,
) -> tuple[list[float], list[float]]:
    """REQ-ENGINE-3.7: 100 randomized trade sequences, each a bootstrap
    resample (with replacement) of the realized R-multiples, compounded at
    `risk_pct`. Returns (returns %, max drawdowns %), one entry per iteration."""
    rng = rng or np.random.default_rng()
    r_multiples = np.array([t.r_multiple for t in trades], dtype=float)
    r_multiples = r_multiples[np.isfinite(r_multiples)]
    if r_multiples.size == 0:
        return [], []

    returns: list[float] = []
    drawdowns: list[float] = []
    for _ in range(iterations):
        sample = rng.choice(r_multiples, size=r_multiples.size, replace=True)
        total_return, max_dd = _compound_r_sequence(sample, risk_pct)
        returns.append(total_return)
        drawdowns.append(max_dd)
    return returns, drawdowns


@dataclass(frozen=True)
class NoiseParams:
    """One draw of REQ-ENGINE-3.8's perturbation, applied by re-running the
    strategy (see `simulate_trades(entry_delay_bars=..., cost_points=...)`)
    rather than by editing trades after the fact."""

    extra_cost_points: float
    entry_delay_bars: int


# Draws taken per robustness evaluation. One is a coin flip; the median
# of five is an estimate of the distribution, which is what the gate is
# actually trying to ask about. Five re-simulations instead of one is
# affordable: REASSESS is a manual action on one strategy, and Phase 4
# runs on the handful of candidates that already cleared three phases.
NOISE_SAMPLES = 5


def draw_noise_params(rng: np.random.Generator | None = None) -> NoiseParams:
    """+1 to +5 points of spread, 0-15ms of slippage and a 1-2 bar entry
    displacement. Slippage (no tick-level replay is available) is modeled
    as a spread-proportional cost scaled by ms/15 — a documented
    simplification, not a literal latency simulation."""
    rng = rng or np.random.default_rng()
    spread_points = float(rng.integers(SPREAD_POINTS_RANGE[0], SPREAD_POINTS_RANGE[1] + 1))
    slippage_ms = float(rng.integers(SLIPPAGE_MS_RANGE[0], SLIPPAGE_MS_RANGE[1] + 1))
    slippage_points = (slippage_ms / 15) * spread_points
    return NoiseParams(
        extra_cost_points=spread_points + slippage_points,
        entry_delay_bars=int(rng.integers(DISPLACEMENT_BARS_RANGE[0], DISPLACEMENT_BARS_RANGE[1] + 1)),
    )


def inject_noise(
    trades: list[Trade],
    df: pd.DataFrame,
    point_size: float,
    rng: np.random.Generator | None = None,
) -> list[Trade]:
    """Deprecated per-trade approximation of REQ-ENGINE-3.8, kept only for
    the tests that document why it was replaced: patching a simulated trade
    cannot know what its stop and target would have done from a displaced
    entry, which biased the stress test in the candidate's favour
    (measured: +0.1 to +0.5R per trade). Use `draw_noise_params` and
    re-simulate instead — see `run_phase_4_robustness`."""
    rng = rng or np.random.default_rng()
    perturbed = []
    opens = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    highs, lows = df["high"].to_numpy(), df["low"].to_numpy()

    for trade in trades:
        displacement = int(rng.integers(DISPLACEMENT_BARS_RANGE[0], DISPLACEMENT_BARS_RANGE[1] + 1))
        displaced_idx = min(trade.entry_idx + displacement, len(opens) - 1)
        new_entry_price = float(opens[displaced_idx])
        # The exit slides with the entry: the trade is the same trade, taken
        # a bar or two later. Holding the original exit price fixed while
        # moving only the entry is not noise — it is a systematic edge
        # (measured: +0.1R on average), which made this test reward the
        # candidates it was supposed to stress.

        spread_points = rng.integers(SPREAD_POINTS_RANGE[0], SPREAD_POINTS_RANGE[1] + 1)
        slippage_ms = rng.integers(SLIPPAGE_MS_RANGE[0], SLIPPAGE_MS_RANGE[1] + 1)
        spread_cost = spread_points * point_size
        slippage_cost = (slippage_ms / 15) * spread_cost
        total_cost_price = spread_cost + slippage_cost

        stop_distance = abs(trade.entry_price - trade.stop_price)
        target_distance = abs(trade.target_price - trade.entry_price)
        if stop_distance <= 0:
            perturbed.append(trade)
            continue

        # Entry displacement shifts the effective entry price; the adverse
        # cost (spread+slippage) always works against the trade's direction:
        # a long pays up (higher entry), a short sells lower.
        adjusted_entry = new_entry_price + trade.direction * total_cost_price
        # Replay the trade from the displaced entry under its own rules: the
        # stop and target move with the entry and the exit is whichever the
        # market reaches first. Reusing the original exit price (or sliding
        # it by the same displacement) silently rewards the perturbation —
        # measured at +0.1 to +0.5R per trade, i.e. the stress test made
        # strategies look better than they were.
        new_stop = adjusted_entry - trade.direction * stop_distance
        new_target = adjusted_entry + trade.direction * target_distance
        displaced_exit_idx, new_exit_price, exit_code = _scan_exit_numba(
            highs, lows, opens, displaced_idx, trade.direction, new_stop, new_target, MAX_HOLD_BARS
        )
        if exit_code == EXIT_MAX_HOLD:
            new_exit_price = float(closes[displaced_exit_idx])

        r_multiple = (new_exit_price - adjusted_entry) * trade.direction / stop_distance
        risk_amount = abs(trade.pnl / trade.r_multiple) if trade.r_multiple != 0 else 0.0
        pnl = risk_amount * r_multiple

        perturbed.append(
            Trade(
                entry_idx=displaced_idx, exit_idx=displaced_exit_idx, direction=trade.direction,
                entry_price=adjusted_entry, exit_price=new_exit_price,
                stop_price=new_stop, target_price=new_target,
                r_multiple=r_multiple, pnl=pnl, exit_reason=trade.exit_reason, partial_closed=trade.partial_closed,
            )
        )
    return perturbed


def compute_degradation_pct(baseline_return_pct: float, perturbed_return_pct: float) -> float:
    if baseline_return_pct <= 0:
        return 100.0 if perturbed_return_pct < baseline_return_pct else 0.0
    return max(0.0, (baseline_return_pct - perturbed_return_pct) / baseline_return_pct * 100)


def _r_sequence_return(trades: list[Trade], risk_pct: float) -> float:
    r_multiples = np.array([t.r_multiple for t in trades], dtype=float)
    r_multiples = r_multiples[np.isfinite(r_multiples)]
    if r_multiples.size == 0:
        return 0.0
    return _compound_r_sequence(r_multiples, risk_pct)[0]


def robustness_band(result: RobustnessResult) -> tuple[float, float | None, float | None]:
    """robust_pct and the range it came from, as percentages.

    Robustness is 100 minus the worst degradation, so the *pessimistic*
    end of the band uses the p90 degradation and the optimistic end uses
    p10. Returns (value, low, high); low and high are None when the
    result came from a single draw and there is no range to report.
    """
    worst = max(result.monte_carlo_degradation_pct, result.noise_degradation_pct)
    value = 100 - worst
    if result.noise_degradation_p10_pct is None or result.noise_degradation_p90_pct is None:
        return value, None, None
    low = 100 - max(result.monte_carlo_degradation_pct, result.noise_degradation_p90_pct)
    high = 100 - max(result.monte_carlo_degradation_pct, result.noise_degradation_p10_pct)
    return value, low, high


def sample_noise_runs(
    run: Callable[[NoiseParams], list[Trade]],
    rng: np.random.Generator,
    samples: int = NOISE_SAMPLES,
) -> list[list[Trade]]:
    """Re-simulates the genome under `samples` independent perturbations.

    `run` takes one draw and returns the trades it produced; the caller
    owns the genome and the backtest arguments, this module owns the
    distribution being sampled.
    """
    return [run(draw_noise_params(rng)) for _ in range(samples)]


def evaluate_robustness(
    oos_trades: list[Trade],
    df: pd.DataFrame,
    initial_balance: float,
    point_size: float,
    rng: np.random.Generator | None = None,
    risk_pct: float = 0.01,
    noisy_trades: list[Trade] | None = None,
    noisy_runs: list[list[Trade]] | None = None,
) -> RobustnessResult:
    """Baseline and both perturbations are expressed the same way — a
    compounded R-multiple sequence at the campaign's risk fraction — so the
    degradation percentages compare like with like."""
    rng = rng or np.random.default_rng()
    baseline_return = _r_sequence_return(oos_trades, risk_pct)

    mc_returns, mc_drawdowns = monte_carlo_trade_sequences(oos_trades, initial_balance, rng=rng, risk_pct=risk_pct)
    mc_median = float(np.median(mc_returns)) if mc_returns else baseline_return
    mc_p5 = float(np.percentile(mc_returns, 5)) if mc_returns else baseline_return
    mc_worst_dd = float(np.percentile(mc_drawdowns, 95)) if mc_drawdowns else 0.0
    # The gate uses the median of the bootstrap: the p5 tail is reported for
    # the Inspector, but gating on it would reject any strategy with normal
    # trade-level variance, which is every real strategy.
    mc_degradation = compute_degradation_pct(baseline_return, mc_median)

    # `noisy_runs` are re-simulations of the same genome under independent
    # perturbations (the caller has the genome; this module does not).
    # `noisy_trades` is the single-draw form, kept for callers that have
    # one — it is a sample of one and reads as such in the result.
    noise_p10 = noise_p90 = None
    if noisy_runs:
        returns = [_r_sequence_return(trades, risk_pct) for trades in noisy_runs]
        degradations = [compute_degradation_pct(baseline_return, value) for value in returns]
        noise_return = float(np.median(returns))
        noise_degradation = float(np.median(degradations))
        noise_p10 = float(np.percentile(degradations, 10))
        noise_p90 = float(np.percentile(degradations, 90))
        noise_sample_count = len(noisy_runs)
    else:
        if noisy_trades is None:
            noisy_trades = inject_noise(oos_trades, df, point_size, rng=rng)
        noise_return = _r_sequence_return(noisy_trades, risk_pct)
        noise_degradation = compute_degradation_pct(baseline_return, noise_return)
        noise_sample_count = 1

    # The gate reads the median of the sampled noise.
    #
    # It briefly read p90 instead, on the reasoning that a gate is a
    # safety question and the degradation is bimodal - clustered by
    # whether the entry slips one bar or two, not spread around a centre.
    # That reasoning was sound and the consequence was not measured until
    # afterwards. Across 73 real Phase 4 candidates, 365 draws: 63.6% of
    # draws degrade a candidate by more than half, so p90 lands in the
    # worse cluster for nearly everyone. It approved 3 of 73 where the
    # median approved 14, against 17 of 80 on the last campaign before
    # any of this changed. That is not a stricter gate, it is a closed
    # door - a HIGH campaign ran ten generations over a thousand genomes
    # and put nothing in the Vault.
    #
    # So the decision uses the median, which is what the sampling was for
    # in the first place. The spread is still measured and still
    # reported: p10 and p90 travel with the result and reach the
    # Inspector, where a range of 15 to 88 says something about a
    # strategy that "72%" on its own hides.
    # The median of a bootstrap sits on the baseline by construction, so
    # the degradation test above almost never fails - it could not tell a
    # real edge from a handful of lucky trades. The 5th percentile can:
    # when one resample in twenty loses money, the out-of-sample return is
    # not distinguishable from zero at 95%, whatever its median says.
    mc_significant = mc_p5 > 0
    passed = (
        baseline_return > 0
        and mc_significant
        and mc_degradation <= DEGRADATION_DISCARD_THRESHOLD_PCT
        and noise_degradation <= DEGRADATION_DISCARD_THRESHOLD_PCT
    )
    fail_reason = None
    if not passed:
        if baseline_return <= 0:
            fail_reason = "no out-of-sample edge (baseline OOS return is not positive)"
        elif not mc_significant:
            fail_reason = "Monte Carlo: 1 in 20 resampled trade sequences loses money (edge not significant at 95%)"
        else:
            worst = "Monte Carlo" if mc_degradation > noise_degradation else "noise injection"
            fail_reason = f"{worst} degradation exceeds {DEGRADATION_DISCARD_THRESHOLD_PCT}% threshold"

    return RobustnessResult(
        baseline_return_pct=baseline_return,
        monte_carlo_median_return_pct=mc_median,
        monte_carlo_worst_return_pct=mc_p5,
        monte_carlo_degradation_pct=mc_degradation,
        monte_carlo_worst_drawdown_pct=mc_worst_dd,
        noise_return_pct=noise_return,
        noise_degradation_pct=noise_degradation,
        noise_degradation_p10_pct=noise_p10,
        noise_degradation_p90_pct=noise_p90,
        noise_samples=noise_sample_count,
        passed=passed,
        fail_reason=fail_reason,
    )
