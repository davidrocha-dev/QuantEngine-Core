"""Phase 4 robustness perturbations — REQ-ENGINE-3.7/3.8/3.10.

Two independent stress tests run against the OOS trade sequence:
  1. Trade-sequence Monte Carlo: reshuffle trade order 100x (same trades,
     different sequence) to test sequence-risk / drawdown-path sensitivity.
  2. Noise injection: perturb each trade with a random spread (+1 to +5
     points), slippage (0-15ms), and entry displacement (1-2 bars), then
     recompute the return.

A candidate is discarded if EITHER perturbation degrades its return by
more than 25% relative to the clean OOS baseline (REQ-ENGINE-3.10).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.engine.backtest import Trade

DEGRADATION_DISCARD_THRESHOLD_PCT = 25.0
MONTE_CARLO_ITERATIONS = 100
SPREAD_POINTS_RANGE = (1, 5)
SLIPPAGE_MS_RANGE = (0, 15)
DISPLACEMENT_BARS_RANGE = (1, 2)


@dataclass
class RobustnessResult:
    baseline_return_pct: float
    monte_carlo_worst_return_pct: float
    monte_carlo_degradation_pct: float
    noise_return_pct: float
    noise_degradation_pct: float
    passed: bool
    fail_reason: str | None = None


def _return_pct_from_sequence(pnls: list[float], initial_balance: float) -> float:
    balance = initial_balance
    for pnl in pnls:
        balance += pnl
    return (balance - initial_balance) / initial_balance * 100


def monte_carlo_trade_sequences(
    trades: list[Trade], initial_balance: float, iterations: int = MONTE_CARLO_ITERATIONS, rng: np.random.Generator | None = None
) -> list[float]:
    """REQ-ENGINE-3.7: 100 trade-sequence randomizations. Returns the list
    of resulting total-return percentages, one per iteration."""
    rng = rng or np.random.default_rng()
    pnls = [t.pnl for t in trades]
    if not pnls:
        return []
    results = []
    for _ in range(iterations):
        shuffled = rng.permutation(pnls).tolist()
        results.append(_return_pct_from_sequence(shuffled, initial_balance))
    return results


def inject_noise(
    trades: list[Trade],
    df: pd.DataFrame,
    point_size: float,
    rng: np.random.Generator | None = None,
) -> list[Trade]:
    """REQ-ENGINE-3.8: perturbs each trade with random spread/slippage/entry
    displacement. Slippage (given in ms, no tick-level replay available) is
    modeled as a spread-proportional cost scaled by ms/15 — a documented
    simplification, not a literal latency simulation."""
    rng = rng or np.random.default_rng()
    perturbed = []
    opens = df["open"].to_numpy()

    for trade in trades:
        displacement = rng.integers(DISPLACEMENT_BARS_RANGE[0], DISPLACEMENT_BARS_RANGE[1] + 1)
        displaced_idx = min(trade.entry_idx + displacement, len(opens) - 1)
        new_entry_price = float(opens[displaced_idx])

        spread_points = rng.integers(SPREAD_POINTS_RANGE[0], SPREAD_POINTS_RANGE[1] + 1)
        slippage_ms = rng.integers(SLIPPAGE_MS_RANGE[0], SLIPPAGE_MS_RANGE[1] + 1)
        spread_cost = spread_points * point_size
        slippage_cost = (slippage_ms / 15) * spread_cost
        total_cost_price = spread_cost + slippage_cost

        stop_distance = abs(trade.entry_price - trade.stop_price)
        if stop_distance <= 0:
            perturbed.append(trade)
            continue

        # Entry displacement shifts the effective entry price; the adverse
        # cost (spread+slippage) always works against the trade's direction.
        adjusted_entry = new_entry_price - trade.direction * total_cost_price
        r_multiple = (trade.exit_price - adjusted_entry) * trade.direction / stop_distance
        risk_amount = abs(trade.pnl / trade.r_multiple) if trade.r_multiple != 0 else 0.0
        pnl = risk_amount * r_multiple

        perturbed.append(
            Trade(
                entry_idx=displaced_idx, exit_idx=trade.exit_idx, direction=trade.direction,
                entry_price=adjusted_entry, exit_price=trade.exit_price, stop_price=trade.stop_price,
                r_multiple=r_multiple, pnl=pnl, exit_reason=trade.exit_reason, partial_closed=trade.partial_closed,
            )
        )
    return perturbed


def compute_degradation_pct(baseline_return_pct: float, perturbed_return_pct: float) -> float:
    if baseline_return_pct <= 0:
        return 100.0 if perturbed_return_pct < baseline_return_pct else 0.0
    return max(0.0, (baseline_return_pct - perturbed_return_pct) / baseline_return_pct * 100)


def evaluate_robustness(
    oos_trades: list[Trade],
    df: pd.DataFrame,
    initial_balance: float,
    point_size: float,
    rng: np.random.Generator | None = None,
) -> RobustnessResult:
    rng = rng or np.random.default_rng()
    baseline_return = _return_pct_from_sequence([t.pnl for t in oos_trades], initial_balance)

    mc_returns = monte_carlo_trade_sequences(oos_trades, initial_balance, rng=rng)
    mc_worst = min(mc_returns) if mc_returns else baseline_return
    mc_degradation = compute_degradation_pct(baseline_return, mc_worst)

    noisy_trades = inject_noise(oos_trades, df, point_size, rng=rng)
    noise_return = _return_pct_from_sequence([t.pnl for t in noisy_trades], initial_balance)
    noise_degradation = compute_degradation_pct(baseline_return, noise_return)

    passed = mc_degradation <= DEGRADATION_DISCARD_THRESHOLD_PCT and noise_degradation <= DEGRADATION_DISCARD_THRESHOLD_PCT
    fail_reason = None
    if not passed:
        worst = "Monte Carlo" if mc_degradation > noise_degradation else "noise injection"
        fail_reason = f"{worst} degradation exceeds {DEGRADATION_DISCARD_THRESHOLD_PCT}% threshold"

    return RobustnessResult(
        baseline_return_pct=baseline_return,
        monte_carlo_worst_return_pct=mc_worst,
        monte_carlo_degradation_pct=mc_degradation,
        noise_return_pct=noise_return,
        noise_degradation_pct=noise_degradation,
        passed=passed,
        fail_reason=fail_reason,
    )
