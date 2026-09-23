"""Re-validating a Vault strategy against current data — REQ-VAULT-2.1.

A strategy's Vault row is a snapshot of the data that existed when it was
bred. Weeks later there is more history, and the honest question is not
"what did it score then?" but "does it still hold up now?".

REFRESH re-runs the strategy over the latest cached bars and updates its
stored stats. REASSESS does that and re-applies the Phase 4 gates —
robustness and walk-forward persistence — so a strategy that no longer
clears them says so instead of keeping a score it earned on old data.

Neither re-optimises anything: the genome is fixed. That matters, because
"re-fit it until it looks good again" is how a Vault fills with strategies
that have only ever been fitted.
"""

from __future__ import annotations

import hashlib

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy.orm import object_session

from app.engine.backtest import simulate_trades
from app.engine.metrics import compute_trade_stats
from app.engine.monte_carlo import evaluate_robustness, robustness_band, sample_noise_runs
from app.engine.phases import (
    IS_FRACTION,
    MIN_OOS_TRADES,
    MIN_PERSISTENCE_PCT,
    serialize_equity_curve,
    split_is_oos,
)
from app.engine.scoring import compute_grade, compute_score
from app.engine.trade_log import replace_trades
from app.engine.walk_forward import run_walk_forward
from app.genome.gene_pool import GenePoolConfig
from app.genome.genome import Genome
from app.models import Strategy


@dataclass
class ReassessResult:
    strategy_id: str
    bars: int
    trades: int
    return_pct: float
    max_dd_pct: float
    previous_score: int | None
    score: int | None = None
    grade: str | None = None
    persistence_pct: float | None = None
    robust_pct: float | None = None
    still_passes: bool | None = None
    failed_gate: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}


def _seed_for(strategy_id: str) -> int:
    """A stable seed per strategy.

    Not to remove the randomness - the noise test has to be random to
    mean anything - but so that re-running the same assessment gives the
    same answer. Deriving it from the id rather than fixing one constant
    keeps different strategies on different draws, so no single lucky
    perturbation flatters the whole Vault.
    """
    return int(hashlib.sha256(strategy_id.encode()).hexdigest()[:16], 16)


def _backtest_kwargs(config) -> dict:
    return {
        "point_size": config.point_size,
        "cost_points": config.cost_points,
        "commission_per_lot": config.commission_per_lot,
        "contract_size": config.contract_size,
        "risk_pct": config.risk_pct,
    }


def refresh_strategy(
    strategy: Strategy, df: pd.DataFrame, gene_pool: GenePoolConfig, config
) -> ReassessResult:
    """Re-runs the stored genome over `df` and updates the row's stats."""
    genome = Genome.from_dict(_genome_dict(strategy))
    is_df, oos_df = split_is_oos(df, IS_FRACTION)
    kwargs = _backtest_kwargs(config)

    oos_result = simulate_trades(
        oos_df, genome, gene_pool, initial_balance=config.initial_balance, **kwargs
    )
    oos_stats = compute_trade_stats(oos_result, oos_df, config.initial_balance, strategy.timeframe)
    is_result = simulate_trades(is_df, genome, gene_pool, initial_balance=config.initial_balance, **kwargs)

    previous_score = strategy.score
    strategy.trades = oos_stats.trade_count
    strategy.return_pct = oos_stats.net_profit_pct
    strategy.cagr_pct = oos_stats.cagr_pct
    strategy.max_dd_pct = oos_stats.max_dd_pct
    strategy.icir = oos_stats.icir
    strategy.win_rate_pct = oos_stats.win_rate * 100
    strategy.avg_hold_hours = oos_stats.avg_hold_hours
    strategy.equity_is = _json(serialize_equity_curve(is_df, is_result.equity_curve))
    strategy.equity_oos = _json(serialize_equity_curve(oos_df, oos_result.equity_curve))
    # A REFRESH re-runs the same genome over newer data, so the stored
    # trades are replaced rather than added to: keeping both runs would
    # leave a history that belongs to no single backtest.
    session = object_session(strategy)
    if session is not None:
        replace_trades(session, strategy.id, oos_result.trades, oos_df)

    return ReassessResult(
        strategy_id=strategy.id,
        bars=len(df),
        trades=oos_stats.trade_count,
        return_pct=round(oos_stats.net_profit_pct, 2),
        max_dd_pct=round(oos_stats.max_dd_pct, 2),
        previous_score=previous_score,
    )


def reassess_strategy(
    strategy: Strategy, df: pd.DataFrame, gene_pool: GenePoolConfig, config
) -> ReassessResult:
    """REFRESH plus the Phase 4 gates, re-scored against current data."""
    result = refresh_strategy(strategy, df, gene_pool, config)
    genome = Genome.from_dict(_genome_dict(strategy))
    is_df, oos_df = split_is_oos(df, IS_FRACTION)
    kwargs = _backtest_kwargs(config)

    oos_result = simulate_trades(oos_df, genome, gene_pool, initial_balance=config.initial_balance, **kwargs)
    oos_stats = compute_trade_stats(oos_result, oos_df, config.initial_balance, strategy.timeframe)

    if oos_stats.trade_count < MIN_OOS_TRADES:
        result.still_passes = False
        result.failed_gate = f"only {oos_stats.trade_count} out-of-sample trades (needs {MIN_OOS_TRADES})"
        return result
    if oos_stats.net_profit_pct <= 0:
        result.still_passes = False
        result.failed_gate = "no longer profitable out of sample"
        return result

    # Two sources of randomness used to run unseeded from here: the noise
    # draw and the bootstrap. Measured on a real strategy, six identical
    # REASSESS runs gave scores of 64 to 80, robustness of 36.6% to 100%,
    # and a pass/fail verdict that flipped - on the same strategy, the
    # same bars and the same settings. The randomness is the point of the
    # test, so it stays; what changes is that the answer now comes from
    # the distribution rather than from one draw of it, and that the seed
    # is derived from the strategy so pressing the button twice cannot
    # give two answers.
    rng = np.random.default_rng(_seed_for(strategy.id))

    def run_noisy(noise) -> list:
        return simulate_trades(
            oos_df, genome, gene_pool, initial_balance=config.initial_balance,
            entry_delay_bars=noise.entry_delay_bars,
            **{**kwargs, "cost_points": kwargs["cost_points"] + noise.extra_cost_points},
        ).trades

    robustness = evaluate_robustness(
        oos_result.trades, oos_df, config.initial_balance, config.point_size,
        risk_pct=config.risk_pct, rng=rng,
        noisy_runs=sample_noise_runs(run_noisy, rng),
    )

    walk_forward = run_walk_forward(
        genome, df, gene_pool, timeframe=strategy.timeframe,
        initial_balance=config.initial_balance, train_bars=len(is_df), **kwargs,
    )

    score = compute_score(oos_stats, robustness)
    strategy.score = score
    strategy.grade = compute_grade(score, oos_stats.trade_count)
    robust_value, robust_low, robust_high = robustness_band(robustness)
    strategy.robust_pct = robust_value
    strategy.robust_low_pct = robust_low
    strategy.robust_high_pct = robust_high
    if walk_forward is not None:
        strategy.persistence_pct = walk_forward.persistence_pct
        strategy.walk_forward = _json(walk_forward.to_dict())

    result.score = score
    result.grade = strategy.grade
    result.robust_pct = round(strategy.robust_pct, 1)
    result.persistence_pct = walk_forward.persistence_pct if walk_forward else None

    if not robustness.passed:
        result.still_passes = False
        result.failed_gate = robustness.fail_reason
    elif result.persistence_pct is not None and result.persistence_pct < MIN_PERSISTENCE_PCT:
        result.still_passes = False
        result.failed_gate = (
            f"profitable in only {result.persistence_pct}% of the walk-forward windows after "
            f"its training period (needs {MIN_PERSISTENCE_PCT}%)"
        )
    else:
        result.still_passes = True

    return result


def _genome_dict(strategy: Strategy) -> dict:
    import json

    return json.loads(strategy.genome_json)


def _json(value) -> str | None:
    import json

    return json.dumps(value) if value else None
