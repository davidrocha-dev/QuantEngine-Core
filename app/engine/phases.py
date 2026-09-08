"""The 5-phase evolution pipeline — REQ-ENGINE-3, design.md B.5.

Phase 0 (Screening) and Phases 1-3 only ever see the in-sample (IS) 70% of
history; the locked out-of-sample (OOS) 30% is exposed to the model for
the first time in Phase 4 (REQ-ENGINE-3.6).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from app.engine.backtest import BacktestResult, simulate_trades
from app.engine.fitness import fitness_f1_stability, fitness_f2_expansion, passes_phase_0_screening
from app.engine.hash_id import assign_hash_id
from app.engine.metrics import TradeStats, compute_trade_stats
from app.engine.monte_carlo import RobustnessResult, evaluate_robustness
from app.engine.news_filter import EconomicEvent, NewsMode
from app.genome.gene_pool import GenePoolConfig
from app.genome.genome import Genome, crossover, mutate

IS_FRACTION = 0.7
DEFAULT_INITIAL_BALANCE = 10_000.0
DEFAULT_RISK_PCT = 0.01
DEFAULT_ELITE_FRACTION = 0.2
DEFAULT_MUTATION_RATE = 0.15


@dataclass
class CandidateResult:
    genome: Genome
    # Optional so a checkpoint-resumed candidate (genome + fitness only,
    # per REQ-ENGINE-4.3) doesn't need its full BacktestResult round-tripped
    # through JSON — see app/engine/campaign.py `load_checkpoint`.
    stats: TradeStats | None = None
    result: BacktestResult | None = None
    tribe: str | None = None
    fitness: float | None = None


@dataclass
class PhaseOutcome:
    passed: bool
    survivors: list[CandidateResult] = field(default_factory=list)
    rejected: list[tuple[Genome, str]] = field(default_factory=list)
    fail_reason: str | None = None


@dataclass
class VaultCandidate:
    genome: Genome
    hash_id: str
    stats_is: TradeStats
    stats_oos: TradeStats
    robustness: RobustnessResult


def split_is_oos(df: pd.DataFrame, is_fraction: float = IS_FRACTION) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split — time series data must never be split randomly."""
    split_idx = int(len(df) * is_fraction)
    is_df = df.iloc[:split_idx].reset_index(drop=True)
    oos_df = df.iloc[split_idx:].reset_index(drop=True)
    return is_df, oos_df


def evaluate_candidate(
    genome: Genome,
    df: pd.DataFrame,
    gene_pool: GenePoolConfig,
    htf_df: pd.DataFrame | None = None,
    news_events: list[EconomicEvent] | None = None,
    news_mode: NewsMode = "FTMO",
    initial_balance: float = DEFAULT_INITIAL_BALANCE,
    risk_pct: float = DEFAULT_RISK_PCT,
    timeframe: str = "H1",
    max_leverage_breach_balance: float = 0.0,
) -> CandidateResult:
    result = simulate_trades(
        df, genome, gene_pool, htf_df, news_events, news_mode,
        initial_balance, risk_pct, max_leverage_breach_balance=max_leverage_breach_balance,
    )
    stats = compute_trade_stats(result, df, initial_balance, timeframe)
    return CandidateResult(genome=genome, stats=stats, result=result)


# Phase 0's 4,000 candidates are fully independent single backtests — no
# selection/crossover dependency between them — so it's the one phase that
# parallelizes cleanly across cores. REQ-DASH-1.5's duration estimate is
# explicitly "based on detected CPU cores," which is a real promise, not a
# cosmetic one, to keep. The initializer sends `df`/`gene_pool`/kwargs to
# each worker process ONCE instead of re-pickling them per candidate.
_worker_state: dict = {}


def _init_phase_0_worker(df: pd.DataFrame, gene_pool: GenePoolConfig, eval_kwargs: dict) -> None:
    _worker_state["df"] = df
    _worker_state["gene_pool"] = gene_pool
    _worker_state["eval_kwargs"] = eval_kwargs


def _evaluate_in_worker(genome: Genome) -> CandidateResult:
    return evaluate_candidate(genome, _worker_state["df"], _worker_state["gene_pool"], **_worker_state["eval_kwargs"])


# Below this, a fresh process pool's startup cost (interpreter + numba's
# on-disk JIT cache load) outweighs whatever parallel speedup it would buy.
_MIN_POPULATION_FOR_PARALLEL_SCREENING = 200


def run_phase_0_screening(
    population: list[Genome], df: pd.DataFrame, gene_pool: GenePoolConfig, max_workers: int | None = None, **eval_kwargs
) -> PhaseOutcome:
    """REQ-ENGINE-3.1/3.2 — 4,000 random candidates in, PF/trades/margin gate applied."""
    workers = max_workers if max_workers is not None else (os.cpu_count() or 1)

    if workers <= 1 or len(population) < _MIN_POPULATION_FOR_PARALLEL_SCREENING:
        results = (evaluate_candidate(genome, df, gene_pool, **eval_kwargs) for genome in population)
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_phase_0_worker, initargs=(df, gene_pool, eval_kwargs)
        ) as executor:
            results = list(executor.map(_evaluate_in_worker, population, chunksize=max(1, len(population) // (workers * 4))))

    survivors, rejected = [], []
    for candidate in results:
        ok, reason = passes_phase_0_screening(candidate.stats, candidate.result)
        if ok:
            survivors.append(candidate)
        else:
            rejected.append((candidate.genome, reason or "screening failed"))
    return PhaseOutcome(passed=True, survivors=survivors, rejected=rejected)


GenerationCheckpointCallback = Callable[[int, list[CandidateResult]], None]


def run_generational_phase(
    seed_survivors: list[CandidateResult],
    fitness_fn: Callable[[TradeStats], float],
    tribe_label: str,
    gene_pool: GenePoolConfig,
    df: pd.DataFrame,
    generations: int,
    top_n: int,
    elite_fraction: float = DEFAULT_ELITE_FRACTION,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    on_generation_complete: GenerationCheckpointCallback | None = None,
    start_generation: int = 0,
    **eval_kwargs,
) -> PhaseOutcome:
    """Iterated select -> crossover -> mutate -> evaluate loop within one
    tribe, ranked by `fitness_fn`. `on_generation_complete` fires after
    every generation for checkpointing (REQ-ENGINE-4.1) and is also what
    feeds the BREED screen's per-generation Fitness Radar (REQ-BREED-1.3)."""
    population = list(seed_survivors)
    if start_generation == 0:
        # Resumed populations (start_generation > 0) already carry their
        # fitness from the checkpoint — recomputing would need `stats`,
        # which isn't persisted (see CandidateResult docstring).
        for candidate in population:
            candidate.fitness = fitness_fn(candidate.stats)
            candidate.tribe = tribe_label
    population.sort(key=lambda c: c.fitness, reverse=True)

    pool_size = len(population)
    elite_count = max(1, int(pool_size * elite_fraction))

    for generation in range(start_generation, generations):
        elites = population[:elite_count]
        children: list[CandidateResult] = []
        while len(children) < pool_size - elite_count:
            parent_a, parent_b = elites[generation % len(elites)], elites[(generation + 1) % len(elites)]
            child_genome = mutate(crossover(parent_a.genome, parent_b.genome), gene_pool, mutation_rate)
            candidate = evaluate_candidate(child_genome, df, gene_pool, **eval_kwargs)
            candidate.fitness = fitness_fn(candidate.stats)
            candidate.tribe = tribe_label
            children.append(candidate)

        population = elites + children
        population.sort(key=lambda c: c.fitness, reverse=True)

        if on_generation_complete is not None:
            on_generation_complete(generation, population)

    return PhaseOutcome(passed=True, survivors=population[:top_n])


def run_phase_1_stability(
    phase_0_survivors: list[CandidateResult],
    gene_pool: GenePoolConfig,
    df: pd.DataFrame,
    generations: int,
    top_n: int,
    on_generation_complete: GenerationCheckpointCallback | None = None,
    start_generation: int = 0,
    **eval_kwargs,
) -> PhaseOutcome:
    """REQ-ENGINE-3.3 — Tribe A, ranked by F1 = (Net Profit / Max DD%) × Win Rate × Sharpe."""
    return run_generational_phase(
        phase_0_survivors, fitness_f1_stability, "A", gene_pool, df, generations, top_n,
        on_generation_complete=on_generation_complete, start_generation=start_generation, **eval_kwargs,
    )


def run_phase_2_expansion(
    phase_0_survivors: list[CandidateResult],
    gene_pool: GenePoolConfig,
    df: pd.DataFrame,
    generations: int,
    top_n: int,
    on_generation_complete: GenerationCheckpointCallback | None = None,
    start_generation: int = 0,
    **eval_kwargs,
) -> PhaseOutcome:
    """REQ-ENGINE-3.4 — Tribe B, ranked by F2 = CAGR% × Profit Factor × Payoff Ratio.
    Runs on the same Phase 0 survivor pool as Phase 1, independently (design.md B.5: "parallel with Phase 1")."""
    return run_generational_phase(
        phase_0_survivors, fitness_f2_expansion, "B", gene_pool, df, generations, top_n,
        on_generation_complete=on_generation_complete, start_generation=start_generation, **eval_kwargs,
    )


def run_phase_3_hybridization(
    tribe_a: list[CandidateResult],
    tribe_b: list[CandidateResult],
    gene_pool: GenePoolConfig,
    df: pd.DataFrame,
    num_children: int,
    **eval_kwargs,
) -> PhaseOutcome:
    """REQ-ENGINE-3.5 — crossover top Phase 1 x Phase 2 individuals. Fails
    outright per REQ-ENGINE-1.4 when the Gene Pool has no Management genes
    enabled at all (no exit recalculation is mathematically viable)."""
    if gene_pool.all_management_disabled():
        return PhaseOutcome(
            passed=False,
            fail_reason="Not Passed: all Management genes are disabled — no exit recalculation is mathematically viable",
        )
    if not tribe_a or not tribe_b:
        return PhaseOutcome(passed=False, fail_reason="Not Passed: Tribe A or Tribe B produced no survivors to crossover")

    children: list[CandidateResult] = []
    for i in range(num_children):
        parent_a = tribe_a[i % len(tribe_a)]
        parent_b = tribe_b[i % len(tribe_b)]
        child_genome = crossover(parent_a.genome, parent_b.genome)
        candidate = evaluate_candidate(child_genome, df, gene_pool, **eval_kwargs)
        candidate.tribe = "C"
        children.append(candidate)

    return PhaseOutcome(passed=True, survivors=children)


def run_phase_4_robustness(
    tribe_c: list[CandidateResult],
    full_df: pd.DataFrame,
    gene_pool: GenePoolConfig,
    point_size: float,
    is_fraction: float = IS_FRACTION,
    initial_balance: float = DEFAULT_INITIAL_BALANCE,
    **eval_kwargs,
) -> list[VaultCandidate]:
    """REQ-ENGINE-3.6-3.11 — 70/30 IS/OOS split (OOS seen for the first
    time here), 100 Monte Carlo trade-sequence randomizations, noise
    injection, ICIR/AVG HOLD, Hash ID assignment on pass."""
    _, oos_df = split_is_oos(full_df, is_fraction)
    approved: list[VaultCandidate] = []

    for candidate in tribe_c:
        oos_result = simulate_trades(
            oos_df, candidate.genome, gene_pool,
            initial_balance=initial_balance, **{k: v for k, v in eval_kwargs.items() if k != "timeframe"},
        )
        timeframe = eval_kwargs.get("timeframe", "H1")
        oos_stats = compute_trade_stats(oos_result, oos_df, initial_balance, timeframe)
        if oos_stats.trade_count == 0:
            continue

        robustness = evaluate_robustness(oos_result.trades, oos_df, initial_balance, point_size)
        if not robustness.passed:
            continue

        approved.append(
            VaultCandidate(
                genome=candidate.genome,
                hash_id=assign_hash_id(candidate.genome),
                stats_is=candidate.stats,
                stats_oos=oos_stats,
                robustness=robustness,
            )
        )

    return approved
