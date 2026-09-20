"""The 5-phase evolution pipeline — REQ-ENGINE-3, design.md B.5.

Phase 0 (Screening) and Phases 1-3 only ever see the in-sample (IS) 70% of
history; the locked out-of-sample (OOS) 30% is exposed to the model for
the first time in Phase 4 (REQ-ENGINE-3.6).
"""

from __future__ import annotations

import os
import random
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from app.engine.backtest import BacktestResult, simulate_trades
from app.engine.fitness import fitness_f1_stability, fitness_f2_expansion, passes_phase_0_screening
from app.engine.hash_id import assign_hash_id
from app.engine.metrics import TradeStats, compute_trade_stats
from app.engine.monte_carlo import RobustnessResult, draw_noise_params, evaluate_robustness
from app.engine.news_filter import EconomicEvent, NewsMode
from app.engine.resources import limit_worker_threads, max_parallel_workers
from app.engine.walk_forward import WalkForwardResult, run_walk_forward
from app.genome.gene_pool import GenePoolConfig
from app.genome.genome import Genome, crossover, mutate

IS_FRACTION = 0.7
DEFAULT_INITIAL_BALANCE = 10_000.0
DEFAULT_RISK_PCT = 0.01
DEFAULT_ELITE_FRACTION = 0.2
DEFAULT_MUTATION_RATE = 0.15
DEFAULT_MAX_BREEDING_POPULATION = 200
# Phase 4 approval needs enough out-of-sample trades for the OOS stats to
# carry information. Scaled from Phase 0's 100-trade rule (REQ-ENGINE-3.2)
# by the OOS share of history: 100 x 30% = 30.
MIN_OOS_TRADES = 30
# Share of walk-forward folds a strategy must have been profitable in.
# Half is a deliberately modest bar: the point is to reject strategies that
# worked in exactly one period, not to demand a strategy that never has a
# losing stretch — no real one exists.
MIN_PERSISTENCE_PCT = 50.0


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
    equity_is: list[dict] | None = None
    equity_oos: list[dict] | None = None
    walk_forward: WalkForwardResult | None = None


EQUITY_CURVE_MAX_POINTS = 500


def serialize_equity_curve(df: pd.DataFrame, curve, max_points: int = EQUITY_CURVE_MAX_POINTS) -> list[dict] | None:
    """Downsamples an equity curve to at most `max_points` (t, equity)
    points for the Vault Inspector's IS/OOS chart — REQ-VAULT-1.4."""
    if curve is None or len(curve) == 0 or df.empty:
        return None
    timestamps = pd.DatetimeIndex(df["timestamp"])
    # A candidate evaluated on a different slice than `df` (possible when a
    # caller mixes frames) must truncate rather than index past the end.
    length = min(len(curve), len(timestamps))
    if length == 0:
        return None
    step = max(1, length // max_points)
    indices = list(range(0, length, step))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return [{"t": timestamps[i].isoformat(), "equity": round(float(curve[i]), 2)} for i in indices]


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
    **cost_kwargs,
) -> CandidateResult:
    result = simulate_trades(
        df, genome, gene_pool, htf_df, news_events, news_mode,
        initial_balance, risk_pct, max_leverage_breach_balance=max_leverage_breach_balance,
        **cost_kwargs,  # point_size / cost_points / commission_per_lot / contract_size
    )
    stats = compute_trade_stats(result, df, initial_balance, timeframe)
    return CandidateResult(genome=genome, stats=stats, result=result)


# Candidates are independent single backtests — within a generation there
# is no dependency between them — so every phase parallelizes across cores
# (Phase 0's whole population; Phases 1-3 one generation at a time, since a
# generation's parents are the previous generation's survivors).
# REQ-DASH-1.5's duration estimate is explicitly "based on detected CPU
# cores," which is a real promise, not a cosmetic one, to keep. The
# initializer sends `df`/`gene_pool`/kwargs to each worker process ONCE
# instead of re-pickling them per candidate.
_worker_state: dict = {}


def _init_phase_0_worker(df: pd.DataFrame, gene_pool: GenePoolConfig, eval_kwargs: dict) -> None:
    limit_worker_threads()
    _worker_state["df"] = df
    _worker_state["gene_pool"] = gene_pool
    _worker_state["eval_kwargs"] = eval_kwargs


def _evaluate_in_worker(genome: Genome) -> CandidateResult:
    return evaluate_candidate(genome, _worker_state["df"], _worker_state["gene_pool"], **_worker_state["eval_kwargs"])


# Below this, a fresh process pool's startup cost (interpreter + numba's
# on-disk JIT cache load) outweighs whatever parallel speedup it would buy.
_MIN_POPULATION_FOR_PARALLEL_SCREENING = 24


def evaluate_population(
    genomes: list[Genome],
    df: pd.DataFrame,
    gene_pool: GenePoolConfig,
    max_workers: int | None = None,
    executor: ProcessPoolExecutor | None = None,
    **eval_kwargs,
) -> list[CandidateResult]:
    """Backtests a batch of genomes, across cores when the batch is worth
    it. `executor` lets a caller (the generational loop) reuse one pool for
    every generation instead of paying pool startup per generation."""
    workers = max_workers if max_workers is not None else (os.cpu_count() or 1)

    if executor is not None:
        chunksize = max(1, len(genomes) // (max(workers, 1) * 4))
        return list(executor.map(_evaluate_in_worker, genomes, chunksize=chunksize))

    # Cores are not the only budget — see app/engine/resources.py.
    workers = max_parallel_workers(workers, df)
    if workers <= 1 or len(genomes) < _MIN_POPULATION_FOR_PARALLEL_SCREENING:
        return [evaluate_candidate(genome, df, gene_pool, **eval_kwargs) for genome in genomes]

    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_phase_0_worker, initargs=(df, gene_pool, eval_kwargs)
    ) as pool:
        chunksize = max(1, len(genomes) // (workers * 4))
        return list(pool.map(_evaluate_in_worker, genomes, chunksize=chunksize))


def run_phase_0_screening(
    population: list[Genome], df: pd.DataFrame, gene_pool: GenePoolConfig, max_workers: int | None = None, **eval_kwargs
) -> PhaseOutcome:
    """REQ-ENGINE-3.1/3.2 — 4,000 random candidates in, PF/trades/margin gate applied."""
    results = evaluate_population(population, df, gene_pool, max_workers=max_workers, **eval_kwargs)

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
    max_workers: int | None = None,
    max_population: int = DEFAULT_MAX_BREEDING_POPULATION,
    **eval_kwargs,
) -> PhaseOutcome:
    """Iterated select -> crossover -> mutate -> evaluate loop within one
    tribe, ranked by `fitness_fn`. `on_generation_complete` fires after
    every generation for checkpointing (REQ-ENGINE-4.1) and is also what
    feeds the BREED screen's per-generation Fitness Radar (REQ-BREED-1.3).

    The breeding pool is capped at `max_population`: Phase 0 can hand over
    thousands of survivors, and carrying all of them through every
    generation multiplies the campaign's runtime without improving the
    search — the weakest survivors would never be selected as parents anyway."""
    population = list(seed_survivors)
    if start_generation == 0:
        # Resumed populations (start_generation > 0) already carry their
        # fitness from the checkpoint — recomputing would need `stats`,
        # which isn't persisted (see CandidateResult docstring).
        for candidate in population:
            candidate.fitness = fitness_fn(candidate.stats)
            candidate.tribe = tribe_label
    population.sort(key=lambda c: c.fitness, reverse=True)
    population = population[:max_population]

    pool_size = len(population)
    elite_count = max(1, int(pool_size * elite_fraction))
    children_per_generation = max(1, pool_size - elite_count)

    workers = max_parallel_workers(max_workers if max_workers is not None else (os.cpu_count() or 1), df)
    use_pool = workers > 1 and children_per_generation >= _MIN_POPULATION_FOR_PARALLEL_SCREENING

    with ExitStack() as stack:
        executor = None
        if use_pool:
            # One pool for the whole phase: re-creating it per generation
            # would pay interpreter + numba JIT-cache startup every time.
            executor = stack.enter_context(
                ProcessPoolExecutor(
                    max_workers=workers, initializer=_init_phase_0_worker, initargs=(df, gene_pool, eval_kwargs)
                )
            )

        for generation in range(start_generation, generations):
            elites = population[:elite_count]
            child_genomes = []
            for _ in range(children_per_generation):
                # Each child draws its own pair of elite parents — a fixed pair
                # per generation would collapse the whole tribe onto two lineages.
                parent_a, parent_b = random.sample(elites, 2) if len(elites) >= 2 else (elites[0], elites[0])
                child_genomes.append(mutate(crossover(parent_a.genome, parent_b.genome), gene_pool, mutation_rate))

            children = evaluate_population(
                child_genomes, df, gene_pool, max_workers=workers, executor=executor, **eval_kwargs
            )
            for candidate in children:
                candidate.fitness = fitness_fn(candidate.stats)
                candidate.tribe = tribe_label

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
    max_workers: int | None = None,
    max_population: int = DEFAULT_MAX_BREEDING_POPULATION,
    **eval_kwargs,
) -> PhaseOutcome:
    """REQ-ENGINE-3.3 — Tribe A, ranked by F1 = (Net Profit / Max DD%) × Win Rate × Sharpe."""
    return run_generational_phase(
        phase_0_survivors, fitness_f1_stability, "A", gene_pool, df, generations, top_n,
        on_generation_complete=on_generation_complete, start_generation=start_generation,
        max_workers=max_workers, max_population=max_population, **eval_kwargs,
    )


def run_phase_2_expansion(
    phase_0_survivors: list[CandidateResult],
    gene_pool: GenePoolConfig,
    df: pd.DataFrame,
    generations: int,
    top_n: int,
    on_generation_complete: GenerationCheckpointCallback | None = None,
    start_generation: int = 0,
    max_workers: int | None = None,
    max_population: int = DEFAULT_MAX_BREEDING_POPULATION,
    **eval_kwargs,
) -> PhaseOutcome:
    """REQ-ENGINE-3.4 — Tribe B, ranked by F2 = CAGR% × Profit Factor × Payoff Ratio.
    Runs on the same Phase 0 survivor pool as Phase 1, independently (design.md B.5: "parallel with Phase 1")."""
    return run_generational_phase(
        phase_0_survivors, fitness_f2_expansion, "B", gene_pool, df, generations, top_n,
        on_generation_complete=on_generation_complete, start_generation=start_generation,
        max_workers=max_workers, max_population=max_population, **eval_kwargs,
    )


def run_phase_3_hybridization(
    tribe_a: list[CandidateResult],
    tribe_b: list[CandidateResult],
    gene_pool: GenePoolConfig,
    df: pd.DataFrame,
    num_children: int,
    max_workers: int | None = None,
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

    child_genomes = [
        crossover(tribe_a[i % len(tribe_a)].genome, tribe_b[i % len(tribe_b)].genome) for i in range(num_children)
    ]
    children = evaluate_population(child_genomes, df, gene_pool, max_workers=max_workers, **eval_kwargs)
    for candidate in children:
        candidate.tribe = "C"

    return PhaseOutcome(passed=True, survivors=children)


def run_phase_4_robustness(
    tribe_c: list[CandidateResult],
    full_df: pd.DataFrame,
    gene_pool: GenePoolConfig,
    point_size: float = 0.0001,
    is_fraction: float = IS_FRACTION,
    initial_balance: float = DEFAULT_INITIAL_BALANCE,
    rng: np.random.Generator | None = None,
    **eval_kwargs,
) -> list[VaultCandidate]:
    """REQ-ENGINE-3.6-3.11 — 70/30 IS/OOS split (OOS seen for the first
    time here), 100 Monte Carlo trade-sequence randomizations, noise
    injection, ICIR/AVG HOLD, Hash ID assignment on pass."""
    is_df, oos_df = split_is_oos(full_df, is_fraction)
    approved: list[VaultCandidate] = []
    seen: set[str] = set()

    for candidate in tribe_c:
        # Phase 3 crosses a small elite pool, so many children are exact
        # copies of each other — the same strategy must not enter the Vault
        # several times under different Hash IDs.
        fingerprint = candidate.genome.fingerprint()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)

        oos_result = simulate_trades(
            oos_df, candidate.genome, gene_pool,
            initial_balance=initial_balance,
            # `point_size` is a named parameter here (the noise injection
            # needs it too), so it has to be handed back to the backtester
            # explicitly — the OOS run must price costs exactly like the IS run.
            point_size=point_size,
            **{k: v for k, v in eval_kwargs.items() if k not in ("timeframe", "initial_balance", "point_size")},
        )
        timeframe = eval_kwargs.get("timeframe", "H1")
        oos_stats = compute_trade_stats(oos_result, oos_df, initial_balance, timeframe)
        if oos_stats.trade_count < MIN_OOS_TRADES or oos_stats.net_profit_pct <= 0:
            # Two gates, both about evidence: a strategy that loses money on
            # the locked 30% has no out-of-sample edge, and one that made a
            # handful of trades has no evidence either way — a 2-trade,
            # +0.4% sample would otherwise enter the Vault scoring 94.
            continue

        # REQ-ENGINE-3.8 — the perturbed run is a real re-simulation of the
        # same genome with extra spread/slippage and a delayed fill, so the
        # stop and target respond to the perturbation as they would live.
        noise = draw_noise_params(rng)
        noisy_result = simulate_trades(
            oos_df, candidate.genome, gene_pool,
            initial_balance=initial_balance,
            point_size=point_size,
            entry_delay_bars=noise.entry_delay_bars,
            **{
                **{k: v for k, v in eval_kwargs.items() if k not in ("timeframe", "initial_balance", "point_size", "cost_points")},
                "cost_points": eval_kwargs.get("cost_points", 0.0) + noise.extra_cost_points,
            },
        )

        robustness = evaluate_robustness(
            oos_result.trades, oos_df, initial_balance, point_size,
            risk_pct=eval_kwargs.get("risk_pct", DEFAULT_RISK_PCT),
            noisy_trades=noisy_result.trades,
            rng=rng,
        )
        if not robustness.passed:
            continue

        # Walk-forward: the 70/30 split tests one period, which a strategy
        # can pass by being right about one regime. This re-tests it on
        # consecutive slices and requires it to have been profitable in
        # most of them. Short histories yield no folds, in which case
        # persistence is unknown (None) and cannot gate anything.
        walk_forward = run_walk_forward(
            candidate.genome, full_df, gene_pool,
            timeframe=eval_kwargs.get("timeframe", "H1"),
            initial_balance=initial_balance,
            point_size=point_size,
            train_bars=len(is_df),
            **{
                k: v
                for k, v in eval_kwargs.items()
                if k in ("cost_points", "commission_per_lot", "contract_size", "risk_pct", "htf_df", "news_events", "news_mode")
            },
        )
        if walk_forward is not None and walk_forward.persistence_pct < MIN_PERSISTENCE_PCT:
            continue

        approved.append(
            VaultCandidate(
                genome=candidate.genome,
                hash_id=assign_hash_id(candidate.genome),
                stats_is=candidate.stats,
                stats_oos=oos_stats,
                robustness=robustness,
                equity_is=serialize_equity_curve(is_df, candidate.result.equity_curve) if candidate.result else None,
                equity_oos=serialize_equity_curve(oos_df, oos_result.equity_curve),
                walk_forward=walk_forward,
            )
        )

    return approved
