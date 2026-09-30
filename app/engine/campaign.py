"""Campaign orchestration: runs the 5-phase pipeline against a `campaigns`
DB row, checkpointing every completed generation (REQ-ENGINE-4.1) and
supporting crash-resume (REQ-ENGINE-4.2-4.4).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.engine.backtest import simulate_trades
from app.engine.fitness import PHASE_0_MIN_TRADES
from app.engine.metrics import TradeStats, compute_trade_stats
from app.config import get_config
from app.engine.control import CampaignControl, CampaignPaused
from app.engine.failure_signal import load as load_gene_outcomes
from app.engine.failure_signal import record_screening
from app.engine.failure_signal import save as save_gene_outcomes
from app.engine.phases import (
    DEFAULT_MAX_BREEDING_POPULATION,
    IS_FRACTION,
    CandidateResult,
    PhaseOutcome,
    run_phase_0_screening,
    run_phase_1_stability,
    run_phase_2_expansion,
    run_phase_3_hybridization,
    run_phase_4_robustness,
    split_search_holdout,
    split_is_oos,
)
from app.engine.vault_writer import HIGHER_TF_OFF, write_vault_candidates
from app.genome.gene_pool import GenePoolConfig
from app.genome.genome import Genome, random_population
from app.models import Campaign

# A campaign on a few days of data produces metrics (CAGR especially) that
# are arithmetically valid and practically meaningless — refuse instead.
MIN_HISTORY_DAYS = 90

# Phase 0 requires 100 trades per candidate (REQ-ENGINE-3.2). Measured on
# real EURUSD data, a random genome trades roughly once every 30 bars, so
# below ~3,000 in-sample bars almost nothing can clear that gate however
# good it is — the campaign would "run" for its full length and approve
# nothing. Re-measure this if the gene defaults change.
MIN_IS_BARS = 3_000

# Phase 0 keeps drawing candidates until the tribes have at least this many
# survivors to breed from, up to MAX_SCREENING_ROUNDS x the population size.
MIN_BREEDING_POPULATION = 40
MAX_SCREENING_ROUNDS = 5

PHASE_SCREENING = 0
PHASE_STABILITY = 1
PHASE_EXPANSION = 2
PHASE_HYBRIDIZATION = 3
PHASE_ROBUSTNESS = 4


@dataclass
class CampaignConfig:
    population_size: int = 4000
    generations_per_tribe: int = 20
    phase1_top_n: int = 200
    phase2_top_n: int = 200
    phase3_children: int = 80
    initial_balance: float = 10_000.0
    risk_pct: float = 0.01
    # Settings -> Risk -> MAX DRAWDOWN, as a Phase 4 gate. 0 disables it.
    max_drawdown_pct: float = 0.0
    # And the daily one, which is the limit a prop firm enforces hardest.
    max_daily_drawdown_pct: float = 0.0
    timeframe: str = "H1"
    point_size: float = 0.0001
    max_workers: int | None = None  # None = all detected cores
    max_breeding_population: int = DEFAULT_MAX_BREEDING_POPULATION
    # Settings → Costs, charged on every trade in every phase (see backtest.py).
    cost_points: float = 0.0
    commission_per_lot: float = 0.0
    contract_size: float = 100_000.0
    # Settings → Broker caps. Zero means continuous sizing, which flatters
    # every small account; see simulate_trades.
    lot_step: float = 0.0
    min_lot: float = 0.0
    max_lot: float = 0.0
    # Account currency per unit of the instrument's quote currency, by
    # timestamp (app/data/conversion.py). None when they are the same.
    quote_to_account: pd.Series | None = None
    # Overnight financing per lot per night (app/engine/swap.py); None = none charged.
    swap: object | None = None
    data_source: str = "dukascopy"
    # Smart Mode sampling weights (app/engine/smart_mode.py). Empty means
    # uniform sampling, which is what every campaign did before Smart Mode.
    gene_weights: dict = field(default_factory=dict)
    # REQ-BREED-2 — higher-timeframe context bars, None when HIGHER TF is off,
    # and the timeframe they are (recorded on every Vault row it produces).
    htf_df: pd.DataFrame | None = None
    higher_timeframe: str | None = None
    # REQ-ENGINE-2 — economic calendar for the EVENT gene, empty when none is configured.
    news_events: list = field(default_factory=list)
    news_mode: str = "FTMO"


def _history_span_days(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    timestamps = pd.DatetimeIndex(df["timestamp"])
    return int((timestamps.max() - timestamps.min()).total_seconds() // 86_400)


def _serialize_population(population: list[CandidateResult]) -> list[dict]:
    # `stats` is persisted too (not just genome/fitness/tribe): a resumed
    # population may need to seed a tribe that hasn't started yet, and
    # run_generational_phase recomputes fitness from `stats` at generation
    # 0 for any freshly-seeded tribe (see phases.py `run_generational_phase`).
    return [
        {"genome": c.genome.to_dict(), "fitness": c.fitness, "tribe": c.tribe, "stats": asdict(c.stats) if c.stats else None}
        for c in population
    ]


def _deserialize_population(raw: list[dict]) -> list[CandidateResult]:
    return [
        CandidateResult(
            genome=Genome.from_dict(entry["genome"]),
            stats=TradeStats(**entry["stats"]) if entry.get("stats") else None,
            tribe=entry.get("tribe"),
            fitness=entry.get("fitness"),
        )
        for entry in raw
    ]


def _checkpoint(
    session: Session,
    campaign_id: str,
    phase: int,
    generation: int,
    population: list[CandidateResult],
    phase_0_survivors: list[CandidateResult],
) -> None:
    """REQ-ENGINE-4.1 — checkpoint the population (genomes + fitness) at
    every completed generation within the active phase. Phase 0's
    survivors are carried in every checkpoint (not just the active tribe's
    population) so a resume that lands before the *other* tribe has
    started still has a seed population for it — Phase 0 itself is not
    re-run on resume, since it draws a fresh random population each time."""
    campaign = session.get(Campaign, campaign_id)
    if campaign is None:
        return
    campaign.phase = phase
    campaign.checkpoint_generation = generation
    campaign.checkpoint_population = json.dumps(
        {
            "phase_0_survivors": _serialize_population(phase_0_survivors),
            "active_population": _serialize_population(population),
        }
    )
    campaign.last_checkpoint_at = datetime.now(timezone.utc)
    session.commit()


def _set_survivors_per_phase(session: Session, campaign_id: str, phase: int, count: int) -> None:
    campaign = session.get(Campaign, campaign_id)
    if campaign is None:
        return
    survivors = json.loads(campaign.survivors_per_phase) if campaign.survivors_per_phase else {}
    survivors[str(phase)] = count
    campaign.survivors_per_phase = json.dumps(survivors)
    session.commit()


def mark_crashed_campaigns(session: Session) -> list[str]:
    """REQ-ENGINE-4.2 — called at engine startup. A `campaigns` row still
    marked "running" at that point means the previous engine process died
    mid-campaign (a live process would have kept updating it)."""
    stmt = select(Campaign).where(Campaign.status == "running")
    crashed_ids = []
    for campaign in session.scalars(stmt):
        campaign.status = "crashed"
        crashed_ids.append(campaign.id)
    session.commit()
    return crashed_ids


def discard_campaign(session: Session, campaign_id: str) -> None:
    """REQ-ENGINE-4.4."""
    campaign = session.get(Campaign, campaign_id)
    if campaign is None:
        raise ValueError(f"Unknown campaign: {campaign_id}")
    campaign.status = "cancelled"
    session.commit()


@dataclass
class LoadedCheckpoint:
    generation: int
    active_population: list[CandidateResult]
    phase_0_survivors: list[CandidateResult]


def load_checkpoint(campaign: Campaign) -> LoadedCheckpoint | None:
    """REQ-ENGINE-4.3 — reloads the last checkpoint's genomes + fitness."""
    if not campaign.checkpoint_population:
        return None
    raw = json.loads(campaign.checkpoint_population)
    return LoadedCheckpoint(
        generation=campaign.checkpoint_generation or 0,
        active_population=_deserialize_population(raw["active_population"]),
        phase_0_survivors=_deserialize_population(raw["phase_0_survivors"]),
    )


def run_campaign(
    session: Session,
    campaign_id: str,
    df: pd.DataFrame,
    gene_pool: GenePoolConfig,
    config: CampaignConfig,
    resume: bool = False,
    on_generation: Callable[[int, int, list[CandidateResult]], None] | None = None,
    control: CampaignControl | None = None,
) -> PhaseOutcome:
    """Runs Phase 0 -> 1 -> 2 -> 3 -> 4 -> Vault write end to end,
    checkpointing after every Phase 1/2 generation. `resume=True` reloads
    the last checkpoint (REQ-ENGINE-4.3) instead of starting from scratch.

    `on_generation(phase, generation, population)`, when given, fires
    synchronously at the same moment as each checkpoint — REQ-BREED-1.3's
    live Fitness Radar needs one event per generation, and a campaign
    against real (fast) data can blow through several generations between
    two ticks of an external poller, silently dropping them.

    Known resume limitation: only the tribe that was actively evolving at
    crash time resumes from its exact generation. If that tribe was B
    (meaning tribe A already finished before the crash), tribe A's
    generations are re-run from the Phase 0 seed rather than reusing its
    pre-crash result, since only the in-flight tribe's population is
    checkpointed. This still satisfies REQ-ENGINE-4.3 for the tribe that
    was actually interrupted; it costs some recomputation for the other."""
    campaign = session.get(Campaign, campaign_id)
    if campaign is None:
        raise ValueError(f"Unknown campaign: {campaign_id}")

    campaign.status = "running"
    session.commit()

    # The final holdout. Everything from screening to the Phase 4 gates
    # runs on the older part of history; the newest slice is measured once,
    # after approval, and decides nothing. With OOS as the last gate, the
    # OOS numbers in the Vault are the best of dozens of candidates on that
    # window - the holdout is the one figure selection never saw. Only
    # taken when what remains can still hold a full campaign.
    df, holdout_df = split_search_holdout(df, MIN_IS_BARS)

    is_df, _ = split_is_oos(df)
    span_days = _history_span_days(df)
    if span_days < MIN_HISTORY_DAYS:
        campaign.status = "cancelled"
        session.commit()
        return PhaseOutcome(
            passed=False,
            fail_reason=(
                f"Not Passed: only {span_days} days of {config.timeframe} history available — a campaign needs at "
                f"least {MIN_HISTORY_DAYS} days for its metrics to mean anything. Let the data sync finish and "
                "try again."
            ),
        )

    if len(is_df) < MIN_IS_BARS:
        campaign.status = "cancelled"
        session.commit()
        return PhaseOutcome(
            passed=False,
            fail_reason=(
                f"Not Passed: {len(df)} {config.timeframe} bars is too few — Phase 0 needs {PHASE_0_MIN_TRADES} "
                f"trades per candidate, which takes about {MIN_IS_BARS} in-sample bars "
                f"({int(MIN_IS_BARS / IS_FRACTION)} total). Sync more history, or breed on a lower timeframe."
            ),
        )

    eval_kwargs = {
        "initial_balance": config.initial_balance,
        "risk_pct": config.risk_pct,
        "timeframe": config.timeframe,
        "point_size": config.point_size,
        "cost_points": config.cost_points,
        "commission_per_lot": config.commission_per_lot,
        "contract_size": config.contract_size,
        "lot_step": config.lot_step,
        "min_lot": config.min_lot,
        "max_lot": config.max_lot,
        "quote_to_account": config.quote_to_account,
        "swap": config.swap,
        "htf_df": config.htf_df,
        "news_events": config.news_events or None,
        "news_mode": config.news_mode,
    }
    phase_kwargs = {"max_workers": config.max_workers, "control": control}

    loaded = load_checkpoint(campaign) if resume else None

    if loaded is not None:
        phase_0_survivors = loaded.phase_0_survivors
    else:
        # Screening keeps drawing fresh candidates until the tribes have a
        # pool worth evolving. With real costs charged, a single 4,000-draw
        # can leave a handful of survivors, and two tribes seeded from ~9
        # genomes inbreed immediately — every child descends from the same
        # two elites. Extra rounds are still screened by the same Phase 0
        # gate (REQ-ENGINE-3.2), so nothing unscreened reaches a tribe.
        phase_0_survivors: list[CandidateResult] = []
        drawn = 0
        max_draws = config.population_size * MAX_SCREENING_ROUNDS
        # What screening rejects is the only large sample of failure this
        # product ever sees, and it was being thrown away. Two counters
        # per gene turn it into something the next campaign can use.
        outcomes = load_gene_outcomes(get_config().data_dir)
        while drawn < max_draws:
            population = random_population(gene_pool, config.population_size, config.gene_weights or None)
            phase_0 = run_phase_0_screening(population, is_df, gene_pool, max_workers=config.max_workers, **eval_kwargs)
            record_screening(outcomes, population, [c.genome for c in phase_0.survivors])
            phase_0_survivors.extend(phase_0.survivors)
            drawn += config.population_size
            _set_survivors_per_phase(session, campaign_id, PHASE_SCREENING, len(phase_0_survivors))
            if len(phase_0_survivors) >= MIN_BREEDING_POPULATION:
                break

        outcomes.campaigns += 1
        save_gene_outcomes(get_config().data_dir, outcomes)

        if not phase_0_survivors:
            campaign.status = "cancelled"
            session.commit()
            return PhaseOutcome(
                passed=False,
                fail_reason=(
                    f"Not Passed: no candidate survived Phase 0 screening in {drawn} draws — with trading costs "
                    "charged, profit factor 1.1 over 100+ trades is a real bar. Try a longer history, a different "
                    "market/timeframe, or a wider Gene Pool."
                ),
            )

    def make_checkpoint_cb(phase: int):
        def _cb(generation: int, pop: list[CandidateResult]) -> None:
            _checkpoint(session, campaign_id, phase, generation, pop, phase_0_survivors)
            if on_generation is not None:
                on_generation(phase, generation, pop)
        return _cb

    if loaded is not None and campaign.phase == PHASE_STABILITY:
        tribe_a_seed, start_gen = loaded.active_population, loaded.generation
    else:
        tribe_a_seed, start_gen = phase_0_survivors, 0
    phase_1 = run_phase_1_stability(
        tribe_a_seed, gene_pool, is_df, config.generations_per_tribe, config.phase1_top_n,
        on_generation_complete=make_checkpoint_cb(PHASE_STABILITY), start_generation=start_gen,
        max_population=config.max_breeding_population, **phase_kwargs, **eval_kwargs,
    )
    _set_survivors_per_phase(session, campaign_id, PHASE_STABILITY, len(phase_1.survivors))

    if loaded is not None and campaign.phase == PHASE_EXPANSION:
        tribe_b_seed, start_gen = loaded.active_population, loaded.generation
    else:
        tribe_b_seed, start_gen = phase_0_survivors, 0
    phase_2 = run_phase_2_expansion(
        tribe_b_seed, gene_pool, is_df, config.generations_per_tribe, config.phase2_top_n,
        on_generation_complete=make_checkpoint_cb(PHASE_EXPANSION), start_generation=start_gen,
        max_population=config.max_breeding_population, **phase_kwargs, **eval_kwargs,
    )
    _set_survivors_per_phase(session, campaign_id, PHASE_EXPANSION, len(phase_2.survivors))

    campaign.phase = PHASE_HYBRIDIZATION
    session.commit()
    phase_3 = run_phase_3_hybridization(
        phase_1.survivors, phase_2.survivors, gene_pool, is_df, config.phase3_children, **phase_kwargs, **eval_kwargs,
    )
    if not phase_3.passed:
        campaign.status = "cancelled"
        session.commit()
        return phase_3
    _set_survivors_per_phase(session, campaign_id, PHASE_HYBRIDIZATION, len(phase_3.survivors))

    campaign.phase = PHASE_ROBUSTNESS
    session.commit()
    rejections: dict[str, int] = {}
    vault_candidates = run_phase_4_robustness(
        phase_3.survivors, df, gene_pool, max_drawdown_pct=config.max_drawdown_pct,
        max_daily_drawdown_pct=config.max_daily_drawdown_pct,
        rejections=rejections, **eval_kwargs,
    )
    _set_survivors_per_phase(session, campaign_id, PHASE_ROBUSTNESS, len(vault_candidates))

    if holdout_df is not None and not holdout_df.empty:
        backtest_kwargs = {k: v for k, v in eval_kwargs.items() if k not in ("timeframe", "initial_balance")}
        for candidate in vault_candidates:
            result = simulate_trades(
                holdout_df, candidate.genome, gene_pool, initial_balance=config.initial_balance, **backtest_kwargs
            )
            candidate.stats_holdout = compute_trade_stats(result, holdout_df, config.initial_balance, config.timeframe)
            candidate.holdout_start = pd.Timestamp(holdout_df["timestamp"].iloc[0]).isoformat()

    # The out-of-sample slice is the window Phase 4 judged these on, so it
    # is the one their trade timestamps belong to.
    _, oos_df = split_is_oos(df)
    write_vault_candidates(
        session, vault_candidates,
        asset=campaign.asset or "UNKNOWN",
        timeframe=campaign.timeframe or config.timeframe,
        data_source=config.data_source,
        oos_bars=oos_df,
        higher_timeframe=config.higher_timeframe if config.htf_df is not None else HIGHER_TF_OFF,
    )

    campaign.status = "completed"
    campaign.finished_at = datetime.now(timezone.utc)
    session.commit()

    # A campaign that approves nothing used to report "completed" over an
    # empty Vault and leave the user to guess. Phase 4 counts why it said
    # no, and the two commonest reasons are worth more than silence -
    # especially when one of them is a limit the user set in Settings and
    # can change.
    if not vault_candidates and rejections:
        ranked = sorted(rejections.items(), key=lambda item: item[1], reverse=True)
        detail = ", ".join(f"{count} for {reason}" for reason, count in ranked[:2])
        return PhaseOutcome(
            passed=True,
            survivors=[],
            fail_reason=f"No candidate cleared Phase 4: {detail}.",
            rejections=dict(ranked),
        )

    return PhaseOutcome(passed=True, survivors=[])
