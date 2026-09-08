"""Campaign orchestration: runs the 5-phase pipeline against a `campaigns`
DB row, checkpointing every completed generation (REQ-ENGINE-4.1) and
supporting crash-resume (REQ-ENGINE-4.2-4.4).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.engine.metrics import TradeStats
from app.engine.phases import (
    CandidateResult,
    PhaseOutcome,
    run_phase_0_screening,
    run_phase_1_stability,
    run_phase_2_expansion,
    run_phase_3_hybridization,
    run_phase_4_robustness,
    split_is_oos,
)
from app.engine.vault_writer import write_vault_candidates
from app.genome.gene_pool import GenePoolConfig
from app.genome.genome import Genome, random_population
from app.models import Campaign

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
    timeframe: str = "H1"
    point_size: float = 0.0001


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

    is_df, _ = split_is_oos(df)
    eval_kwargs = {"initial_balance": config.initial_balance, "risk_pct": config.risk_pct, "timeframe": config.timeframe}

    loaded = load_checkpoint(campaign) if resume else None

    if loaded is not None:
        phase_0_survivors = loaded.phase_0_survivors
    else:
        population = random_population(gene_pool, config.population_size)
        phase_0 = run_phase_0_screening(population, is_df, gene_pool, **eval_kwargs)
        _set_survivors_per_phase(session, campaign_id, PHASE_SCREENING, len(phase_0.survivors))
        if not phase_0.survivors:
            campaign.status = "cancelled"
            session.commit()
            return PhaseOutcome(passed=False, fail_reason="Not Passed: no candidates survived Phase 0 screening")
        phase_0_survivors = phase_0.survivors

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
        on_generation_complete=make_checkpoint_cb(PHASE_STABILITY), start_generation=start_gen, **eval_kwargs,
    )
    _set_survivors_per_phase(session, campaign_id, PHASE_STABILITY, len(phase_1.survivors))

    if loaded is not None and campaign.phase == PHASE_EXPANSION:
        tribe_b_seed, start_gen = loaded.active_population, loaded.generation
    else:
        tribe_b_seed, start_gen = phase_0_survivors, 0
    phase_2 = run_phase_2_expansion(
        tribe_b_seed, gene_pool, is_df, config.generations_per_tribe, config.phase2_top_n,
        on_generation_complete=make_checkpoint_cb(PHASE_EXPANSION), start_generation=start_gen, **eval_kwargs,
    )
    _set_survivors_per_phase(session, campaign_id, PHASE_EXPANSION, len(phase_2.survivors))

    campaign.phase = PHASE_HYBRIDIZATION
    session.commit()
    phase_3 = run_phase_3_hybridization(
        phase_1.survivors, phase_2.survivors, gene_pool, is_df, config.phase3_children, **eval_kwargs,
    )
    if not phase_3.passed:
        campaign.status = "cancelled"
        session.commit()
        return phase_3
    _set_survivors_per_phase(session, campaign_id, PHASE_HYBRIDIZATION, len(phase_3.survivors))

    campaign.phase = PHASE_ROBUSTNESS
    session.commit()
    vault_candidates = run_phase_4_robustness(
        phase_3.survivors, df, gene_pool, config.point_size, **eval_kwargs,
    )
    _set_survivors_per_phase(session, campaign_id, PHASE_ROBUSTNESS, len(vault_candidates))

    write_vault_candidates(session, vault_candidates, asset=campaign.asset or "UNKNOWN", timeframe=campaign.timeframe or config.timeframe)

    campaign.status = "completed"
    campaign.finished_at = datetime.now(timezone.utc)
    session.commit()

    return PhaseOutcome(passed=True, survivors=[])
