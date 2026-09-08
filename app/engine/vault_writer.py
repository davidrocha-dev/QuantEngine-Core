"""Writes Phase 4-approved candidates into the Vault — REQ-ENGINE-3.11."""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from app.engine.phases import VaultCandidate
from app.engine.scoring import classify_archetype, compute_grade, compute_score
from app.models import Strategy


def vault_candidate_to_row(candidate: VaultCandidate, asset: str, timeframe: str, data_source: str = "dukascopy") -> Strategy:
    score = compute_score(candidate.stats_oos, candidate.robustness)
    return Strategy(
        id=candidate.hash_id,
        data_source=data_source,
        asset=asset,
        timeframe=timeframe,
        trades=candidate.stats_oos.trade_count,
        return_pct=candidate.stats_oos.net_profit_pct,
        cagr_pct=candidate.stats_oos.cagr_pct,
        max_dd_pct=candidate.stats_oos.max_dd_pct,
        archetype=classify_archetype(candidate.genome),
        robust_pct=100 - max(candidate.robustness.monte_carlo_degradation_pct, candidate.robustness.noise_degradation_pct),
        icir=candidate.stats_oos.icir,
        avg_hold_hours=candidate.stats_oos.avg_hold_hours,
        score=score,
        grade=compute_grade(score),
        genome_json=json.dumps(candidate.genome.to_dict()),
        equity_is=None,
        equity_oos=None,
        deployed=False,
        active_mt5=False,
    )


def write_vault_candidates(session: Session, candidates: list[VaultCandidate], asset: str, timeframe: str) -> list[Strategy]:
    rows = [vault_candidate_to_row(c, asset, timeframe) for c in candidates]
    session.add_all(rows)
    session.commit()
    return rows
