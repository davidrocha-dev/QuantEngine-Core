"""Writes Phase 4-approved candidates into the Vault — REQ-ENGINE-3.11."""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

import pandas as pd

from app.engine.hash_id import assign_hash_id
from app.engine.monte_carlo import robustness_band
from app.engine.trade_log import replace_trades
from app.engine.phases import VaultCandidate
from app.engine.scoring import classify_archetype, compute_grade, compute_score
from app.models import Strategy


def _band(candidate: VaultCandidate):
    return robustness_band(candidate.robustness)


def vault_candidate_to_row(candidate: VaultCandidate, asset: str, timeframe: str, data_source: str = "dukascopy") -> Strategy:
    """`data_source` records which data the strategy was bred on — the
    Vault's BUILT-IN DATA BRED purge criterion (REQ-SET-4.1) depends on it."""
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
        max_daily_dd_pct=candidate.stats_oos.max_daily_dd_pct,
        archetype=classify_archetype(candidate.genome),
        robust_pct=_band(candidate)[0],
        robust_low_pct=_band(candidate)[1],
        robust_high_pct=_band(candidate)[2],
        icir=candidate.stats_oos.icir,
        win_rate_pct=candidate.stats_oos.win_rate * 100,
        avg_hold_hours=candidate.stats_oos.avg_hold_hours,
        score=score,
        grade=compute_grade(score, candidate.stats_oos.trade_count, candidate.stats_oos.net_profit_pct),
        genome_json=json.dumps(candidate.genome.to_dict()),
        equity_is=json.dumps(candidate.equity_is) if candidate.equity_is else None,
        equity_oos=json.dumps(candidate.equity_oos) if candidate.equity_oos else None,
        persistence_pct=candidate.walk_forward.persistence_pct if candidate.walk_forward else None,
        walk_forward=json.dumps(candidate.walk_forward.to_dict()) if candidate.walk_forward else None,
        deployed=False,
        active_mt5=False,
    )


def _unique_hash_id(session: Session, candidate: VaultCandidate, taken: set[str]) -> str:
    # 6 hex chars is only ~16.7M IDs — by the birthday bound a Vault of a
    # few thousand strategies is likely to see a collision, and a duplicate
    # primary key would fail the whole campaign's Vault write.
    hash_id = candidate.hash_id
    while hash_id in taken or session.get(Strategy, hash_id) is not None:
        hash_id = assign_hash_id(candidate.genome)
    taken.add(hash_id)
    return hash_id


def write_vault_candidates(
    session: Session,
    candidates: list[VaultCandidate],
    asset: str,
    timeframe: str,
    data_source: str = "dukascopy",
    oos_bars: pd.DataFrame | None = None,
) -> list[Strategy]:
    taken: set[str] = set()
    for candidate in candidates:
        candidate.hash_id = _unique_hash_id(session, candidate, taken)
    rows = [vault_candidate_to_row(c, asset, timeframe, data_source) for c in candidates]
    session.add_all(rows)
    session.commit()

    # The trades behind each row, so the strategy can be interrogated
    # later without re-simulating it. `oos_bars` is optional because a
    # caller that does not have them should still be able to write the
    # Vault rows themselves.
    if oos_bars is not None:
        for candidate, row in zip(candidates, rows):
            if candidate.trades:
                replace_trades(session, row.id, candidate.trades, oos_bars)
        session.commit()

    return rows
