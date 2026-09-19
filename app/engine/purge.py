"""Vault maintenance — REQ-SET-4.

Purge evaluates the criteria configured in Settings → Vault Maintenance
against every Vault strategy. Nothing is deleted outright: a purged
strategy is soft-deleted into the Recycle Bin (REQ-SET-4.3), recoverable
for `RECYCLE_BIN_RETENTION_DAYS`, after which it is dropped for good.

Two safety rules are part of the feature, not optional extras:
  * PROTECT PORTFOLIO (REQ-SET-4.2) — a strategy that is a member of any
    portfolio is never purged while the setting is on.
  * A purge never runs while a campaign is active (REQ-SET-4.4), since the
    campaign is writing into the same table.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Campaign, PortfolioStrategy, Strategy

RECYCLE_BIN_RETENTION_DAYS = 30


@dataclass
class PurgeResult:
    purged: int
    protected: int
    reasons: dict[str, int]
    skipped_reason: str | None = None


def _criterion(criteria: dict, key: str) -> float | None:
    """Returns the threshold for an enabled criterion, else None."""
    entry = criteria.get(key) or {}
    if not entry.get("enabled"):
        return None
    threshold = entry.get("threshold")
    return float(threshold) if threshold is not None else None


def evaluate_purge_reason(row: Strategy, criteria: dict, now: datetime) -> str | None:
    """The first criterion this strategy fails, or None when it survives."""
    if (criteria.get("dd_gt_return") or {}).get("enabled"):
        if row.max_dd_pct is not None and row.return_pct is not None and row.max_dd_pct > row.return_pct:
            return "drawdown exceeds return"

    if (criteria.get("built_in_data_bred") or {}).get("enabled") and row.data_source == "bundled":
        return "bred on built-in data"

    checks: list[tuple[str, float | None, float | None, str]] = [
        ("min_trades", _criterion(criteria, "min_trades"), _as_float(row.trades), "too few trades"),
        ("min_ret_pct", _criterion(criteria, "min_ret_pct"), row.return_pct, "return below minimum"),
        ("min_score", _criterion(criteria, "min_score"), _as_float(row.score), "score below minimum"),
        ("min_robustness_pct", _criterion(criteria, "min_robustness_pct"), row.robust_pct, "robustness below minimum"),
        ("min_icir", _criterion(criteria, "min_icir"), row.icir, "ICIR below minimum"),
        ("min_avg_hold_hours", _criterion(criteria, "min_avg_hold_hours"), row.avg_hold_hours, "average hold below minimum"),
    ]
    for _, threshold, value, reason in checks:
        if threshold is not None and value is not None and value < threshold:
            return reason

    max_age_days = _criterion(criteria, "max_age_days")
    if max_age_days is not None and row.created_at is not None:
        created = row.created_at if row.created_at.tzinfo else row.created_at.replace(tzinfo=timezone.utc)
        if (now - created).days > max_age_days:
            return "older than the age limit"

    return None


def _as_float(value) -> float | None:
    return None if value is None else float(value)


def run_purge(session: Session, settings: dict, now: datetime | None = None) -> PurgeResult:
    now = now or datetime.now(timezone.utc)
    maintenance = settings["vault_maintenance"]
    criteria = maintenance["purge_criteria"]

    # REQ-SET-4.4 — a running campaign is actively writing to this table.
    if session.scalar(select(Campaign).where(Campaign.status == "running")) is not None:
        return PurgeResult(purged=0, protected=0, reasons={}, skipped_reason="a campaign is running")

    protected_ids: set[str] = set()
    if maintenance["protect_portfolio"]:  # REQ-SET-4.2
        protected_ids = set(session.scalars(select(PortfolioStrategy.strategy_id)))

    purged, protected = 0, 0
    reasons: dict[str, int] = {}
    for row in session.scalars(select(Strategy).where(Strategy.deleted_at.is_(None))):
        reason = evaluate_purge_reason(row, criteria, now)
        if reason is None:
            continue
        if row.id in protected_ids:
            protected += 1
            continue
        row.deleted_at, row.delete_reason = now, reason
        reasons[reason] = reasons.get(reason, 0) + 1
        purged += 1

    session.commit()
    return PurgeResult(purged=purged, protected=protected, reasons=reasons)


def restore_from_recycle_bin(session: Session, strategy_id: str) -> bool:
    row = session.get(Strategy, strategy_id)
    if row is None or row.deleted_at is None:
        return False
    row.deleted_at, row.delete_reason = None, None
    session.commit()
    return True


def empty_expired_recycle_bin(session: Session, now: datetime | None = None) -> int:
    """Permanently drops strategies past the 30-day recovery window."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=RECYCLE_BIN_RETENTION_DAYS)
    removed = 0
    for row in session.scalars(select(Strategy).where(Strategy.deleted_at.is_not(None))):
        deleted_at = row.deleted_at if row.deleted_at.tzinfo else row.deleted_at.replace(tzinfo=timezone.utc)
        if deleted_at < cutoff:
            session.delete(row)
            removed += 1
    session.commit()
    return removed
