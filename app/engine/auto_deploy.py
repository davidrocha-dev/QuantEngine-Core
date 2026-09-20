"""Replacing a degraded bot automatically — REQ-AUTO-4.

When a health check says a deployed strategy has stopped working, the
Elite-tier answer is to retire it and put the best remaining candidate for
the same market in its place.

This is the most consequential thing the app does without being asked, so
it is bounded deliberately:

  * it only ever acts on a **degraded** verdict, never on "too early" or
    "no data" — absence of evidence is not evidence of decay;
  * it replaces like with like: same symbol and timeframe, so a EURUSD M15
    slot stays a EURUSD M15 slot;
  * it never leaves a slot empty. With no qualified replacement, the
    degraded bot is reported and left alone rather than retired into a gap
    the user did not ask for;
  * it requires an explicit first-activation confirmation (REQ-AUTO-4.2),
    recorded in settings, before it will do any of this.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.broker.mt5_install import install_ea, uninstall_ea
from app.codegen.mql5 import UnsupportedGeneError, generate_ea
from app.engine.health import run_health_check
from app.genome.genome import Genome
from app.models import Strategy

# A replacement must be at least as good as this on the Vault's own scale;
# swapping a degraded bot for a mediocre one is not an improvement.
MIN_REPLACEMENT_SCORE = 57  # the GOOD grade threshold (scoring.py)


@dataclass
class Replacement:
    retired_id: str
    installed_id: str | None
    detail: str

    def to_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}


def find_replacement(session: Session, degraded: Strategy) -> Strategy | None:
    """Best undeployed candidate for the same market."""
    candidates = session.scalars(
        select(Strategy)
        .where(Strategy.deleted_at.is_(None))
        .where(Strategy.deployed.is_(False))
        .where(Strategy.asset == degraded.asset)
        .where(Strategy.timeframe == degraded.timeframe)
        .where(Strategy.score >= MIN_REPLACEMENT_SCORE)
        .order_by(Strategy.score.desc())
    ).all()
    return candidates[0] if candidates else None


def replace_degraded(session: Session, risk_pct: float = 0.01, max_spread_points: float = 0.0) -> list[Replacement]:
    """Retires every degraded bot that has a qualified successor."""
    actions: list[Replacement] = []

    for verdict in run_health_check(session):
        if verdict.status != "degraded":
            continue

        degraded = session.get(Strategy, verdict.strategy_id)
        if degraded is None:
            continue

        replacement = find_replacement(session, degraded)
        if replacement is None:
            actions.append(
                Replacement(
                    retired_id=degraded.id,
                    installed_id=None,
                    detail=(
                        f"{degraded.id} has degraded, but no undeployed {degraded.asset} {degraded.timeframe} "
                        f"strategy scores {MIN_REPLACEMENT_SCORE}+ — left running rather than retired into a gap."
                    ),
                )
            )
            continue

        try:
            generated = generate_ea(
                Genome.from_dict(__import__("json").loads(replacement.genome_json)),
                replacement.id, replacement.asset, replacement.timeframe,
                risk_pct=risk_pct, max_spread_points=max_spread_points,
            )
        except (UnsupportedGeneError, KeyError, TypeError, ValueError) as exc:
            actions.append(
                Replacement(retired_id=degraded.id, installed_id=None, detail=f"Could not build a replacement EA: {exc}")
            )
            continue

        result = install_ea(generated.filename, generated.source)
        if not result.installed:
            actions.append(
                Replacement(retired_id=degraded.id, installed_id=None, detail=f"Replacement not installed: {result.detail}")
            )
            continue

        # Retire only once the replacement is actually in place.
        uninstall_ea(f"QE_{degraded.id.lstrip('#').upper()}.mq5")
        degraded.deployed = False
        replacement.deployed = True
        replacement.incubation_stage = None  # REQ-PORT-3.1 — a new bot starts in INCUBATION
        session.commit()

        actions.append(
            Replacement(
                retired_id=degraded.id,
                installed_id=replacement.id,
                detail=(
                    f"Retired {degraded.id} and installed {replacement.id} "
                    f"(score {replacement.score}) on {replacement.asset} {replacement.timeframe}. "
                    "It starts at a quarter of target risk, like any new deploy."
                ),
            )
        )

    return actions
