"""Live trading: the Pilot EA, health checks, incubation and auto-deploy."""


from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.campaign_setup import _strategy_higher_timeframe
from app.broker import pilot
from app.config import get_config
from app.data.broker_clock import clock_from_settings
from app.db import make_engine
from app.engine.auto_deploy import replace_degraded
from app.engine.health import auto_promote, check_all_brokers, ladder, promote, risk_shares, run_health_check
from app.models import LiveTrade
from app.settings_store import load_settings, update_section

router = APIRouter()

@router.get("/pilot/status")
def get_pilot_status() -> dict:
    """REQ-PORT-4.2 — whether the Pilot is reporting. The Dashboard banner
    and the Incubation tab both need to know, because without it no live
    result reaches the app and no bot can be promoted (REQ-PORT-4.3)."""
    path = pilot.report_path()
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        latest = session.scalar(select(LiveTrade).order_by(LiveTrade.closed_at.desc()))
        total = len(list(session.scalars(select(LiveTrade.deal_ticket))))
    return {
        "report_found": path is not None,
        "report_path": str(path) if path else None,
        "live_trades": total,
        "last_trade_at": latest.closed_at.isoformat() if latest else None,
        "detail": (
            "Pilot is reporting."
            if path
            else "No Pilot report found. Install the Pilot and attach it to any chart in MT5."
        ),
    }


@router.post("/pilot/install")
def post_pilot_install() -> dict:
    # REQ-PORT-4.1 — install the Pilot helper EA into every terminal.
    result = pilot.install_pilot()
    if not result.installed:
        raise HTTPException(status_code=409, detail=result.detail)
    return {
        "installed_paths": result.installed_paths,
        "compiled_paths": result.compiled_paths,
        "detail": result.detail + " Attach it to any one chart per terminal.",
    }


@router.post("/pilot/import")
def post_pilot_import() -> dict:
    """Reads whatever the Pilot has written since last time. Idempotent:
    rows are keyed by the broker's own deal ticket."""
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        return pilot.import_live_trades(session).to_dict()


@router.post("/health-check")
def post_health_check(strategy_ids: list[str] | None = None) -> dict:
    """REQ-AUTO-2/5 — compare deployed strategies against the backtests
    that approved them. Imports the Pilot's latest report first, so the
    verdict is never based on stale live data."""
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        imported = pilot.import_live_trades(session)
        verdicts = run_health_check(session, strategy_ids)
        broker_verdicts = [v for v in check_all_brokers(session) if v.classification != "inconclusive"]

    counts = {status: sum(1 for v in verdicts if v.status == status) for status in ("healthy", "degraded", "too_early", "no_data")}
    return {
        "imported": imported.to_dict(),
        "verdicts": [verdict.to_dict() for verdict in verdicts],
        # REQ-AUTO-5's "N holding / N decayed / N too early".
        "summary": counts,
        # REQ-AUTO-3 — only says anything when a strategy runs on more than
        # one broker; otherwise the two causes cannot be told apart.
        "brokers": [verdict.to_dict() for verdict in broker_verdicts],
    }


@router.get("/incubation")
def get_incubation() -> dict:
    """REQ-PORT-3 — the ladder: which stage each deployed bot is at, and
    what is holding back the ones that cannot advance."""
    settings = load_settings(get_config().settings_path)
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        pilot.import_live_trades(session)
        if settings["automation"]["auto_graduate_enabled"]:
            auto_promote(session)  # REQ-PORT-3.6 — AUTO on
        entries = ladder(session)
        # REQ-PORT-3.7 — publish the sizes so running EAs pick them up for
        # their next trade; positions already open are untouched.
        pilot.write_risk_shares(risk_shares(session))
    return {
        "pilot_reporting": pilot.report_path() is not None,
        "auto_graduate": settings["automation"]["auto_graduate_enabled"],
        "entries": [entry.to_dict() for entry in entries],
    }


@router.post("/incubation/{strategy_id}/graduate")
def post_graduate(strategy_id: str) -> dict:
    """REQ-PORT-3.6 — the manual GRADUATE action. Same blocking rules as
    AUTO: a promotion is allowed by live evidence, never forced by it."""
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        pilot.import_live_trades(session)
        promoted, detail, entry = promote(session, strategy_id)
        if not promoted:
            raise HTTPException(status_code=409, detail=detail)
        pilot.write_risk_shares(risk_shares(session))
    return {"promoted": True, "detail": detail, "entry": entry.to_dict() if entry else None}


@router.post("/automation/auto-deploy/confirm")
def post_confirm_auto_deploy() -> dict:
    """REQ-AUTO-4.2 — the explicit first-activation confirmation.

    Kept separate from the toggle on purpose: turning a switch on is not
    the same as understanding that the app may retire a running bot and
    install another one without asking again.
    """
    updated = update_section(get_config().settings_path, "automation", {"auto_deploy_confirmed": True})
    return {"confirmed": updated["auto_deploy_confirmed"]}


@router.post("/automation/auto-deploy/run")
def post_run_auto_deploy() -> dict:
    """Runs the replacement pass now, for a user who would rather trigger
    it than wait for the scheduler."""
    settings = load_settings(get_config().settings_path)
    if not settings["automation"]["auto_deploy_confirmed"]:
        raise HTTPException(status_code=409, detail="Auto-deploy has not been confirmed yet.")

    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        pilot.import_live_trades(session)
        actions = replace_degraded(
            session,
            risk_pct=settings["risk"]["per_trade_risk_pct"] / 100,
            max_spread_points=0.0,
            higher_timeframe_for=lambda row: _strategy_higher_timeframe(settings, row),
            broker_clock=clock_from_settings(settings),
        )
    return {"actions": [action.to_dict() for action in actions]}
