"""Background loops started with the engine: startup delta sync and the automation scheduler."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.api import runtime
from app.api.campaign_setup import _strategy_higher_timeframe
from app.api.routes.campaign import start_campaign
from app.api.runtime import _sync_state
from app.broker import pilot
from app.config import get_config
from app.data.broker_clock import clock_from_settings
from app.data.symbols import BUNDLED_SYMBOLS
from app.data.sync import DeltaSyncRunner
from app.db import make_engine
from app.engine.auto_deploy import replace_degraded
from app.engine.notifications import notify_event
from app.engine.purge import (
    empty_expired_recycle_bin,
    run_purge,
)
from app.settings_store import load_settings, update_section


async def _startup_delta_sync() -> None:
    config = get_config()
    if config.built_in_data_only:
        return  # REQ-DATA-1.6 — the debug toggle pins the app to bundled data.
    settings = load_settings(config.settings_path)
    symbols = settings["target_markets"]["enabled_symbols"] or list(BUNDLED_SYMBOLS)
    runner = DeltaSyncRunner(history_years=settings["campaign"]["history_depth_years"], state=_sync_state)
    runtime.sync_task = asyncio.current_task()
    try:
        await runner.sync_all(symbols)
    except asyncio.CancelledError:
        raise
    except Exception:
        import traceback

        traceback.print_exc()

SCHEDULER_TICK_SECONDS = 60
# How often auto-deploy re-checks. Hourly is deliberate: a degraded
# verdict comes from live trades, which arrive over hours, not seconds.
AUTO_DEPLOY_INTERVAL_HOURS = 6


def _due(last_at: str | None, interval: timedelta, now: datetime) -> bool:
    if last_at is None:
        return True
    try:
        previous = datetime.fromisoformat(last_at)
    except ValueError:
        return True
    if previous.tzinfo is None:
        previous = previous.replace(tzinfo=timezone.utc)
    return now - previous >= interval


async def _scheduler_loop() -> None:
    """Recurring campaigns and automatic purge — tasks.md 12.2, REQ-SET-4.5.

    Both are deliberately conservative: a campaign only starts when none is
    running, and a purge is skipped entirely while one is (REQ-SET-4.4).
    Timestamps live in settings, so a restart never re-fires something that
    already ran.
    """
    while True:
        await asyncio.sleep(SCHEDULER_TICK_SECONDS)
        try:
            config = get_config()
            settings = load_settings(config.settings_path)
            automation = settings["automation"]
            now = datetime.now(timezone.utc)

            if (
                automation["recurring_campaigns_enabled"]
                and runtime.active_campaign_id is None
                and _due(automation["last_campaign_started_at"], timedelta(hours=automation["recurring_interval_hours"]), now)
            ):
                try:
                    start_campaign()
                    update_section(config.settings_path, "automation", {"last_campaign_started_at": now.isoformat()})
                except HTTPException:
                    pass  # A campaign is already running; try again next tick.

            # REQ-AUTO-4 — only with both the toggle and the explicit
            # first-activation confirmation, and never mid-campaign.
            if (
                automation["auto_deploy_enabled"]
                and automation["auto_deploy_confirmed"]
                and runtime.active_campaign_id is None
                and _due(automation["last_auto_deploy_at"], timedelta(hours=AUTO_DEPLOY_INTERVAL_HOURS), now)
            ):
                engine = make_engine(config.vault_db_path)
                with Session(engine) as session:
                    pilot.import_live_trades(session)
                    actions = replace_degraded(
                        session,
                        risk_pct=settings["risk"]["per_trade_risk_pct"] / 100,
                        max_spread_points=0.0,
                        higher_timeframe_for=lambda row: _strategy_higher_timeframe(settings, row),
                        broker_clock=clock_from_settings(settings),
                    )
                update_section(config.settings_path, "automation", {"last_auto_deploy_at": now.isoformat()})
                for action in actions:
                    logging.getLogger("quantengine").info("Auto-deploy: %s", action.detail)
                    notify_event(settings, "bot_degradation", f"QuantEngine: {action.detail}")

            maintenance = settings["vault_maintenance"]
            if maintenance["automatic_purge_enabled"] and _due(
                automation["last_purge_at"], timedelta(days=maintenance["automatic_purge_interval_days"]), now
            ):
                engine = make_engine(config.vault_db_path)
                with Session(engine) as session:
                    empty_expired_recycle_bin(session, now)
                    result = run_purge(session, settings, now)
                if result.skipped_reason is None:
                    update_section(config.settings_path, "automation", {"last_purge_at": now.isoformat()})
        except asyncio.CancelledError:
            raise
        except Exception:
            # A scheduling hiccup must never take the engine down.
            import traceback

            traceback.print_exc()
