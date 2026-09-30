"""Process-wide state of the engine API and the live event fan-out to the renderer."""

import asyncio
import json

from fastapi import WebSocket
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.broker.mt5_bridge import MT5Bridge
from app.config import get_config
from app.data.sync import SyncState
from app.db import make_engine
from app.engine.control import CampaignControl
from app.models import Campaign

# REQ-BREED-1 — campaign lifecycle state and the WS fan-out that lets the
# BREED screen watch it live. `run_campaign` (app/engine/campaign.py) runs
# synchronously for 15-60 minutes, so it's launched on a background thread;
# a lightweight poller (not a hook into the GA loop itself) diffs each
# campaign row every second and broadcasts what changed. This keeps the
# well-tested phase/GA code completely untouched.
active_campaign_id: str | None = None
_ws_clients: set[WebSocket] = set()
_campaign_poll_state: dict[str, tuple] = {}

# Phase -> the tribe whose population is actively evolving in that phase,
# per design.md's Tribe A/B/C color roles (campaign.py's PHASE_* constants).
TRIBE_BY_PHASE = {0: None, 1: "A", 2: "B", 3: "C", 4: "C"}


def _best_fitness(checkpoint_population_json: str | None) -> float | None:
    if not checkpoint_population_json:
        return None
    try:
        data = json.loads(checkpoint_population_json)
        fitnesses = [c["fitness"] for c in data.get("active_population", []) if c.get("fitness") is not None]
        return max(fitnesses) if fitnesses else None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


main_loop: asyncio.AbstractEventLoop | None = None


def _emit_event_threadsafe(event: dict) -> None:
    # `_run_campaign_background` runs on a plain thread (not the asyncio
    # loop), so `_broadcast` — a coroutine — can't just be awaited directly
    # from there; hand it to the loop captured at startup instead.
    if main_loop is not None:
        asyncio.run_coroutine_threadsafe(_broadcast(event), main_loop)


async def _broadcast(event: dict) -> None:
    dead = []
    for client in list(_ws_clients):
        try:
            await client.send_json(event)
        except Exception:
            dead.append(client)
    for client in dead:
        _ws_clients.discard(client)


_GENERATIONAL_PHASES = {1, 2}  # PHASE_STABILITY, PHASE_EXPANSION — see app/engine/campaign.py


async def _poll_campaigns_loop() -> None:
    engine = make_engine(get_config().vault_db_path)
    while True:
        await asyncio.sleep(1.0)
        try:
            with Session(engine) as session:
                for campaign in session.scalars(select(Campaign)):
                    state = (campaign.status, campaign.phase, campaign.checkpoint_generation)
                    if _campaign_poll_state.get(campaign.id) == state:
                        continue
                    _campaign_poll_state[campaign.id] = state

                    if campaign.status != "running":
                        event_type = "campaign_complete"
                    elif campaign.phase in _GENERATIONAL_PHASES:
                        event_type = "generation"
                    else:
                        # Phase 0 (Screening) and 3/4 (Hybridization/Robustness)
                        # are single-pass, not generational — they never call
                        # _checkpoint, so campaign.checkpoint_population here
                        # is still whichever generational phase ran last.
                        # Tagging that stale population as this phase's tribe
                        # is exactly the bug that made Tribe C briefly show
                        # Tribe B's numbers verbatim — so for these phases we
                        # report progress but omit tribe/fitness entirely.
                        event_type = "phase_progress"

                    await _broadcast(
                        {
                            "type": event_type,
                            "campaign_id": campaign.id,
                            "status": campaign.status,
                            "phase": campaign.phase,
                            "generation": campaign.checkpoint_generation,
                            "tribe": TRIBE_BY_PHASE.get(campaign.phase) if event_type == "generation" else None,
                            "fitness": _best_fitness(campaign.checkpoint_population) if event_type == "generation" else None,
                            "survivors_per_phase": json.loads(campaign.survivors_per_phase)
                            if campaign.survivors_per_phase
                            else {},
                        }
                    )
        except Exception:
            # A broadcast hiccup must never take the poller down — the BREED
            # screen just misses an update and catches up on the next tick.
            pass


# PAUSE and SKIP set flags here; the phase loop reads them at a
# generation boundary. One object for the process, because only one
# campaign runs at a time (see the guard in launch_campaign).
_campaign_control = CampaignControl()

_sync_state = SyncState()
# The running sync, so it can be stopped. `create_task` used to be called
# and its result dropped, which left no way to reach a download that runs
# for hours - and the startup sync relaunches on every open, so closing
# the app did not stop it either, it restarted it.
sync_task: asyncio.Task | None = None

# Live price reference/execution only — never a backtest history source.
# Single process-wide instance since MetaTrader5's Python API is itself a
# singleton (mt5.initialize() attaches to one terminal per process).
_mt5_bridge = MT5Bridge()
