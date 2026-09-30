"""Campaign lifecycle: launch, resume, pause, skip, and the background run."""

import json
import os
import threading
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api import runtime
from app.api.campaign_setup import POWER_PRESETS, _gene_pool_for_campaign, _load_bars, build_campaign_config
from app.api.runtime import TRIBE_BY_PHASE, _best_fitness, _campaign_control, _emit_event_threadsafe
from app.config import get_config
from app.db import make_engine
from app.engine.campaign import (
    CampaignConfig,
    discard_campaign,
    run_campaign,
)
from app.engine.campaign_estimate import estimate_campaign_minutes
from app.engine.control import CampaignPaused
from app.engine.notifications import notify_event
from app.genome.gene_pool import GenePoolConfig
from app.models import Campaign
from app.settings_store import load_settings

router = APIRouter()

def _run_campaign_background(
    campaign_id: str, symbol: str, timeframe: str, gene_pool: GenePoolConfig, config: CampaignConfig, resume: bool = False
) -> None:

    def on_generation(phase: int, generation: int, population: list) -> None:
        fitnesses = [c.fitness for c in population if c.fitness is not None]
        _emit_event_threadsafe(
            {
                "type": "generation",
                "campaign_id": campaign_id,
                "status": "running",
                "phase": phase,
                "generation": generation,
                "tribe": TRIBE_BY_PHASE.get(phase),
                "fitness": max(fitnesses) if fitnesses else None,
                "survivors_per_phase": {},
            }
        )

    engine = make_engine(get_config().vault_db_path)
    try:
        settings = load_settings(get_config().settings_path)
        df = _load_bars(symbol, timeframe, settings["campaign"]["history_depth_years"])
        with Session(engine) as session:
            outcome = run_campaign(
                session, campaign_id, df, gene_pool, config,
                resume=resume, on_generation=on_generation, control=_campaign_control,
            )
        # REQ-SET-7.1/7.2/7.4 — a campaign runs for tens of minutes, so its
        # ending is announced on whichever channels are enabled. The OS
        # notification is the renderer's (Electron owns that API); Telegram
        # goes from here, so it arrives even with the window closed.
        summary = (
            f"QuantEngine: campaign on {symbol} {timeframe} "
            + (f"could not finish — {outcome.fail_reason}" if outcome.fail_reason else "completed.")
        )
        notify_event(settings, "campaign_complete", summary)

        if outcome.fail_reason:
            _emit_event_threadsafe(
                {"type": "campaign_complete", "campaign_id": campaign_id, "status": "cancelled",
                 "phase": None, "generation": None, "tribe": None, "fitness": None,
                 "survivors_per_phase": {}, "detail": outcome.fail_reason,
                 "rejections": outcome.rejections or {}}
            )
    except CampaignPaused:
        # Not a failure. The generation that was running finished and was
        # checkpointed; the campaign stops there and resumes from it.
        with Session(engine) as session:
            campaign = session.get(Campaign, campaign_id)
            if campaign is not None:
                campaign.status = "paused"
                session.commit()
        _emit_event_threadsafe(
            {"type": "campaign_paused", "campaign_id": campaign_id, "status": "paused",
             "phase": None, "generation": None, "tribe": None, "fitness": None,
             "survivors_per_phase": {}, "detail": "Paused after the current generation."}
        )
    except Exception:
        import traceback

        traceback.print_exc()
        with Session(engine) as session:
            campaign = session.get(Campaign, campaign_id)
            if campaign is not None:
                campaign.status = "crashed"
                session.commit()
    finally:
        runtime.active_campaign_id = None
        _campaign_control.clear()


def _serialize_campaign(campaign: Campaign) -> dict:
    return {
        "id": campaign.id,
        "status": campaign.status,
        "phase": campaign.phase,
        "tribe": TRIBE_BY_PHASE.get(campaign.phase),
        "asset": campaign.asset,
        "timeframe": campaign.timeframe,
        "started_at": campaign.started_at.isoformat() if campaign.started_at else None,
        "finished_at": campaign.finished_at.isoformat() if campaign.finished_at else None,
        "checkpoint_generation": campaign.checkpoint_generation,
        "survivors_per_phase": json.loads(campaign.survivors_per_phase) if campaign.survivors_per_phase else {},
        "best_fitness": _best_fitness(campaign.checkpoint_population),
    }


@router.get("/campaign/estimate")
def get_campaign_estimate() -> dict:
    """REQ-DASH-1.5 — duration estimate before launch, computed from the
    campaign this machine is actually about to run: POWER preset, cores,
    and how many bars of history the selected market has."""
    settings = load_settings(get_config().settings_path)
    campaign_settings = settings["campaign"]
    preset = POWER_PRESETS.get(campaign_settings["power"], POWER_PRESETS["MEDIUM"])
    cpu_cores = campaign_settings["cores"] or (os.cpu_count() or 1)

    symbol = next(iter(settings["target_markets"]["enabled_symbols"]), "EURUSD")
    timeframe = next(iter(settings["target_markets"]["enabled_timeframes"]), "H1")
    bars = len(_load_bars(symbol, timeframe, campaign_settings["history_depth_years"]))

    min_minutes, max_minutes = estimate_campaign_minutes(
        cpu_cores,
        population_size=preset["population_size"],
        generations=preset["generations_per_tribe"],
        bars=bars or 20_000,
    )
    return {
        "cpu_cores": cpu_cores,
        "min_minutes": min_minutes,
        "max_minutes": max_minutes,
        "power": campaign_settings["power"],
        "symbol": symbol,
        "timeframe": timeframe,
        "bars": bars,
    }


@router.post("/campaign/start")
def start_campaign() -> dict:
    # REQ-BREED-1.1/1.2 — LAUNCH. One campaign at a time: run_campaign holds
    # a DB session for its full 15-60 minute run, so a second concurrent run
    # would just contend for the same SQLite file rather than usefully queue.
    if runtime.active_campaign_id is not None:
        raise HTTPException(status_code=409, detail="A campaign is already running")

    settings = load_settings(get_config().settings_path)
    target_markets = settings["target_markets"]
    symbol = next(iter(target_markets["enabled_symbols"]), "EURUSD")
    timeframe = next(iter(target_markets["enabled_timeframes"]), "H1")
    try:
        config = build_campaign_config(settings, symbol, timeframe)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    gene_pool = _gene_pool_for_campaign(config)

    campaign_id = uuid.uuid4().hex
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        session.add(
            Campaign(
                id=campaign_id,
                started_at=datetime.now(timezone.utc),
                asset=symbol,
                timeframe=timeframe,
                phase=0,
                status="running",
            )
        )
        session.commit()

    runtime.active_campaign_id = campaign_id
    threading.Thread(
        target=_run_campaign_background,
        args=(campaign_id, symbol, timeframe, gene_pool, config),
        daemon=True,
    ).start()
    return {"campaign_id": campaign_id, "status": "running"}


@router.get("/campaign/crashed")
def list_crashed_campaigns() -> list:
    # REQ-ENGINE-4.2 — the resume/discard prompt's data source: campaigns
    # the previous engine process left behind when it died mid-run.
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        rows = session.scalars(
            select(Campaign).where(Campaign.status == "crashed").order_by(Campaign.started_at.desc())
        ).all()
        return [_serialize_campaign(row) for row in rows if row.checkpoint_population]


@router.post("/campaign/{campaign_id}/resume")
def resume_campaign(campaign_id: str) -> dict:
    # REQ-ENGINE-4.3 — continue evolution from the last checkpoint.
    if runtime.active_campaign_id is not None:
        raise HTTPException(status_code=409, detail="A campaign is already running")

    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        campaign = session.get(Campaign, campaign_id)
        if campaign is None:
            raise HTTPException(status_code=404, detail=f"Unknown campaign: {campaign_id}")
        if not campaign.checkpoint_population:
            raise HTTPException(status_code=409, detail="This campaign has no checkpoint to resume from")
        symbol = campaign.asset or "EURUSD"
        timeframe = campaign.timeframe or "H1"

    settings = load_settings(get_config().settings_path)
    try:
        config = build_campaign_config(settings, symbol, timeframe)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    runtime.active_campaign_id = campaign_id
    threading.Thread(
        target=_run_campaign_background,
        args=(campaign_id, symbol, timeframe, GenePoolConfig.load(get_config().gene_pool_path), config, True),
        daemon=True,
    ).start()
    return {"campaign_id": campaign_id, "status": "running", "resumed": True}


@router.post("/campaign/{campaign_id}/discard")
def discard_crashed_campaign(campaign_id: str) -> dict:
    # REQ-ENGINE-4.4 — mark cancelled and drop it from the queue.
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        try:
            discard_campaign(session, campaign_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"campaign_id": campaign_id, "status": "cancelled"}


@router.post("/campaign/pause")
def pause_campaign() -> dict:
    """REQ-BREED-1.2 — stop after the current generation, keep the work.

    Nothing is interrupted mid-generation: that would throw away the work
    in flight and leave the checkpoint describing a population that no
    longer exists. The flag is read at the next generation boundary, so a
    pause costs at most one generation of waiting and resumes from a
    checkpoint that was always going to be written.
    """
    if runtime.active_campaign_id is None:
        raise HTTPException(status_code=409, detail="No campaign is running.")
    _campaign_control.request_pause()
    return {
        "status": "pausing",
        "campaign_id": runtime.active_campaign_id,
        "detail": "Pausing after the current generation finishes. Resume picks up from there.",
    }


@router.post("/campaign/skip")
def skip_campaign() -> dict:
    """REQ-BREED-1.2 — end the current phase early and move on.

    The phase keeps whatever it has evolved so far and hands it to the
    next one, which is what someone means when a tribe has clearly
    stopped improving. It applies to the current phase only; skipping
    every remaining phase with one click is not what anyone wants.
    """
    if runtime.active_campaign_id is None:
        raise HTTPException(status_code=409, detail="No campaign is running.")
    _campaign_control.request_skip()
    return {
        "status": "skipping",
        "campaign_id": runtime.active_campaign_id,
        "detail": "Ending this phase after the current generation; its survivors carry forward.",
    }
