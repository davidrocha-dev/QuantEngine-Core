import asyncio
import json
import os

import pandas as pd
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.broker.mt5_bridge import MT5Bridge
from app.broker.mt5_install import install_ea
from app.codegen.mql5 import GeneratedEa, UnsupportedGeneError, generate_ea
from app.config import get_config, set_auto_backup_before_updates, set_built_in_data_only
from app.data.provider import get_historical_bars
from app.data.sync import DeltaSyncRunner, SyncState
from app.data.symbols import BUNDLED_SYMBOLS, INSTRUMENTS, TIMEFRAMES, get_instrument
from app.db import make_engine
from app.engine.campaign import (
    MIN_HISTORY_DAYS,
    MIN_IS_BARS,
    CampaignConfig,
    discard_campaign,
    mark_crashed_campaigns,
    run_campaign,
)
from app.engine.news_filter import load_calendar
from app.engine.notifications import notify_event, send_telegram
from app.engine.purge import (
    RECYCLE_BIN_RETENTION_DAYS,
    empty_expired_recycle_bin,
    restore_from_recycle_bin,
    run_purge,
)
from app.engine.campaign_estimate import estimate_campaign_minutes
from app.engine.neural_map import MIN_STRATEGIES_FOR_MAP, compute_neural_map
from app.engine.phases import IS_FRACTION
from app.engine.portfolio import compute_health
from app.engine import smart_mode
from app.engine.reassess import reassess_strategy as reassess_strategy_engine
from app.engine.true_sim import run_true_sim
from app.engine.reassess import refresh_strategy as refresh_strategy_engine
from app.genome.gene_pool import GenePoolConfig
from app.genome.genes import ALL_GENES
from app.migrations.runner import MigrationError, export_vault, restore_database, run_migrations
from app.genome.genome import Genome
from app.models import Campaign, Portfolio, PortfolioStrategy, Strategy
from app.settings_store import load_settings, update_section


# REQ-BREED-1 — campaign lifecycle state and the WS fan-out that lets the
# BREED screen watch it live. `run_campaign` (app/engine/campaign.py) runs
# synchronously for 15-60 minutes, so it's launched on a background thread;
# a lightweight poller (not a hook into the GA loop itself) diffs each
# campaign row every second and broadcasts what changed. This keeps the
# well-tested phase/GA code completely untouched.
_active_campaign_id: str | None = None
_ws_clients: set[WebSocket] = set()
_campaign_poll_state: dict[str, tuple] = {}

# POWER (Settings → Campaign) controls compute intensity — REQ-SET-3/design.md
# A.9. The full 4,000/20-generation run is EXTREME; lighter presets keep
# ECO/MEDIUM campaigns fast enough to actually watch complete.
POWER_PRESETS = {
    "ECO": {"population_size": 60, "generations_per_tribe": 3},
    "MEDIUM": {"population_size": 300, "generations_per_tribe": 5},
    "HIGH": {"population_size": 1000, "generations_per_tribe": 10},
    "EXTREME": {"population_size": 4000, "generations_per_tribe": 20},
}

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


_main_loop: asyncio.AbstractEventLoop | None = None


def _emit_event_threadsafe(event: dict) -> None:
    # `_run_campaign_background` runs on a plain thread (not the asyncio
    # loop), so `_broadcast` — a coroutine — can't just be awaited directly
    # from there; hand it to the loop captured at startup instead.
    if _main_loop is not None:
        asyncio.run_coroutine_threadsafe(_broadcast(event), _main_loop)


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


# REQ-BREED-2.1 — the higher-timeframe context each trading timeframe reads.
HIGHER_TF_BY_TIMEFRAME = {"M15": "H1", "M30": "H1", "H1": "H4", "H2": "D1", "H4": "D1"}

# Contract size per lot, used to price Settings → Costs commission against
# the position that fixed-fractional sizing implies.
CONTRACT_SIZE_BY_CATEGORY = {"fx": 100_000.0, "metal": 100.0, "index": 1.0}


def _spread_points(costs: dict, symbol: str, instrument) -> float:
    """AUTO SPREAD reads the live broker spread (the one thing the MT5
    connection is for besides execution — REQ-DATA-1.3); MANUAL uses the
    configured value. AUTO with no terminal connected falls back to MANUAL
    rather than silently backtesting at zero spread."""
    if costs["spread_mode"] != "AUTO":
        return float(costs["manual_spread_points"])
    quote = _mt5_bridge.get_live_quote(symbol) if _mt5_bridge.is_connected() else None
    if quote is None:
        return float(costs["manual_spread_points"])
    return float(quote.spread_points * instrument.point_divisor)


def build_campaign_config(settings: dict, symbol: str, timeframe: str) -> CampaignConfig:
    """Translates the Settings screen into engine parameters. Everything the
    user configures — risk, costs, cores, POWER, HIGHER TF — has to reach the
    engine here, or the Settings screen is decoration."""
    campaign_settings = settings["campaign"]
    costs = settings["costs"]
    preset = POWER_PRESETS.get(campaign_settings["power"], POWER_PRESETS["MEDIUM"])
    instrument = get_instrument(symbol)

    htf_df = None
    if campaign_settings["higher_tf_enabled"]:
        htf_timeframe = (
            campaign_settings["higher_tf_manual_timeframe"]
            if campaign_settings["higher_tf_mode"] == "MANUAL" and campaign_settings["higher_tf_manual_timeframe"]
            else HIGHER_TF_BY_TIMEFRAME.get(timeframe)
        )
        if htf_timeframe:
            htf_df = _load_bars(symbol, htf_timeframe, campaign_settings["history_depth_years"])

    # REQ-ENGINE-2 — the EVENT gene needs a calendar; without one it would
    # silently veto nothing, so an absent calendar simply means no events.
    news_events = load_calendar(
        get_config().data_dir / "calendar.json", get_config().data_dir / "calendar.csv"
    )

    return CampaignConfig(
        population_size=preset["population_size"],
        generations_per_tribe=preset["generations_per_tribe"],
        timeframe=timeframe,
        # Settings → Risk (REQ-SET-1.2) is a percentage; the backtester takes a fraction.
        risk_pct=settings["risk"]["per_trade_risk_pct"] / 100,
        # Costs are in the instrument's own points (0.00001 on EURUSD,
        # 0.001 on USDJPY/XAUUSD), not a fixed pip.
        point_size=1 / instrument.point_divisor,
        cost_points=_spread_points(costs, symbol, instrument) + float(costs["slippage_points"]),
        commission_per_lot=float(costs["commission_per_lot"]),
        contract_size=CONTRACT_SIZE_BY_CATEGORY.get(instrument.category, 100_000.0),
        max_workers=campaign_settings["cores"] or None,  # 0 = auto
        data_source="bundled" if get_config().built_in_data_only else "dukascopy",
        # SMART MODE (Settings → Campaign) biases which genes get drawn,
        # using what the Vault has already shown to work.
        gene_weights=smart_mode.load(get_config().data_dir).weights if campaign_settings["smart_mode"] else {},
        htf_df=htf_df,
        news_events=news_events,
    )


def _load_bars(symbol: str, timeframe: str, history_depth_years: int):
    """Historical bars trimmed to the configured HISTORY DEPTH — breeding on
    more history than the user asked for silently multiplies campaign time."""
    bars = get_historical_bars(symbol, timeframe)
    if bars.empty or not history_depth_years:
        return bars
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365 * history_depth_years)
    trimmed = bars[pd.DatetimeIndex(bars["timestamp"]) >= cutoff]
    return trimmed.reset_index(drop=True) if not trimmed.empty else bars


def _run_campaign_background(
    campaign_id: str, symbol: str, timeframe: str, gene_pool: GenePoolConfig, config: CampaignConfig, resume: bool = False
) -> None:
    global _active_campaign_id

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
            outcome = run_campaign(session, campaign_id, df, gene_pool, config, resume=resume, on_generation=on_generation)
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
                 "survivors_per_phase": {}, "detail": outcome.fail_reason}
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
        _active_campaign_id = None


_sync_state = SyncState()


async def _startup_delta_sync() -> None:
    config = get_config()
    if config.built_in_data_only:
        return  # REQ-DATA-1.6 — the debug toggle pins the app to bundled data.
    settings = load_settings(config.settings_path)
    symbols = settings["target_markets"]["enabled_symbols"] or list(BUNDLED_SYMBOLS)
    runner = DeltaSyncRunner(history_years=settings["campaign"]["history_depth_years"], state=_sync_state)
    try:
        await runner.sync_all(symbols)
    except asyncio.CancelledError:
        raise
    except Exception:
        import traceback

        traceback.print_exc()


SCHEDULER_TICK_SECONDS = 60


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
                and _active_campaign_id is None
                and _due(automation["last_campaign_started_at"], timedelta(hours=automation["recurring_interval_hours"]), now)
            ):
                try:
                    start_campaign()
                    update_section(config.settings_path, "automation", {"last_campaign_started_at": now.isoformat()})
                except HTTPException:
                    pass  # A campaign is already running; try again next tick.

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


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    config = get_config()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    # REQ-SET-6 — versioned migrations, auto-backed-up (when enabled) and
    # transactionally applied, instead of a bare create-tables-if-missing.
    try:
        run_migrations(config.vault_db_path, auto_backup=config.auto_backup_before_updates)
    except MigrationError:
        # REQ-SET-6.3 — the database is already rolled back to its
        # pre-migration state at this point; surface the failure loudly
        # (Electron detects the engine process exit and shows the error)
        # rather than starting the app against a possibly stale schema.
        raise
    engine = make_engine(config.vault_db_path)
    # REQ-ENGINE-4.2 — a campaign still "running" at this point means the
    # previous engine process died mid-campaign.
    with Session(engine) as session:
        mark_crashed_campaigns(session)
    poll_task = asyncio.create_task(_poll_campaigns_loop())
    # REQ-DATA-1.5 — delta sync runs on startup. It is a background task:
    # a first sync downloads years of history and must never block the API.
    sync_task = asyncio.create_task(_startup_delta_sync())
    # tasks.md 12.2 / REQ-SET-4.5 — recurring campaigns and automatic purge.
    scheduler_task = asyncio.create_task(_scheduler_loop())
    yield
    poll_task.cancel()
    sync_task.cancel()
    scheduler_task.cancel()


# FastAPI engine — design.md B.2, B.7. Bound to 127.0.0.1:7831 only,
# spawned/managed by the Electron main process as a child process.
app = FastAPI(title="QuantEngine Engine", lifespan=lifespan)

# The renderer (http://localhost:5173 in dev, file:// once packaged) is a
# different origin than 127.0.0.1:7831, so the browser's CORS policy blocks
# fetch() here without this — loopback-only binding is the real security
# boundary (design.md B.2), CORS is just satisfying the browser on top of it.
# Restricted to the renderer's own origins: "null" is what a packaged
# build's file:// page sends, 5173 is the Vite dev server. With "*" any
# website open in any browser on this machine could call these endpoints —
# including DELETE /vault/{id} and POST /vault/restore.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Live price reference/execution only — never a backtest history source.
# Single process-wide instance since MetaTrader5's Python API is itself a
# singleton (mt5.initialize() attaches to one terminal per process).
_mt5_bridge = MT5Bridge()


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


@app.get("/status")
def get_status() -> dict:
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        campaign = None
        if _active_campaign_id:
            campaign = session.get(Campaign, _active_campaign_id)
        if campaign is None:
            campaign = session.scalar(select(Campaign).where(Campaign.status == "running"))
        return {
            "state": "running" if campaign else "idle",
            "campaign": _serialize_campaign(campaign) if campaign else None,
        }


@app.get("/mt5/status")
def get_mt5_status() -> dict:
    # REQ-DASH-1.2, 1.3, 1.4 — Dashboard's MT5 connection card.
    if not _mt5_bridge.is_connected() and not _mt5_bridge.connect():
        return {"connected": False, "server": None, "latency_ms": None, "is_metaquotes_demo": False}

    started = time.perf_counter()
    server_info = _mt5_bridge.get_server_info()
    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    if server_info is None:
        return {"connected": False, "server": None, "latency_ms": None, "is_metaquotes_demo": False}

    return {
        "connected": True,
        "server": server_info["server"],
        "latency_ms": latency_ms,
        "is_metaquotes_demo": server_info["is_metaquotes_demo"],
    }


@app.get("/campaign/estimate")
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


@app.post("/campaign/start")
def start_campaign() -> dict:
    # REQ-BREED-1.1/1.2 — LAUNCH. One campaign at a time: run_campaign holds
    # a DB session for its full 15-60 minute run, so a second concurrent run
    # would just contend for the same SQLite file rather than usefully queue.
    global _active_campaign_id
    if _active_campaign_id is not None:
        raise HTTPException(status_code=409, detail="A campaign is already running")

    settings = load_settings(get_config().settings_path)
    target_markets = settings["target_markets"]
    symbol = next(iter(target_markets["enabled_symbols"]), "EURUSD")
    timeframe = next(iter(target_markets["enabled_timeframes"]), "H1")
    try:
        config = build_campaign_config(settings, symbol, timeframe)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    gene_pool = GenePoolConfig.load(get_config().gene_pool_path)

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

    _active_campaign_id = campaign_id
    threading.Thread(
        target=_run_campaign_background,
        args=(campaign_id, symbol, timeframe, gene_pool, config),
        daemon=True,
    ).start()
    return {"campaign_id": campaign_id, "status": "running"}


@app.get("/campaign/crashed")
def list_crashed_campaigns() -> list:
    # REQ-ENGINE-4.2 — the resume/discard prompt's data source: campaigns
    # the previous engine process left behind when it died mid-run.
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        rows = session.scalars(
            select(Campaign).where(Campaign.status == "crashed").order_by(Campaign.started_at.desc())
        ).all()
        return [_serialize_campaign(row) for row in rows if row.checkpoint_population]


@app.post("/campaign/{campaign_id}/resume")
def resume_campaign(campaign_id: str) -> dict:
    # REQ-ENGINE-4.3 — continue evolution from the last checkpoint.
    global _active_campaign_id
    if _active_campaign_id is not None:
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

    _active_campaign_id = campaign_id
    threading.Thread(
        target=_run_campaign_background,
        args=(campaign_id, symbol, timeframe, GenePoolConfig.load(get_config().gene_pool_path), config, True),
        daemon=True,
    ).start()
    return {"campaign_id": campaign_id, "status": "running", "resumed": True}


@app.post("/campaign/{campaign_id}/discard")
def discard_crashed_campaign(campaign_id: str) -> dict:
    # REQ-ENGINE-4.4 — mark cancelled and drop it from the queue.
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        try:
            discard_campaign(session, campaign_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"campaign_id": campaign_id, "status": "cancelled"}


@app.post("/campaign/pause")
def pause_campaign() -> dict:
    # The GA loop (app/engine/phases.py) has no cooperative-cancellation
    # hook to pause mid-generation — honest "not supported" beats a pause
    # button that silently does nothing.
    return {"status": "not_supported", "detail": "Pausing mid-run isn't implemented yet — the campaign keeps running."}


@app.post("/campaign/skip")
def skip_campaign() -> dict:
    return {"status": "not_supported", "detail": "Skipping a phase mid-run isn't implemented yet."}


def _active_genes(genome_json: str) -> list[dict]:
    # REQ-VAULT-1.4 — Inspector's "active genes": every non-null layer entry
    # plus each filter, parsed server-side so the frontend never needs its
    # own copy of the Genome shape.
    try:
        genome = Genome.from_dict(json.loads(genome_json))
    except (json.JSONDecodeError, KeyError, TypeError):
        return []
    genes = [{"layer": "signal", "name": genome.signal.name}]
    if genome.bias is not None:
        genes.append({"layer": "bias", "name": genome.bias.name})
    genes.extend({"layer": "filter", "name": f.name} for f in genome.filters)
    genes.append({"layer": "execution", "name": genome.execution.name})
    if genome.management is not None:
        genes.append({"layer": "management", "name": genome.management.name})
    return genes


def _mechanism(genome_json: str) -> str | None:
    genes = _active_genes(genome_json)
    signal = next((g["name"] for g in genes if g["layer"] == "signal"), None)
    return signal


def _serialize_strategy(row: Strategy, *, full: bool) -> dict:
    base = {
        "id": row.id,
        "data_source": row.data_source,
        "asset": row.asset,
        "timeframe": row.timeframe,
        "trades": row.trades,
        "return_pct": row.return_pct,
        "cagr_pct": row.cagr_pct,
        "max_dd_pct": row.max_dd_pct,
        "archetype": row.archetype,
        "mechanism": _mechanism(row.genome_json),
        "robust_pct": row.robust_pct,
        # REQ-VAULT-1.1 PERSISTENCE — real walk-forward result, not a
        # placeholder: the share of test windows the strategy was
        # profitable in (null when the history was too short to fold).
        "persistence_pct": row.persistence_pct,
        "icir": row.icir,
        "avg_hold_hours": row.avg_hold_hours,
        "score": row.score,
        "grade": row.grade,
        "deployed": row.deployed,
        "active_mt5": row.active_mt5,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
    if full:
        # REQ-VAULT-1.4 — full Inspector detail. equity_is/equity_oos are
        # honestly null: app/engine/vault_writer.py never populates them
        # (no engine code serializes an equity curve into the Vault yet).
        base["equity_is"] = json.loads(row.equity_is) if row.equity_is else None
        base["equity_oos"] = json.loads(row.equity_oos) if row.equity_oos else None
        base["active_genes"] = _active_genes(row.genome_json)
        base["walk_forward"] = json.loads(row.walk_forward) if row.walk_forward else None
    return base


@app.get("/vault")
def list_vault() -> list:
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        # Recycle Bin rows (REQ-SET-4.3) are still in the table but are not
        # part of the Vault until they are restored.
        rows = session.scalars(
            select(Strategy).where(Strategy.deleted_at.is_(None)).order_by(Strategy.created_at.desc())
        ).all()
        return [_serialize_strategy(row, full=True) for row in rows]


def _breakdown(rows: list[Strategy], key) -> list[dict]:
    """Per-group counts and averages, ordered by group size."""
    groups: dict[str, list[Strategy]] = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)

    summary = []
    for name, members in groups.items():
        returns = [r.return_pct for r in members if r.return_pct is not None]
        drawdowns = [r.max_dd_pct for r in members if r.max_dd_pct is not None]
        scores = [r.score for r in members if r.score is not None]
        summary.append(
            {
                "name": name,
                "count": len(members),
                "avg_return_pct": round(sum(returns) / len(returns), 2) if returns else None,
                "avg_drawdown_pct": round(sum(drawdowns) / len(drawdowns), 2) if drawdowns else None,
                "avg_score": round(sum(scores) / len(scores), 1) if scores else None,
                "profitable_pct": round(100 * sum(1 for r in returns if r > 0) / len(returns), 1) if returns else None,
            }
        )
    return sorted(summary, key=lambda entry: entry["count"], reverse=True)


@app.get("/analytics")
def get_analytics(top: int = 10) -> dict:
    """REQ-ANLY-1 — Vault-wide KPIs and breakdowns. Everything here is an
    aggregation of stored strategies; nothing is recomputed, so it reflects
    exactly what the Vault screen shows."""
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        rows = list(session.scalars(select(Strategy).where(Strategy.deleted_at.is_(None))))

        returns = [r.return_pct for r in rows if r.return_pct is not None]
        drawdowns = [r.max_dd_pct for r in rows if r.max_dd_pct is not None]
        top_performers = sorted(rows, key=lambda r: (r.score or 0), reverse=True)[:top]

        return {
            "total_strategies": len(rows),
            "avg_return_pct": round(sum(returns) / len(returns), 2) if returns else None,
            "avg_drawdown_pct": round(sum(drawdowns) / len(drawdowns), 2) if drawdowns else None,
            "profitable_pct": round(100 * sum(1 for r in returns if r > 0) / len(returns), 1) if returns else None,
            "good_grade_count": sum(1 for r in rows if r.grade == "GOOD"),
            "deployed_count": sum(1 for r in rows if r.deployed),
            "by_market": _breakdown(rows, lambda r: f"{r.asset} {r.timeframe}"),
            "by_archetype": _breakdown(rows, lambda r: r.archetype or "UNKNOWN"),
            "by_symbol": _breakdown(rows, lambda r: r.asset),
            "by_timeframe": _breakdown(rows, lambda r: r.timeframe),
            "by_broker": _breakdown(rows, lambda r: r.data_source),
            "top_performers": [_serialize_strategy(row, full=False) for row in top_performers],
        }


@app.get("/neural-map")
def get_neural_map() -> dict:
    # REQ-MAP-1 — 2D UMAP projection + auto-labeled clusters of the whole
    # Vault. `min_strategies` lets the frontend render an honest "need N
    # more strategies" message instead of a blank canvas when there isn't
    # enough population yet (design.md Part A's Neural Map empty-state note).
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        rows = list(session.scalars(select(Strategy).where(Strategy.deleted_at.is_(None))))
        result = compute_neural_map(rows)
        if result is None:
            return {"ready": False, "strategy_count": len(rows), "min_strategies": MIN_STRATEGIES_FOR_MAP, "nodes": [], "clusters": []}
        return {
            "ready": True,
            "strategy_count": len(rows),
            "min_strategies": MIN_STRATEGIES_FOR_MAP,
            "nodes": [vars(n) for n in result.nodes],
            "clusters": [vars(c) for c in result.clusters],
        }


@app.delete("/vault/{strategy_id}")
def delete_strategy(strategy_id: str) -> dict:
    # REQ-VAULT-2.2 — DELETE is a Free-tier action; real deletion, not a stub.
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        row = session.get(Strategy, strategy_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Unknown strategy: {strategy_id}")
        session.delete(row)
        session.commit()
    return {"deleted": strategy_id}


def _generate_ea_for(strategy_id: str) -> tuple[Strategy, "GeneratedEa"]:
    """Loads a Vault strategy and regenerates its EA from the stored genome."""
    settings = load_settings(get_config().settings_path)
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        row = session.get(Strategy, strategy_id)
        if row is None or row.deleted_at is not None:
            raise HTTPException(status_code=404, detail=f"Unknown strategy: {strategy_id}")
        session.expunge(row)

    try:
        genome = Genome.from_dict(json.loads(row.genome_json))
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="This strategy's genome could not be read") from exc

    try:
        generated = generate_ea(
            genome,
            row.id,
            row.asset,
            row.timeframe,
            # The EA risks what Settings says, not what the backtest used —
            # position size is the user's decision, the rules are not.
            risk_pct=settings["risk"]["per_trade_risk_pct"] / 100,
            max_spread_points=float(settings["costs"]["manual_spread_points"]),
        )
    except UnsupportedGeneError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return row, generated


class ExportEaRequest(BaseModel):
    destination_dir: str


@app.post("/vault/{strategy_id}/export-mql5")
def post_export_mql5(strategy_id: str, body: ExportEaRequest) -> dict:
    """REQ-TIER-1.3 (MQL5 export) — writes the EA source to a folder of the
    user's choosing, for people who compile or review it themselves."""
    _, generated = _generate_ea_for(strategy_id)
    destination_dir = Path(body.destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / generated.filename
    destination.write_text(generated.source, encoding="utf-8")
    return {"exported_path": str(destination), "filename": generated.filename}


@app.post("/deploy/{strategy_id}")
def deploy_strategy(strategy_id: str) -> dict:
    """REQ-VAULT-2.1 — generate the EA from the stored genome, install it
    into every MT5 terminal on this machine and compile it. Tier gating is
    the renderer's job (REQ-TIER-1.5); this endpoint does the work."""
    row, generated = _generate_ea_for(strategy_id)
    result = install_ea(generated.filename, generated.source)
    if not result.installed:
        raise HTTPException(status_code=409, detail=result.detail)

    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        strategy = session.get(Strategy, strategy_id)
        if strategy is not None:
            strategy.deployed = True
            # active_mt5 stays false until the EA is actually attached to a
            # chart — installing a file is not the same as running it.
            session.commit()

    return {
        "status": "deployed",
        "strategy_id": row.id,
        "filename": generated.filename,
        "installed_paths": result.installed_paths,
        "compiled_paths": result.compiled_paths,
        "detail": result.detail,
    }


def _revalidate(strategy_id: str, full: bool) -> dict:
    """Shared by REFRESH and REASSESS: re-run a stored genome against the
    data that exists now. The genome is never re-optimised — re-fitting a
    strategy until it looks good again is how a Vault fills with strategies
    that have only ever been fitted."""
    settings = load_settings(get_config().settings_path)
    engine = make_engine(get_config().vault_db_path)

    with Session(engine) as session:
        strategy = session.get(Strategy, strategy_id)
        if strategy is None or strategy.deleted_at is not None:
            raise HTTPException(status_code=404, detail=f"Unknown strategy: {strategy_id}")

        try:
            config = build_campaign_config(settings, strategy.asset, strategy.timeframe)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        df = _load_bars(strategy.asset, strategy.timeframe, settings["campaign"]["history_depth_years"])
        if len(df) < int(MIN_IS_BARS / IS_FRACTION):
            raise HTTPException(
                status_code=409,
                detail=f"Not enough {strategy.timeframe} history cached for {strategy.asset} to re-check this strategy.",
            )

        gene_pool = GenePoolConfig.load(get_config().gene_pool_path)
        runner = reassess_strategy_engine if full else refresh_strategy_engine
        result = runner(strategy, df, gene_pool, config)
        session.commit()

    return {"status": "ok", **result.to_dict()}


@app.post("/vault/{strategy_id}/refresh")
def refresh_strategy(strategy_id: str) -> dict:
    # REQ-VAULT-2.1 REFRESH — restate the strategy's numbers against the
    # data cached today, without re-running the Phase 4 gates.
    return _revalidate(strategy_id, full=False)


@app.post("/vault/{strategy_id}/reassess")
def reassess_strategy(strategy_id: str) -> dict:
    # REQ-VAULT-2.1 REASSESS — REFRESH plus the robustness and walk-forward
    # gates, re-scored. `still_passes` says whether it would be approved
    # if it were bred today.
    return _revalidate(strategy_id, full=True)


@app.post("/vault/{strategy_id}/parity-test")
def parity_test_strategy(strategy_id: str) -> dict:
    return {"status": "not_supported", "detail": "Parity test isn't implemented yet.", "strategy_id": strategy_id}


@app.get("/portfolios")
def list_portfolios() -> list:
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        rows = session.scalars(select(Portfolio).order_by(Portfolio.created_at.desc())).all()
        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "member_count": len(_portfolio_members(session, p.id)),
            }
            for p in rows
        ]


class CreatePortfolioRequest(BaseModel):
    name: str
    description: str | None = None


@app.post("/portfolios")
def create_portfolio(body: CreatePortfolioRequest) -> dict:
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        portfolio = Portfolio(id=uuid.uuid4().hex, name=body.name, description=body.description)
        session.add(portfolio)
        session.commit()
        return {"id": portfolio.id, "name": portfolio.name, "description": portfolio.description}


def _portfolio_members(session: Session, portfolio_id: str) -> list[Strategy]:
    member_ids = session.scalars(
        select(PortfolioStrategy.strategy_id).where(PortfolioStrategy.portfolio_id == portfolio_id)
    ).all()
    if not member_ids:
        return []
    return list(
        session.scalars(
            select(Strategy).where(Strategy.id.in_(member_ids)).where(Strategy.deleted_at.is_(None))
        )
    )


@app.get("/portfolios/{portfolio_id}")
def get_portfolio(portfolio_id: str) -> dict:
    # REQ-PORT-1.1 — the portfolio and its members, for the STRATEGIES tab.
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        portfolio = session.get(Portfolio, portfolio_id)
        if portfolio is None:
            raise HTTPException(status_code=404, detail=f"Unknown portfolio: {portfolio_id}")
        members = _portfolio_members(session, portfolio_id)
        return {
            "id": portfolio.id,
            "name": portfolio.name,
            "description": portfolio.description,
            "members": [_serialize_strategy(row, full=True) for row in members],
        }


@app.get("/portfolios/{portfolio_id}/health")
def get_portfolio_health(portfolio_id: str) -> dict:
    # REQ-PORT-1.2/1.3/1.4 — AVG CORRELATION, Diversity Grade and the
    # recommendation chips, plus the combined equity curve (HEALTH tab).
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        if session.get(Portfolio, portfolio_id) is None:
            raise HTTPException(status_code=404, detail=f"Unknown portfolio: {portfolio_id}")
        members = _portfolio_members(session, portfolio_id)
        return compute_health(members).to_dict()


class TrueSimRequest(BaseModel):
    sizing: str = "COMPOUNDING"  # COMPOUNDING | FIXED
    max_aggregate_risk_pct: float = 5.0


@app.post("/portfolios/{portfolio_id}/true-sim")
def post_true_sim(portfolio_id: str, body: TrueSimRequest) -> dict:
    """REQ-PORT-2 — replay the members onto one account with Symbol Lock
    and a cap on total open risk, and report what those guardrails cost."""
    settings = load_settings(get_config().settings_path)
    engine = make_engine(get_config().vault_db_path)

    with Session(engine) as session:
        if session.get(Portfolio, portfolio_id) is None:
            raise HTTPException(status_code=404, detail=f"Unknown portfolio: {portfolio_id}")
        members = _portfolio_members(session, portfolio_id)
        if not members:
            raise HTTPException(status_code=409, detail="This portfolio has no members to simulate")
        markets = {(m.asset, m.timeframe) for m in members}
        session.expunge_all()

    depth_years = settings["campaign"]["history_depth_years"]
    bars_by_market = {}
    for symbol, timeframe in markets:
        bars = _load_bars(symbol, timeframe, depth_years)
        if not bars.empty:
            bars_by_market[(symbol, timeframe)] = bars
    if not bars_by_market:
        raise HTTPException(status_code=409, detail="No cached history for these members' markets.")

    # Every member is re-simulated on the same cost and risk settings, so
    # the account they share is internally consistent.
    config = build_campaign_config(settings, members[0].asset, members[0].timeframe)
    result = run_true_sim(
        members,
        bars_by_market,
        GenePoolConfig.load(get_config().gene_pool_path),
        config,
        sizing=body.sizing,
        max_aggregate_risk_pct=body.max_aggregate_risk_pct,
    )
    return result.to_dict()


@app.delete("/portfolios/{portfolio_id}")
def delete_portfolio(portfolio_id: str) -> dict:
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        portfolio = session.get(Portfolio, portfolio_id)
        if portfolio is None:
            raise HTTPException(status_code=404, detail=f"Unknown portfolio: {portfolio_id}")
        for membership in session.scalars(
            select(PortfolioStrategy).where(PortfolioStrategy.portfolio_id == portfolio_id)
        ):
            session.delete(membership)
        session.delete(portfolio)
        session.commit()
    return {"deleted": portfolio_id}


@app.delete("/portfolios/{portfolio_id}/members/{strategy_id}")
def remove_portfolio_member(portfolio_id: str, strategy_id: str) -> dict:
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        membership = session.get(PortfolioStrategy, {"portfolio_id": portfolio_id, "strategy_id": strategy_id})
        if membership is None:
            raise HTTPException(status_code=404, detail="That strategy is not in this portfolio")
        session.delete(membership)
        session.commit()
    return {"portfolio_id": portfolio_id, "strategy_id": strategy_id, "removed": True}


@app.post("/vault/{strategy_id}/add-to-portfolio/{portfolio_id}")
def add_strategy_to_portfolio(strategy_id: str, portfolio_id: str) -> dict:
    # REQ-VAULT-2.2 — ADD TO PORTFOLIO is a Free-tier action.
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        if session.get(Strategy, strategy_id) is None:
            raise HTTPException(status_code=404, detail=f"Unknown strategy: {strategy_id}")
        if session.get(Portfolio, portfolio_id) is None:
            raise HTTPException(status_code=404, detail=f"Unknown portfolio: {portfolio_id}")
        existing = session.get(PortfolioStrategy, {"portfolio_id": portfolio_id, "strategy_id": strategy_id})
        if existing is None:
            session.add(PortfolioStrategy(portfolio_id=portfolio_id, strategy_id=strategy_id))
            session.commit()
    return {"portfolio_id": portfolio_id, "strategy_id": strategy_id}


class BuiltInDataOnlyRequest(BaseModel):
    enabled: bool


@app.get("/data/config")
def get_data_config() -> dict:
    config = get_config()
    return {
        "built_in_data_only": config.built_in_data_only,
        "bundled_symbols": list(BUNDLED_SYMBOLS),
    }


@app.get("/data/instruments")
def get_instruments() -> dict:
    # REQ-SET-1.1 — Settings → Target Markets: full catalog, not just the
    # pre-bundled subset (/data/config's bundled_symbols).
    return {"symbols": list(INSTRUMENTS.keys()), "timeframes": list(TIMEFRAMES)}


@app.post("/data/built-in-only")
def set_data_built_in_only(body: BuiltInDataOnlyRequest) -> dict:
    # REQ-DATA-1.6 — debug toggle, forces the pre-bundled subset regardless
    # of full cache/broker availability.
    set_built_in_data_only(body.enabled)
    return {"built_in_data_only": body.enabled}


@app.post("/data/sync")
async def sync_data(symbols: list[str] | None = None) -> dict:
    """Manual re-sync (Settings → System). The startup sync is the normal
    path — REQ-DATA-1.5 — and this returns immediately rather than holding
    the request open for what can be an hours-long first download; poll
    /data/sync/status for progress."""
    if _sync_state.running:
        raise HTTPException(status_code=409, detail="A data sync is already running")
    settings = load_settings(get_config().settings_path)
    target_symbols = symbols or settings["target_markets"]["enabled_symbols"] or list(BUNDLED_SYMBOLS)
    runner = DeltaSyncRunner(history_years=settings["campaign"]["history_depth_years"], state=_sync_state)
    asyncio.create_task(runner.sync_all(target_symbols))
    return {"started": True, "symbols": target_symbols}


@app.get("/data/sync/status")
def get_sync_status() -> dict:
    return _sync_state.snapshot()


@app.get("/data/coverage")
def get_data_coverage() -> dict:
    """How much history is actually cached per enabled symbol/timeframe —
    the Dashboard needs this to tell the user whether a campaign can run
    (see MIN_HISTORY_DAYS in app/engine/campaign.py)."""
    settings = load_settings(get_config().settings_path)
    depth_years = settings["campaign"]["history_depth_years"]
    min_bars = int(MIN_IS_BARS / IS_FRACTION)
    coverage = []
    for symbol in settings["target_markets"]["enabled_symbols"]:
        for timeframe in settings["target_markets"]["enabled_timeframes"]:
            bars = _load_bars(symbol, timeframe, depth_years)
            if bars.empty:
                coverage.append(
                    {"symbol": symbol, "timeframe": timeframe, "bars": 0, "days": 0, "ready": False}
                )
                continue
            timestamps = pd.DatetimeIndex(bars["timestamp"])
            days = int((timestamps.max() - timestamps.min()).total_seconds() // 86_400)
            coverage.append(
                {
                    "symbol": symbol, "timeframe": timeframe, "bars": len(bars), "days": days,
                    "from": timestamps.min().isoformat(), "to": timestamps.max().isoformat(),
                    # Both gates matter: enough calendar span for the metrics
                    # to mean anything, and enough bars for Phase 0's trade
                    # minimum to be reachable at all.
                    "ready": days >= MIN_HISTORY_DAYS and len(bars) >= min_bars,
                }
            )
    return {"min_days": MIN_HISTORY_DAYS, "min_bars": min_bars, "coverage": coverage}


class AutoBackupRequest(BaseModel):
    enabled: bool


class ExportVaultRequest(BaseModel):
    destination_dir: str


class RestoreVaultRequest(BaseModel):
    backup_path: str


@app.get("/settings/auto-backup")
def get_auto_backup() -> dict:
    return {"auto_backup_before_updates": get_config().auto_backup_before_updates}


@app.post("/settings/auto-backup")
def set_auto_backup(body: AutoBackupRequest) -> dict:
    # REQ-SET-5.3 — default ON; wired into the migration runner at startup.
    set_auto_backup_before_updates(body.enabled)
    return {"auto_backup_before_updates": body.enabled}


@app.post("/vault/export")
def post_vault_export(body: ExportVaultRequest) -> dict:
    # REQ-SET-5.1 — `destination_dir` comes from the Electron main process's
    # native save/folder dialog; this endpoint just performs the copy.
    try:
        exported_path = export_vault(get_config().vault_db_path, Path(body.destination_dir))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"exported_path": str(exported_path)}


@app.post("/vault/restore")
def post_vault_restore(body: RestoreVaultRequest) -> dict:
    # REQ-SET-5.2 — destructive. The confirmation modal is the caller's
    # (renderer's) responsibility; this endpoint executes unconditionally.
    backup_path = Path(body.backup_path)
    if not backup_path.exists():
        raise HTTPException(status_code=404, detail=f"Backup file not found: {backup_path}")
    restore_database(get_config().vault_db_path, backup_path)
    return {"restored_from": str(backup_path)}


@app.post("/vault/purge")
def post_vault_purge() -> dict:
    # REQ-SET-4 — evaluates the configured purge criteria against the Vault.
    # Purged rows go to the Recycle Bin (recoverable for 30 days), portfolio
    # members are protected, and a running campaign defers the whole run.
    settings = load_settings(get_config().settings_path)
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        empty_expired_recycle_bin(session)
        result = run_purge(session, settings)
    return {
        "purged": result.purged,
        "protected": result.protected,
        "reasons": result.reasons,
        "skipped_reason": result.skipped_reason,
    }


@app.get("/vault/recycle-bin")
def get_recycle_bin() -> list:
    # REQ-SET-4.3 — what is recoverable, and why it was purged.
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        rows = session.scalars(
            select(Strategy).where(Strategy.deleted_at.is_not(None)).order_by(Strategy.deleted_at.desc())
        ).all()
        return [
            {
                **_serialize_strategy(row, full=False),
                "deleted_at": row.deleted_at.isoformat() if row.deleted_at else None,
                "delete_reason": row.delete_reason,
                "recoverable_until": (row.deleted_at + timedelta(days=RECYCLE_BIN_RETENTION_DAYS)).isoformat()
                if row.deleted_at
                else None,
            }
            for row in rows
        ]


@app.post("/vault/recycle-bin/{strategy_id}/restore")
def post_recycle_bin_restore(strategy_id: str) -> dict:
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        if not restore_from_recycle_bin(session, strategy_id):
            raise HTTPException(status_code=404, detail=f"Not in the Recycle Bin: {strategy_id}")
    return {"restored": strategy_id}


@app.get("/settings/app")
def get_app_settings() -> dict:
    # REQ-SET-1, 4, 7 — Target Markets, Risk, Campaign, Costs, Broker Caps,
    # Vault Maintenance, Notifications, Smart Mode: one local JSON blob,
    # design.md B.3 ("Settings, gene pool config, broker caps | Local").
    return load_settings(get_config().settings_path)


@app.post("/settings/app/{section}")
def patch_app_settings(section: str, patch: dict) -> dict:
    try:
        return update_section(get_config().settings_path, section, patch)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


class GenePoolUpdateRequest(BaseModel):
    genes: dict[str, bool]  # "{layer}::{name}" -> enabled, per GenePoolConfig.to_json


def _gene_pool_response(config: GenePoolConfig) -> dict:
    return {
        "genes": {f"{layer}::{name}": config.is_enabled(layer, name) for (layer, name) in ALL_GENES},
        "layer_counts": config.layer_counts(),
    }


@app.get("/settings/gene-pool")
def get_gene_pool() -> dict:
    # REQ-SET-2 — all 46 genes grouped by layer, per-gene toggle + per-layer count.
    return _gene_pool_response(GenePoolConfig.load(get_config().gene_pool_path))


@app.post("/settings/gene-pool")
def set_gene_pool(body: GenePoolUpdateRequest) -> dict:
    config = GenePoolConfig.load(get_config().gene_pool_path)
    for key, enabled in body.genes.items():
        layer, name = key.split("::", 1)
        try:
            config.set_enabled(layer, name, enabled)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    config.save(get_config().gene_pool_path)
    return _gene_pool_response(config)


class TestTelegramRequest(BaseModel):
    bot_token: str | None = None
    chat_id: str | None = None


@app.post("/notifications/test-telegram")
def post_test_telegram(body: TestTelegramRequest) -> dict:
    """REQ-SET-7.3 — TEST NOTIFICATION for the Telegram channel. Uses the
    values passed in (so the user can test before saving) and falls back to
    what is stored."""
    settings = load_settings(get_config().settings_path)
    notifications = settings["notifications"]
    result = send_telegram(
        body.bot_token if body.bot_token is not None else notifications["telegram_bot_token"],
        body.chat_id if body.chat_id is not None else notifications["telegram_chat_id"],
        "QuantEngine test notification — this channel works.",
    )
    return {"sent": result.sent, "detail": result.detail}


@app.get("/smart-mode")
def get_smart_mode() -> dict:
    """tasks.md 12.3 — what Smart Mode has learned, and from how much."""
    fitness = smart_mode.load(get_config().data_dir)
    weights = fitness.weights
    ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    return {
        "learned": bool(weights),
        "strategies_learned_from": fitness.strategies_learned_from,
        "computed_at": fitness.computed_at,
        "reason": fitness.reason,
        "min_strategies": smart_mode.MIN_STRATEGIES_TO_LEARN,
        "favoured": [{"gene": gene, "weight": weight} for gene, weight in ranked[:10]],
        "avoided": [{"gene": gene, "weight": weight} for gene, weight in ranked[-10:]],
    }


@app.post("/smart-mode/rebuild")
def post_smart_mode_rebuild() -> dict:
    """Re-learns from the Vault as it stands now."""
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        rows = list(session.scalars(select(Strategy).where(Strategy.deleted_at.is_(None))))
    fitness = smart_mode.compute_gene_fitness(rows)
    smart_mode.save(fitness, get_config().data_dir)
    return get_smart_mode()


@app.post("/smart-mode/reset")
def post_smart_mode_reset() -> dict:
    # Settings → Smart Mode's reset action: campaigns go back to sampling
    # every enabled gene uniformly.
    removed = smart_mode.clear(get_config().data_dir)
    return {"cleared": removed}


@app.get("/system/info")
def get_system_info() -> dict:
    # REQ-SET (System section) — paths shown in Settings → System, and the
    # Support/Diagnostics snapshot (REQ-SET, Support section).
    config = get_config()
    return {
        "data_dir": str(config.data_dir),
        "vault_db_path": str(config.vault_db_path),
        "cache_dir": str(config.cache_dir),
        "built_in_data_only": config.built_in_data_only,
    }


@app.websocket("/events")
async def events(websocket: WebSocket) -> None:
    # design.md B.7 — real-time stream for the BREED screen. Fed by
    # _poll_campaigns_loop (registered in `lifespan`), not by this handler
    # itself; it only needs to keep the connection open and track clients.
    await websocket.accept()
    _ws_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(websocket)
