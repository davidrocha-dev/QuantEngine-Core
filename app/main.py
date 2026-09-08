import asyncio
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.broker.mt5_bridge import MT5Bridge
from app.config import get_config, set_auto_backup_before_updates, set_built_in_data_only
from app.data.provider import get_historical_bars
from app.data.sync import DeltaSyncRunner
from app.data.symbols import BUNDLED_SYMBOLS, INSTRUMENTS, TIMEFRAMES
from app.db import make_engine
from app.engine.campaign import CampaignConfig, mark_crashed_campaigns, run_campaign
from app.engine.campaign_estimate import estimate_campaign_minutes
from app.engine.neural_map import MIN_STRATEGIES_FOR_MAP, compute_neural_map
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


def _run_campaign_background(campaign_id: str, symbol: str, timeframe: str, gene_pool: GenePoolConfig, config: CampaignConfig) -> None:
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
        df = get_historical_bars(symbol, timeframe)
        with Session(engine) as session:
            run_campaign(session, campaign_id, df, gene_pool, config, on_generation=on_generation)
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
    yield
    poll_task.cancel()


# FastAPI engine — design.md B.2, B.7. Bound to 127.0.0.1:7831 only,
# spawned/managed by the Electron main process as a child process.
app = FastAPI(title="QuantEngine Engine", lifespan=lifespan)

# The renderer (http://localhost:5173 in dev, file:// once packaged) is a
# different origin than 127.0.0.1:7831, so the browser's CORS policy blocks
# fetch() here without this — loopback-only binding is the real security
# boundary (design.md B.2), CORS is just satisfying the browser on top of it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    # REQ-DASH-1.5 — 15-60 min range based on detected CPU cores.
    cpu_cores = os.cpu_count() or 1
    min_minutes, max_minutes = estimate_campaign_minutes(cpu_cores)
    return {"cpu_cores": cpu_cores, "min_minutes": min_minutes, "max_minutes": max_minutes}


@app.post("/campaign/start")
def start_campaign() -> dict:
    # REQ-BREED-1.1/1.2 — LAUNCH. One campaign at a time: run_campaign holds
    # a DB session for its full 15-60 minute run, so a second concurrent run
    # would just contend for the same SQLite file rather than usefully queue.
    global _active_campaign_id
    if _active_campaign_id is not None:
        raise HTTPException(status_code=409, detail="A campaign is already running")

    settings = load_settings(get_config().settings_path)
    campaign_settings = settings["campaign"]
    target_markets = settings["target_markets"]
    symbol = next(iter(target_markets["enabled_symbols"]), "EURUSD")
    timeframe = next(iter(target_markets["enabled_timeframes"]), "H1")
    preset = POWER_PRESETS.get(campaign_settings["power"], POWER_PRESETS["MEDIUM"])
    config = CampaignConfig(
        population_size=preset["population_size"],
        generations_per_tribe=preset["generations_per_tribe"],
        timeframe=timeframe,
    )
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
    return base


@app.get("/vault")
def list_vault() -> list:
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        rows = session.scalars(select(Strategy).order_by(Strategy.created_at.desc())).all()
        return [_serialize_strategy(row, full=True) for row in rows]


@app.get("/neural-map")
def get_neural_map() -> dict:
    # REQ-MAP-1 — 2D UMAP projection + auto-labeled clusters of the whole
    # Vault. `min_strategies` lets the frontend render an honest "need N
    # more strategies" message instead of a blank canvas when there isn't
    # enough population yet (design.md Part A's Neural Map empty-state note).
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        rows = list(session.scalars(select(Strategy)))
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


@app.post("/deploy/{strategy_id}")
def deploy_strategy(strategy_id: str) -> dict:
    # REQ-VAULT-2.1 — Pro+ action. No MQL5 codegen/MT5 install pipeline
    # exists yet (that's design.md Part C / later phases), so this is
    # honestly unsupported rather than silently pretending to deploy.
    return {"status": "not_supported", "detail": "Deploy isn't implemented yet.", "strategy_id": strategy_id}


@app.post("/vault/{strategy_id}/refresh")
def refresh_strategy(strategy_id: str) -> dict:
    return {"status": "not_supported", "detail": "Refresh isn't implemented yet.", "strategy_id": strategy_id}


@app.post("/vault/{strategy_id}/reassess")
def reassess_strategy(strategy_id: str) -> dict:
    return {"status": "not_supported", "detail": "Reassess isn't implemented yet.", "strategy_id": strategy_id}


@app.post("/vault/{strategy_id}/parity-test")
def parity_test_strategy(strategy_id: str) -> dict:
    return {"status": "not_supported", "detail": "Parity test isn't implemented yet.", "strategy_id": strategy_id}


@app.get("/portfolios")
def list_portfolios() -> list:
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        rows = session.scalars(select(Portfolio).order_by(Portfolio.created_at.desc())).all()
        return [{"id": p.id, "name": p.name, "description": p.description} for p in rows]


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
    # REQ-DATA-1.5 — incremental delta sync, called on app startup.
    runner = DeltaSyncRunner()
    target_symbols = symbols or list(BUNDLED_SYMBOLS)
    results = await runner.sync_all(target_symbols)
    return {"synced_hours": results}


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
    # REQ-SET-4 — evaluates configured purge criteria against Vault rows.
    # The Vault table itself lands in Phase 8; until then there is nothing
    # to purge, so this honestly reports zero rather than faking a count.
    return {"purged": 0}


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
