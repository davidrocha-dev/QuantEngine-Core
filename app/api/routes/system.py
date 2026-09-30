"""Engine status, MetaTrader connection, settings, notifications, Smart Mode and the event socket."""


import time

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api import runtime
from app.api.campaign_setup import unavailable_genes
from app.api.routes.campaign import _serialize_campaign
from app.api.runtime import _mt5_bridge, _ws_clients
from app.broker.terminal_watch import terminal_is_running
from app.config import get_config, set_auto_backup_before_updates
from app.db import make_engine
from app.engine import smart_mode
from app.engine.notifications import send_telegram
from app.genome.gene_pool import GenePoolConfig
from app.genome.genes import ALL_GENES
from app.logging_setup import log_dir
from app.models import Campaign, Strategy
from app.settings_store import load_settings, update_section

router = APIRouter()

@router.get("/status")
def get_status() -> dict:
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        campaign = None
        if runtime.active_campaign_id:
            campaign = session.get(Campaign, runtime.active_campaign_id)
        if campaign is None:
            campaign = session.scalar(select(Campaign).where(Campaign.status == "running"))
        # Derived from the campaign's own status, not from the module
        # global being set. `runtime.active_campaign_id` is cleared when a run
        # finishes, and a crash that skips that cleanup used to leave
        # /status reporting a phantom running campaign for the rest of the
        # process - which disables the Launch button against a campaign
        # that ended minutes ago.
        running = campaign is not None and campaign.status == "running"
        if campaign is None:
            # Nothing is running, so report the most recent one rather
            # than nothing at all. BREED lost every detail of a campaign
            # the moment it finished otherwise - its asset, its timeframe,
            # and the start time the live feed filters the Vault by, so
            # the feed went empty and the panel showed a blank asset at
            # precisely the moment you want to see what was produced.
            campaign = session.scalar(select(Campaign).order_by(Campaign.started_at.desc()))
        return {
            "state": "running" if running else "idle",
            "campaign": _serialize_campaign(campaign) if campaign else None,
        }


@router.get("/mt5/status")
def get_mt5_status() -> dict:
    """REQ-DASH-1.2, 1.3, 1.4 — Dashboard's MT5 connection card.

    Polled, so it must never start anything: `connect()` attaches to a
    terminal that is already open and reports nothing otherwise. Opening
    this app used to open MetaTrader, every time, because this line ran.
    """
    if not _mt5_bridge.is_connected() and not _mt5_bridge.connect():
        return {
            "connected": False, "server": None, "latency_ms": None, "is_metaquotes_demo": False,
            "terminal_running": terminal_is_running(),
        }

    started = time.perf_counter()
    server_info = _mt5_bridge.get_server_info()
    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    if server_info is None:
        return {
            "connected": False, "server": None, "latency_ms": None, "is_metaquotes_demo": False,
            "terminal_running": terminal_is_running(),
        }

    return {
        "connected": True,
        "server": server_info["server"],
        "latency_ms": latency_ms,
        "is_metaquotes_demo": server_info["is_metaquotes_demo"],
        # What Settings -> Broker caps -> Server time should say.
        "server_gmt_offset_hours": _mt5_bridge.server_gmt_offset_hours(),
    }


@router.post("/mt5/connect")
def connect_mt5() -> dict:
    """Start MetaTrader and attach to it — the deliberate version.

    Separate from the status poll on purpose. Launching a broker terminal
    is a decision, and the person using the app is the one who gets to
    make it.
    """
    if not _mt5_bridge.connect(launch=True):
        raise HTTPException(
            status_code=409,
            detail=(
                "Could not start MetaTrader 5. Check it is installed, and that you have logged into an "
                "account in it at least once."
            ),
        )
    return get_mt5_status()


class AutoBackupRequest(BaseModel):
    enabled: bool


@router.get("/settings/auto-backup")
def get_auto_backup() -> dict:
    return {"auto_backup_before_updates": get_config().auto_backup_before_updates}


@router.post("/settings/auto-backup")
def set_auto_backup(body: AutoBackupRequest) -> dict:
    # REQ-SET-5.3 — default ON; wired into the migration runner at startup.
    set_auto_backup_before_updates(body.enabled)
    return {"auto_backup_before_updates": body.enabled}


@router.get("/settings/app")
def get_app_settings() -> dict:
    # REQ-SET-1, 4, 7 — Target Markets, Risk, Campaign, Costs, Broker Caps,
    # Vault Maintenance, Notifications, Smart Mode: one local JSON blob,
    # design.md B.3 ("Settings, gene pool config, broker caps | Local").
    return load_settings(get_config().settings_path)


@router.post("/settings/app/{section}")
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
        # Enabled or not, a campaign leaves these out; the screen says so
        # instead of offering a toggle that changes nothing.
        "unavailable": unavailable_genes(load_settings(get_config().settings_path)),
    }


@router.get("/settings/gene-pool")
def get_gene_pool() -> dict:
    # REQ-SET-2 — all 46 genes grouped by layer, per-gene toggle + per-layer count.
    return _gene_pool_response(GenePoolConfig.load(get_config().gene_pool_path))


@router.post("/settings/gene-pool")
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


@router.post("/notifications/test-telegram")
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


@router.get("/smart-mode")
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


@router.post("/smart-mode/rebuild")
def post_smart_mode_rebuild() -> dict:
    """Re-learns from the Vault as it stands now."""
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        rows = list(session.scalars(select(Strategy).where(Strategy.deleted_at.is_(None))))
    fitness = smart_mode.compute_gene_fitness(rows)
    smart_mode.save(fitness, get_config().data_dir)
    return get_smart_mode()


@router.post("/smart-mode/reset")
def post_smart_mode_reset() -> dict:
    # Settings → Smart Mode's reset action: campaigns go back to sampling
    # every enabled gene uniformly.
    removed = smart_mode.clear(get_config().data_dir)
    return {"cleared": removed}


@router.get("/system/info")
def get_system_info() -> dict:
    # REQ-SET (System section) — paths shown in Settings → System, and the
    # Support/Diagnostics snapshot (REQ-SET, Support section).
    config = get_config()
    settings = load_settings(config.settings_path)
    return {
        "data_dir": str(config.data_dir),
        "vault_db_path": str(config.vault_db_path),
        "cache_dir": str(config.cache_dir),
        "built_in_data_only": config.built_in_data_only,
        # REQ-NFR-2 — where to look when something went wrong, and whether
        # anything is being sent anywhere (it is not, unless opted in).
        "log_dir": str(log_dir(config.data_dir)),
        "crash_reporting_enabled": settings["system"]["crash_reporting_enabled"],
    }


@router.websocket("/events")
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
