"""Historical data: coverage, sync, instruments and custom markets."""

import asyncio

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api import runtime
from app.api.campaign_setup import _load_bars
from app.api.runtime import _sync_state
from app.config import get_config, set_built_in_data_only
from app.data.symbols import BUNDLED_SYMBOLS, INSTRUMENTS, TIMEFRAMES
from app.data.sync import DeltaSyncRunner
from app.engine.campaign import (
    MIN_HISTORY_DAYS,
    MIN_IS_BARS,
)
from app.engine.phases import IS_FRACTION
from app.settings_store import load_settings

router = APIRouter()

# Long enough for a Parquet write on a worker thread to finish unwinding.
SYNC_STOP_TIMEOUT_SECONDS = 15


class BuiltInDataOnlyRequest(BaseModel):
    enabled: bool


@router.get("/data/config")
def get_data_config() -> dict:
    config = get_config()
    return {
        "built_in_data_only": config.built_in_data_only,
        "bundled_symbols": list(BUNDLED_SYMBOLS),
    }


@router.get("/data/instruments")
def get_instruments() -> dict:
    # REQ-SET-1.1 — Settings → Target Markets: full catalog, not just the
    # pre-bundled subset (/data/config's bundled_symbols).
    from app.data.custom_instruments import all_instruments

    instruments = all_instruments(get_config().data_dir)
    return {
        "symbols": list(instruments.keys()),
        "timeframes": list(TIMEFRAMES),
        "instruments": [
            {
                "symbol": row.symbol,
                "category": row.category,
                "custom": row.symbol not in INSTRUMENTS,
            }
            for row in instruments.values()
        ],
    }


class CustomInstrumentProbe(BaseModel):
    dukascopy_code: str
    point_divisor: int


class CustomInstrumentRequest(BaseModel):
    symbol: str
    dukascopy_code: str
    point_divisor: int
    category: str = "fx"


@router.post("/markets/custom/probe")
async def probe_custom_instrument(body: CustomInstrumentProbe) -> dict:
    """Ask the feed what this code and divisor actually give.

    The app shows the resulting price before anything is saved, so the
    user confirms a number they recognise instead of a divisor they had
    to guess. A wrong divisor does not fail - it silently backtests at the
    wrong order of magnitude.
    """
    from app.data.custom_instruments import InstrumentRefused, probe

    try:
        result = await probe(body.dukascopy_code.strip(), body.point_divisor)
    except InstrumentRefused as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "code": result.code,
        "point_divisor": result.point_divisor,
        "ticks": result.ticks,
        "median_price": result.median_price,
        "first_price": result.first_price,
        "last_price": result.last_price,
        "hour": result.hour,
        "usable": result.ticks > 0,
        "detail": result.detail,
    }


@router.post("/markets/custom")
def add_custom_instrument(body: CustomInstrumentRequest) -> dict:
    from app.data.custom_instruments import InstrumentRefused, save

    try:
        instrument = save(
            get_config().data_dir, body.symbol, body.dukascopy_code, body.point_divisor, body.category
        )
    except InstrumentRefused as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "symbol": instrument.symbol,
        "dukascopy_code": instrument.dukascopy_code,
        "point_divisor": instrument.point_divisor,
        "category": instrument.category,
    }


@router.delete("/markets/custom/{symbol}")
def delete_custom_instrument(symbol: str) -> dict:
    from app.data.custom_instruments import remove

    if not remove(get_config().data_dir, symbol.upper()):
        raise HTTPException(status_code=404, detail=f"No custom instrument called {symbol}.")
    return {"removed": symbol.upper()}


@router.post("/data/built-in-only")
def set_data_built_in_only(body: BuiltInDataOnlyRequest) -> dict:
    # REQ-DATA-1.6 — debug toggle, forces the pre-bundled subset regardless
    # of full cache/broker availability.
    set_built_in_data_only(body.enabled)
    return {"built_in_data_only": body.enabled}


class SyncRequest(BaseModel):
    """Which markets to sync. Omitted or null means every enabled one.

    This used to be a bare `symbols: list[str] | None` parameter, which
    FastAPI binds as the *whole* body - so the only accepted shape was a
    raw JSON array. The renderer posts `{}`, as it does to every other
    endpoint here, and got a 422 back: "Sync now" had never once worked.
    The cached history came entirely from the startup sync, which calls
    the runner directly and never goes near this route.
    """

    symbols: list[str] | None = None


@router.post("/data/sync")
async def sync_data(request: SyncRequest | None = None) -> dict:
    """Manual re-sync (Dashboard → Historical Data, Settings → System).
    The startup sync is the normal path — REQ-DATA-1.5 — and this returns
    immediately rather than holding the request open for what can be an
    hours-long first download; poll /data/sync/status for progress."""
    if _sync_state.running:
        raise HTTPException(status_code=409, detail="A data sync is already running")
    symbols = request.symbols if request else None
    settings = load_settings(get_config().settings_path)
    target_symbols = symbols or settings["target_markets"]["enabled_symbols"] or list(BUNDLED_SYMBOLS)
    runner = DeltaSyncRunner(history_years=settings["campaign"]["history_depth_years"], state=_sync_state)
    runtime.sync_task = asyncio.create_task(runner.sync_all(target_symbols))
    return {"started": True, "symbols": target_symbols}


@router.post("/data/sync/stop")
async def stop_sync() -> dict:
    """Stops a running sync, keeping everything already downloaded.

    A multi-year backfill is tens of thousands of hourly files and can
    run overnight; a user who wants to breed in the meantime needs to be
    able to call it off. Nothing is lost: whole blocks are committed as
    they land and the watermarks record how far back the cache reaches,
    so the next sync resumes from there rather than starting again.

    It resumes on the next app launch, because the startup sync runs
    unconditionally (REQ-DATA-1.5).
    """
    if not _sync_state.running or runtime.sync_task is None or runtime.sync_task.done():
        raise HTTPException(status_code=409, detail="No data sync is running")

    runtime.sync_task.cancel()
    # A block being written lives in a worker thread, which cancellation
    # does not interrupt, so give it a moment to unwind rather than
    # reporting a stop that has not happened yet.
    await asyncio.wait({runtime.sync_task}, timeout=SYNC_STOP_TIMEOUT_SECONDS)
    runtime.sync_task = None
    return {"stopped": True, **_sync_state.snapshot()}


@router.get("/data/sync/status")
def get_sync_status() -> dict:
    return _sync_state.snapshot()


@router.get("/data/coverage")
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
