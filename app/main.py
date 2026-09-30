"""The QuantEngine engine process - FastAPI app, bound to 127.0.0.1:7831.

Startup and wiring only. The endpoints live in app/api/routes/, the state
they share (the running campaign, the sync task, the MetaTrader bridge,
the event socket's clients) in app/api/runtime.py, and the translation
from Settings to a campaign's configuration in app/api/campaign_setup.py
and app/api/costs.py.

The names tests and older callers import from here are re-exported below.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app import licensing
from app.api import runtime
from app.api.security import LocalAccessMiddleware

# Re-exported: tests and older callers import these from app.main.
from app.api.campaign_setup import (  # noqa: F401
    HIGHER_TF_BY_TIMEFRAME,
    POWER_PRESETS,
    _config_for_strategy,
    _gene_pool_for_campaign,
    _higher_timeframe,
    _learned_gene_weights,
    _load_bars,
    _strategy_higher_timeframe,
    build_campaign_config,
)
from app.api.costs import (  # noqa: F401
    _MT5_SWAP_CURRENCY_DEPOSIT,
    _MT5_SWAP_POINTS,
    CONTRACT_SIZE_BY_CATEGORY,
    SpreadUnavailable,
    _manual_spread_points,
    _spread_points,
    _swap_spec,
)
from app.api.routes import campaign as campaign_routes
from app.api.routes import data as data_routes
from app.api.routes import live as live_routes
from app.api.routes import marketplace as marketplace_routes
from app.api.routes import portfolios as portfolio_routes
from app.api.routes import system as system_routes
from app.api.routes import vault as vault_routes
from app.api.routes.campaign import (  # noqa: F401
    _run_campaign_background,
    _serialize_campaign,
    discard_crashed_campaign,
    get_campaign_estimate,
    list_crashed_campaigns,
    pause_campaign,
    resume_campaign,
    skip_campaign,
    start_campaign,
)
from app.api.routes.data import (  # noqa: F401
    SYNC_STOP_TIMEOUT_SECONDS,
    BuiltInDataOnlyRequest,
    CustomInstrumentProbe,
    CustomInstrumentRequest,
    SyncRequest,
    add_custom_instrument,
    delete_custom_instrument,
    get_data_config,
    get_data_coverage,
    get_instruments,
    get_sync_status,
    probe_custom_instrument,
    set_data_built_in_only,
    stop_sync,
    sync_data,
)
from app.api.routes.live import (  # noqa: F401
    get_incubation,
    get_pilot_status,
    post_confirm_auto_deploy,
    post_graduate,
    post_health_check,
    post_pilot_import,
    post_pilot_install,
    post_run_auto_deploy,
)
from app.api.routes.marketplace import (  # noqa: F401
    MarketplaceInstallRequest,
    post_marketplace_install,
)
from app.api.routes.portfolios import (  # noqa: F401
    CreatePortfolioRequest,
    TrueSimRequest,
    _portfolio_members,
    add_strategy_to_portfolio,
    create_portfolio,
    delete_portfolio,
    get_portfolio,
    get_portfolio_health,
    list_portfolios,
    post_true_sim,
    remove_portfolio_member,
)
from app.api.routes.system import (  # noqa: F401
    AutoBackupRequest,
    GenePoolUpdateRequest,
    TestTelegramRequest,
    _gene_pool_response,
    connect_mt5,
    events,
    get_app_settings,
    get_auto_backup,
    get_gene_pool,
    get_mt5_status,
    get_smart_mode,
    get_status,
    get_system_info,
    patch_app_settings,
    post_smart_mode_rebuild,
    post_smart_mode_reset,
    post_test_telegram,
    set_auto_backup,
    set_gene_pool,
)
from app.api.routes.vault import (  # noqa: F401
    TICK_VERIFY_MAX_HOURS,
    TICK_VERIFY_MAX_TRADES,
    CompareMt5Request,
    EmptyVaultRequest,
    ExportEaRequest,
    ExportVaultRequest,
    RestoreVaultRequest,
    _active_genes,
    _breakdown,
    _generate_ea_for,
    _mechanism,
    _revalidate,
    _serialize_strategy,
    compare_with_mt5_report,
    delete_strategy,
    deploy_strategy,
    empty_vault,
    get_analytics,
    get_neural_map,
    get_recycle_bin,
    get_strategy_trades,
    list_vault,
    parity_test_strategy,
    post_export_mql5,
    post_recycle_bin_restore,
    post_vault_export,
    post_vault_purge,
    post_vault_restore,
    reassess_strategy,
    recompute_vault,
    refresh_strategy,
    tick_verify_strategy,
)
from app.api.runtime import (  # noqa: F401
    _GENERATIONAL_PHASES,
    TRIBE_BY_PHASE,
    _best_fitness,
    _broadcast,
    _campaign_control,
    _campaign_poll_state,
    _emit_event_threadsafe,
    _mt5_bridge,
    _poll_campaigns_loop,
    _sync_state,
    _ws_clients,
)
from app.api.scheduler import (  # noqa: F401
    AUTO_DEPLOY_INTERVAL_HOURS,
    SCHEDULER_TICK_SECONDS,
    _due,
    _scheduler_loop,
    _startup_delta_sync,
)
from app.config import get_config
from app.db import make_engine
from app.engine.campaign import (
    mark_crashed_campaigns,
)
from app.logging_setup import configure_crash_reporting, configure_logging
from app.migrations.runner import MigrationError, run_migrations
from app.settings_store import load_settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.main_loop = asyncio.get_running_loop()
    config = get_config()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    # REQ-NFR-2 — rotating local logs, so "the engine stopped" leaves
    # something diagnosable behind. Crash reporting stays opt-in.
    log_path = configure_logging(config.data_dir)
    licensing.log_startup_state()
    settings = load_settings(config.settings_path)
    configure_crash_reporting(settings["system"]["crash_reporting_enabled"])
    logging.getLogger("quantengine").info("Engine starting — logging to %s", log_path)
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


@app.exception_handler(SpreadUnavailable)
async def _spread_unavailable(_request, exc: SpreadUnavailable):
    # Any endpoint that prices a backtest can land here (REASSESS, True
    # Sim, tick verification) - not only campaign launch - and each one
    # should say why rather than answer 500.
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=409, content={"detail": str(exc)})

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

# Added after CORS so it runs first: every request must come from the
# desktop app's own renderer or shell (app/api/security.py).
app.add_middleware(LocalAccessMiddleware)


# Routers, in the order the endpoints were declared before the split -
# FastAPI matches paths in registration order.
for _router in (
    system_routes.router, campaign_routes.router, vault_routes.router, portfolio_routes.router,
    data_routes.router, live_routes.router, marketplace_routes.router,
):
    app.include_router(_router)
