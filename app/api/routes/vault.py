"""The Vault: strategies, their analysis, re-checks, exports and deployment."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.campaign_setup import _config_for_strategy, _load_bars, _strategy_higher_timeframe
from app.broker.mt5_install import install_ea
from app.codegen.mql5 import GeneratedEa, UnsupportedGeneError, generate_ea
from app.config import get_config
from app.data.broker_clock import clock_from_settings
from app.data.provider import TIMEFRAME_MINUTES, get_historical_bars
from app.db import make_engine
from app.engine.backtest import simulate_trades
from app.engine.campaign import (
    MIN_IS_BARS,
)
from app.engine.neural_map import MIN_STRATEGIES_FOR_MAP, compute_neural_map
from app.engine.phases import IS_FRACTION, split_is_oos
from app.engine.purge import (
    RECYCLE_BIN_RETENTION_DAYS,
    empty_expired_recycle_bin,
    restore_from_recycle_bin,
    run_purge,
)
from app.engine.reassess import reassess_strategy as reassess_strategy_engine
from app.engine.reassess import refresh_strategy as refresh_strategy_engine
from app.genome.gene_pool import GenePoolConfig
from app.genome.genome import Genome
from app.migrations.runner import export_vault, restore_database
from app.models import Campaign, Strategy
from app.settings_store import load_settings

router = APIRouter()

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
        "max_daily_dd_pct": row.max_daily_dd_pct,
        "archetype": row.archetype,
        "mechanism": _mechanism(row.genome_json),
        "robust_pct": row.robust_pct,
        "robust_low_pct": row.robust_low_pct,
        "robust_high_pct": row.robust_high_pct,
        # REQ-VAULT-1.1 PERSISTENCE — real walk-forward result, not a
        # placeholder: the share of test windows the strategy was
        # profitable in (null when the history was too short to fold).
        "persistence_pct": row.persistence_pct,
        # The final holdout: never seen by the search, never gated on.
        "holdout_return_pct": row.holdout_return_pct,
        "holdout_trades": row.holdout_trades,
        "holdout_profit_factor": row.holdout_profit_factor,
        "holdout_max_dd_pct": row.holdout_max_dd_pct,
        "holdout_start": row.holdout_start,
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


@router.get("/vault")
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


@router.get("/analytics")
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


@router.get("/neural-map")
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


class EmptyVaultRequest(BaseModel):
    """Typed so the confirmation cannot be sent by accident.

    The caller has to say how many strategies it believes it is removing.
    A number that no longer matches means the Vault changed between the
    dialog opening and the button being pressed - a campaign finished, or
    another window did something - and the safest thing is to stop and
    let the user look again.
    """

    expected_count: int


@router.post("/vault/empty")
def empty_vault(request: EmptyVaultRequest) -> dict:
    """Sends every strategy to the Recycle Bin — REQ-VAULT-2.2, in bulk.

    Recoverable for RECYCLE_BIN_RETENTION_DAYS, like every other delete
    in this app. There was no way to do this at all: clearing a Vault
    poisoned by a bad campaign meant deleting rows one at a time, or
    reaching into the database by hand, which is how it had been done
    twice.

    A running campaign is writing into this table, so this refuses while
    one is going, the same way PURGE does (REQ-SET-4.4).
    """
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        if session.scalar(select(Campaign).where(Campaign.status == "running")) is not None:
            raise HTTPException(status_code=409, detail="A campaign is running — stop it first.")

        rows = list(session.scalars(select(Strategy).where(Strategy.deleted_at.is_(None))))
        if len(rows) != request.expected_count:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"The Vault holds {len(rows)} strategies, not the {request.expected_count} this was "
                    "asked to remove. Take another look before emptying it."
                ),
            )

        # Named, because the caller has to tell the user which bots are
        # still attached to a chart: removing a strategy from the Vault
        # does not stop an EA that MetaTrader is already running.
        deployed = [row.id for row in rows if row.deployed]
        now = datetime.now(timezone.utc)
        for row in rows:
            row.deleted_at = now
            row.delete_reason = "Vault emptied by hand"
        session.commit()

    return {"deleted": len(rows), "recoverable": True, "still_deployed": deployed}


@router.delete("/vault/{strategy_id}")
def delete_strategy(strategy_id: str) -> dict:
    """REQ-VAULT-2.2 — DELETE, into the Recycle Bin.

    This used to remove the row outright while PURGE - the automated,
    bulk, criteria-driven one - sent its strategies to a Recycle Bin
    recoverable for thirty days (REQ-SET-4.3). The safety net was on the
    operation nobody performs by accident, and absent from the one that
    is a single click next to "Deploy".
    """
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        row = session.get(Strategy, strategy_id)
        if row is None or row.deleted_at is not None:
            raise HTTPException(status_code=404, detail=f"Unknown strategy: {strategy_id}")
        row.deleted_at = datetime.now(timezone.utc)
        row.delete_reason = "Deleted by hand"
        session.commit()
    return {"deleted": strategy_id, "recoverable": True}


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
            # No spread ceiling: the backtest never skipped an entry for
            # its spread, so an EA that does trades a different strategy.
            # It passed the *cost* setting (10 points by default) as the
            # ceiling, which on gold or a standard FX account refused most
            # entries outright. MaxSpreadPoints stays an EA input for
            # anyone who wants one on purpose.
            risk_pct=settings["risk"]["per_trade_risk_pct"] / 100,
            higher_timeframe=_strategy_higher_timeframe(settings, row),
            broker_clock=clock_from_settings(settings),
        )
    except UnsupportedGeneError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return row, generated


class ExportEaRequest(BaseModel):
    destination_dir: str


@router.post("/vault/{strategy_id}/marketplace-package")
def marketplace_package(strategy_id: str) -> dict:
    """What a Marketplace listing is made of: the strategy's own figures,
    read from the Vault rather than typed into a form, and its EA compiled
    to .ex5 - subscribers only ever receive the compiled file, never the
    source (REQ-MKT-1.3).

    Compiling needs MetaEditor, which MetaTrader 5 installs; without it
    this answers 409 and says so.
    """
    import base64
    import hashlib
    import tempfile

    from app.broker.mt5_install import compile_ea

    row, generated = _generate_ea_for(strategy_id)
    with tempfile.TemporaryDirectory(prefix="qe-publish-") as work:
        source_path = Path(work) / generated.filename
        source_path.write_text(generated.source, encoding="utf-8")
        compiled, log = compile_ea(source_path)
        if not compiled:
            raise HTTPException(
                status_code=409,
                detail="The EA could not be compiled: " + (log.strip().splitlines()[-1] if log.strip() else "MetaEditor not found")
                + ". Publishing needs MetaTrader 5 (with MetaEditor) installed on this machine.",
            )
        content = source_path.with_suffix(".ex5").read_bytes()

    return {
        "strategy_id": row.id,
        "symbol": row.asset,
        "timeframe": row.timeframe,
        "score": row.score,
        "robustness_pct": row.robust_pct,
        "trades": row.trades,
        "return_pct": row.return_pct,
        "max_dd_pct": row.max_dd_pct,
        "grade": row.grade,
        "filename": source_path.with_suffix(".ex5").name,
        "content_base64": base64.b64encode(content).decode(),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


@router.post("/vault/{strategy_id}/export-mql5")
def post_export_mql5(strategy_id: str, body: ExportEaRequest) -> dict:
    """REQ-TIER-1.3 (MQL5 export) — writes the EA source to a folder of the
    user's choosing, for people who compile or review it themselves."""
    _, generated = _generate_ea_for(strategy_id)
    destination_dir = Path(body.destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / generated.filename
    destination.write_text(generated.source, encoding="utf-8")
    return {"exported_path": str(destination), "filename": generated.filename}


@router.post("/deploy/{strategy_id}")
def deploy_strategy(strategy_id: str, acknowledge_overlap: bool = False) -> dict:
    """REQ-VAULT-2.1 — generate the EA from the stored genome, install it
    into every MT5 terminal on this machine and compile it. Tier gating is
    the renderer's job (REQ-TIER-1.5); this endpoint does the work.

    Refuses once, with a 409, if this strategy moves in step with one
    already deployed. Correlation used to be checked only inside a
    portfolio - the one place somebody had already decided to think about
    diversification - so three versions of the same idea could go live
    one after another and the account would carry three times the
    intended risk on a single bet.  is the
    caller saying they know and meant it.
    """
    from app.engine.overlap import check_against_deployed

    engine = make_engine(get_config().vault_db_path)
    if not acknowledge_overlap:
        with Session(engine) as session:
            candidate = session.get(Strategy, strategy_id)
            warning = check_against_deployed(session, candidate) if candidate is not None else None
        if warning is not None:
            raise HTTPException(
                status_code=409,
                detail={"error": "correlated_with_deployed", **warning.to_dict()},
            )

    row, generated = _generate_ea_for(strategy_id)
    result = install_ea(generated.filename, generated.source)
    if not result.installed:
        raise HTTPException(status_code=409, detail=result.detail)

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
            config = _config_for_strategy(settings, strategy)
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


@router.post("/vault/recompute")
def recompute_vault(limit: int = 500) -> dict:
    """Bring every Vault row onto the current scoring basis.

    Scores are not comparable across versions of the formula, and the
    Vault is sorted by score - so a Vault holding rows from two bases is
    a list that cannot be read. Strategies bred before the sample-size
    component, the daily Sharpe or the trade log carry none of them, and
    REFRESH fixes one strategy at a time.

    This is REFRESH over all of them. It re-runs each stored genome
    against today's data; it never re-optimises anything, because
    re-fitting a strategy until it looks good again is how a Vault fills
    with strategies that have only ever been fitted.

    Slow by nature - it is a backtest per strategy - which is why it
    reports what it did rather than pretending to be instant.
    """
    settings = load_settings(get_config().settings_path)
    engine = make_engine(get_config().vault_db_path)
    gene_pool = GenePoolConfig.load(get_config().gene_pool_path)

    updated: list[dict] = []
    skipped: list[dict] = []

    with Session(engine) as session:
        strategies = list(
            session.scalars(
                select(Strategy).where(Strategy.deleted_at.is_(None)).order_by(Strategy.score.desc()).limit(
                    max(1, min(limit, 2000))
                )
            )
        )

        # Bars are loaded once per market rather than once per strategy:
        # a Vault of two hundred EURUSD strategies would otherwise read
        # and resample the same year of history two hundred times.
        bars_by_market: dict[tuple[str, str], object] = {}

        for strategy in strategies:
            market = (strategy.asset, strategy.timeframe)
            try:
                if market not in bars_by_market:
                    bars_by_market[market] = _load_bars(
                        strategy.asset, strategy.timeframe, settings["campaign"]["history_depth_years"]
                    )
                df = bars_by_market[market]
                if df is None or len(df) < int(MIN_IS_BARS / IS_FRACTION):
                    skipped.append({"strategy_id": strategy.id, "reason": "not enough history cached"})
                    continue

                config = _config_for_strategy(settings, strategy)
                before = strategy.score
                result = refresh_strategy_engine(strategy, df, gene_pool, config)
                session.commit()
                updated.append(
                    {
                        "strategy_id": strategy.id,
                        "score_before": before,
                        "score_after": strategy.score,
                        "trades": result.trades,
                    }
                )
            except Exception as error:  # one bad row must not stop the rest
                session.rollback()
                skipped.append({"strategy_id": strategy.id, "reason": str(error)})

    moved = [row for row in updated if row["score_before"] != row["score_after"]]
    return {
        "status": "ok",
        "recomputed": len(updated),
        "skipped": len(skipped),
        "changed": len(moved),
        "detail": (
            f"Recomputed {len(updated)} strategies on the current scoring basis; {len(moved)} changed score."
            + (f" {len(skipped)} could not be re-run." if skipped else "")
        ),
        "changes": sorted(moved, key=lambda row: abs((row["score_after"] or 0) - (row["score_before"] or 0)), reverse=True)[:25],
        "skipped_detail": skipped[:25],
    }


@router.post("/vault/{strategy_id}/refresh")
def refresh_strategy(strategy_id: str) -> dict:
    # REQ-VAULT-2.1 REFRESH — restate the strategy's numbers against the
    # data cached today, without re-running the Phase 4 gates.
    return _revalidate(strategy_id, full=False)


@router.post("/vault/{strategy_id}/reassess")
def reassess_strategy(strategy_id: str) -> dict:
    # REQ-VAULT-2.1 REASSESS — REFRESH plus the robustness and walk-forward
    # gates, re-scored. `still_passes` says whether it would be approved
    # if it were bred today.
    return _revalidate(strategy_id, full=True)


# design.md Part C - tick mode. Bounded on purpose: this downloads raw
# ticks from the feed while a user waits, and an unbounded verification of
# a strategy with 900 trades would sit there for an hour. The most recent
# trades are the ones worth checking anyway.
TICK_VERIFY_MAX_TRADES = 60
TICK_VERIFY_MAX_HOURS = 400


class CompareMt5Request(BaseModel):
    report_path: str


@router.post("/vault/{strategy_id}/compare-mt5")
def compare_with_mt5_report(strategy_id: str, body: CompareMt5Request) -> dict:
    """Trade-by-trade: this strategy's backtest against a Strategy Tester
    report of its exported EA, over the report's own period.

    The backtest runs over the whole cached history (so every indicator
    is warmed up exactly as it is in MetaTrader) and only the trades that
    fall inside the report's span are compared.
    """
    from app.engine.mt5_report import ReportUnreadable, read_report
    from app.engine.trade_parity import compare_trades

    try:
        mt5_trades = read_report(body.report_path)
    except ReportUnreadable as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    settings = load_settings(get_config().settings_path)
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        strategy = session.get(Strategy, strategy_id)
        if strategy is None or strategy.deleted_at is not None:
            raise HTTPException(status_code=404, detail=f"Unknown strategy: {strategy_id}")
        session.expunge(strategy)

    try:
        config = _config_for_strategy(settings, strategy)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    df = get_historical_bars(strategy.asset, strategy.timeframe)
    if df.empty:
        raise HTTPException(status_code=409, detail=f"No {strategy.timeframe} history cached for {strategy.asset}.")

    genome = Genome.from_dict(json.loads(strategy.genome_json))
    result = simulate_trades(
        df, genome, GenePoolConfig(),
        initial_balance=config.initial_balance, risk_pct=config.risk_pct,
        point_size=config.point_size, cost_points=config.cost_points,
        commission_per_lot=config.commission_per_lot, contract_size=config.contract_size,
        lot_step=config.lot_step, min_lot=config.min_lot, max_lot=config.max_lot,
        quote_to_account=config.quote_to_account, swap=config.swap, htf_df=config.htf_df,
    )
    report = compare_trades(
        result.trades, df, mt5_trades,
        bar_minutes=TIMEFRAME_MINUTES.get(strategy.timeframe, 60), point_size=config.point_size,
    )
    reported_symbols = {deal.symbol for trade in mt5_trades for deal in trade.deals if deal.symbol}
    foreign = sorted(sym for sym in reported_symbols if not sym.upper().startswith(strategy.asset.upper()))
    return {
        "status": "ok",
        "strategy_id": strategy_id,
        "report_symbols": sorted(reported_symbols),
        "symbol_warning": (
            f"The report trades {', '.join(foreign)}, not {strategy.asset}." if foreign else None
        ),
        **report.to_dict(),
    }


@router.post("/vault/{strategy_id}/tick-verify")
async def tick_verify_strategy(strategy_id: str, max_trades: int = TICK_VERIFY_MAX_TRADES) -> dict:
    """Re-resolves a strategy's exits against raw ticks.

    The bar backtester cannot see inside a bar, so when one bar spans both
    the stop and the target it assumes the stop came first. This measures
    how often that guess was wrong, and what the stops actually filled at
    - a -1R stop that really filled at -2R is invisible in bar data and is
    the kind of thing that empties an account while the curve looks fine.
    """
    from app.data.ticks import TickStore, concat_hours, hours_covering
    from app.engine.tick_verify import verify_trades

    settings = load_settings(get_config().settings_path)
    engine = make_engine(get_config().vault_db_path)

    with Session(engine) as session:
        strategy = session.get(Strategy, strategy_id)
        if strategy is None or strategy.deleted_at is not None:
            raise HTTPException(status_code=404, detail=f"Unknown strategy: {strategy_id}")
        asset, timeframe = strategy.asset, strategy.timeframe
        genome = Genome.from_dict(json.loads(strategy.genome_json))

    try:
        config = _config_for_strategy(settings, strategy)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    df = _load_bars(asset, timeframe, settings["campaign"]["history_depth_years"])
    if df.empty:
        raise HTTPException(status_code=409, detail=f"No {timeframe} history cached for {asset}.")

    # The out-of-sample slice is the half anyone actually judges a
    # strategy on, so it is the half worth verifying.
    _, oos_df = split_is_oos(df, IS_FRACTION)
    gene_pool = GenePoolConfig.load(get_config().gene_pool_path)
    result = simulate_trades(
        oos_df,
        genome,
        gene_pool,
        initial_balance=config.initial_balance,
        point_size=config.point_size,
        cost_points=config.cost_points,
        commission_per_lot=config.commission_per_lot,
        contract_size=config.contract_size,
        risk_pct=config.risk_pct,
        lot_step=config.lot_step,
        min_lot=config.min_lot,
        max_lot=config.max_lot,
        quote_to_account=config.quote_to_account, swap=config.swap,
        htf_df=config.htf_df,
    )
    trades = result.trades[-max(1, min(max_trades, TICK_VERIFY_MAX_TRADES)):]
    if not trades:
        return {
            "status": "ok", "strategy_id": strategy_id, "trades_checked": 0,
            "detail": "This strategy made no trades in its out-of-sample window.",
        }

    bar_duration = timedelta(minutes=TIMEFRAME_MINUTES.get(timeframe, 60))
    windows = [
        (
            pd.Timestamp(oos_df["timestamp"].iloc[trade.entry_idx]).to_pydatetime(),
            pd.Timestamp(oos_df["timestamp"].iloc[trade.exit_idx]).to_pydatetime() + bar_duration,
        )
        for trade in trades
    ]
    hours = hours_covering(windows)
    if len(hours) > TICK_VERIFY_MAX_HOURS:
        raise HTTPException(
            status_code=409,
            detail=(
                f"These {len(trades)} trades span {len(hours)} hours of ticks, over the "
                f"{TICK_VERIFY_MAX_HOURS}-hour limit. Verify fewer trades at a time."
            ),
        )

    store = TickStore(get_config())
    try:
        fetched = await store.ensure(asset, hours)
    except Exception as exc:  # the feed is external and throttles hard
        raise HTTPException(status_code=502, detail=f"Could not fetch ticks: {exc}") from exc
    finally:
        await store.aclose()

    ticks = concat_hours(fetched)
    verification = verify_trades(
        trades, oos_df, ticks, bar_duration=bar_duration,
        cost_points=config.cost_points, point_size=config.point_size,
    )

    return {
        "status": "ok",
        "strategy_id": strategy_id,
        "trades_checked": verification.trades_checked,
        "ambiguous_trades": verification.ambiguous_trades,
        "changed_trades": verification.changed_trades,
        "agreement_pct": round(verification.agreement_pct, 1),
        "r_bars": round(verification.return_pct_bars, 2),
        "r_ticks": round(verification.return_pct_ticks, 2),
        "ticks_used": verification.ticks_used,
        "hours_fetched": len(hours),
        "detail": verification.detail,
        "sample": [
            {
                "entry_time": row.entry_time.isoformat(),
                "direction": row.direction,
                "bar_exit": row.bar_exit,
                "tick_exit": row.tick_exit,
                "ambiguous": row.ambiguous,
                "r_bar": round(row.r_multiple_bar, 2),
                "r_tick": round(row.r_multiple_tick, 2),
            }
            for row in verification.resolutions
            if row.changed or row.ambiguous
        ][:20],
    }


@router.get("/vault/{strategy_id}/trades")
def get_strategy_trades(strategy_id: str, limit: int = 500) -> dict:
    """The trades behind a Vault row.

    Stored at breeding time rather than re-simulated on demand: the
    aggregates rank a strategy, and these are what let anyone actually
    interrogate one. Strategies bred before this existed have none, and
    say so rather than pretending the strategy made no trades.
    """
    from app.models import StrategyTrade

    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        strategy = session.get(Strategy, strategy_id)
        if strategy is None or strategy.deleted_at is not None:
            raise HTTPException(status_code=404, detail=f"Unknown strategy: {strategy_id}")
        rows = list(
            session.scalars(
                select(StrategyTrade)
                .where(StrategyTrade.strategy_id == strategy_id)
                .order_by(StrategyTrade.sequence)
                .limit(max(1, min(limit, 2000)))
            )
        )
        return {
            "strategy_id": strategy_id,
            "trades": [
                {
                    "sequence": row.sequence,
                    "opened_at": row.opened_at.isoformat() if row.opened_at else None,
                    "closed_at": row.closed_at.isoformat() if row.closed_at else None,
                    "direction": row.direction,
                    "entry_price": row.entry_price,
                    "exit_price": row.exit_price,
                    "stop_price": row.stop_price,
                    "target_price": row.target_price,
                    "r_multiple": round(row.r_multiple, 3),
                    "pnl": round(row.pnl, 2),
                    "exit_reason": row.exit_reason,
                }
                for row in rows
            ],
            "detail": (
                "" if rows else
                "This strategy was bred before its trades were kept. REFRESH records them."
            ),
        }


@router.post("/vault/{strategy_id}/parity-test")
def parity_test_strategy(strategy_id: str) -> dict:
    """Is the EA executing what was approved? - guide_parity_test.md.

    Narrower than a health check on purpose. A decayed edge is a market
    problem and the answer is to retire the strategy; an EA on the wrong
    chart, on the wrong timeframe, or with algo trading switched off is an
    operator problem and the strategy was never wrong. Both look like
    disappointing live results, and the fixes are opposites.
    """
    from app.engine.parity import check_parity

    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        strategy = session.get(Strategy, strategy_id)
        if strategy is None or strategy.deleted_at is not None:
            raise HTTPException(status_code=404, detail=f"Unknown strategy: {strategy_id}")
        return {"status": "ok", **check_parity(session, strategy).to_dict()}


class ExportVaultRequest(BaseModel):
    destination_dir: str


class RestoreVaultRequest(BaseModel):
    backup_path: str


@router.post("/vault/export")
def post_vault_export(body: ExportVaultRequest) -> dict:
    # REQ-SET-5.1 — `destination_dir` comes from the Electron main process's
    # native save/folder dialog; this endpoint just performs the copy.
    try:
        exported_path = export_vault(get_config().vault_db_path, Path(body.destination_dir))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"exported_path": str(exported_path)}


@router.post("/vault/restore")
def post_vault_restore(body: RestoreVaultRequest) -> dict:
    # REQ-SET-5.2 — destructive. The confirmation modal is the caller's
    # (renderer's) responsibility; this endpoint executes unconditionally.
    backup_path = Path(body.backup_path)
    if not backup_path.exists():
        raise HTTPException(status_code=404, detail=f"Backup file not found: {backup_path}")
    restore_database(get_config().vault_db_path, backup_path)
    return {"restored_from": str(backup_path)}


@router.post("/vault/purge")
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


@router.get("/vault/recycle-bin")
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


@router.post("/vault/recycle-bin/{strategy_id}/restore")
def post_recycle_bin_restore(strategy_id: str) -> dict:
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        if not restore_from_recycle_bin(session, strategy_id):
            raise HTTPException(status_code=404, detail=f"Not in the Recycle Bin: {strategy_id}")
    return {"restored": strategy_id}
