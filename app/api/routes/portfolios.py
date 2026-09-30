"""Portfolios, their health and the True Sim."""


import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.campaign_setup import _config_for_strategy, _load_bars, build_campaign_config
from app.api.routes.vault import _serialize_strategy
from app.config import get_config
from app.db import make_engine
from app.engine.phases import IS_FRACTION, split_is_oos
from app.engine.portfolio import compute_health
from app.engine.true_sim import run_true_sim
from app.genome.gene_pool import GenePoolConfig
from app.models import Portfolio, PortfolioStrategy, Strategy
from app.settings_store import load_settings

router = APIRouter()

@router.get("/portfolios")
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


@router.post("/portfolios")
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


@router.get("/portfolios/{portfolio_id}")
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


@router.get("/portfolios/{portfolio_id}/health")
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


@router.post("/portfolios/{portfolio_id}/true-sim")
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
            # Out-of-sample only. TRUE SIM exists to be read against the
            # HEALTH tab's combined curve - its own docstring calls the
            # gap between them "exactly what the guardrails cost" - and
            # that curve is built from stored OOS equity. Replaying the
            # full history here put the two numbers on different periods,
            # and the extra 70% is the stretch every member was fitted
            # on, so TRUE SIM came out flattered by the comparison it
            # exists to lose.
            bars_by_market[(symbol, timeframe)] = split_is_oos(bars, IS_FRACTION)[1]
    if not bars_by_market:
        raise HTTPException(status_code=409, detail="No cached history for these members' markets.")

    # One account (balance, risk per trade) shared by every member, and
    # each member re-simulated on its own market: its point size, spread,
    # contract size, currency conversion, swap and HIGHER TF context - the
    # same config REASSESS and the EA export use for it.
    config = build_campaign_config(settings, members[0].asset, members[0].timeframe)
    member_configs = {member.id: _config_for_strategy(settings, member) for member in members}
    result = run_true_sim(
        members,
        bars_by_market,
        GenePoolConfig.load(get_config().gene_pool_path),
        config,
        sizing=body.sizing,
        max_aggregate_risk_pct=body.max_aggregate_risk_pct,
        member_configs=member_configs,
    )
    return result.to_dict()


@router.delete("/portfolios/{portfolio_id}")
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
        # Flush the memberships before removing what they point at. These
        # models carry no ORM relationship, so SQLAlchemy has no dependency
        # to sort the DELETEs by and issues them in mapper order - which
        # puts the portfolio first. Silent while SQLite was ignoring the
        # foreign keys it had been given; an error the moment it stopped.
        session.flush()
        session.delete(portfolio)
        session.commit()
    return {"deleted": portfolio_id}


@router.delete("/portfolios/{portfolio_id}/members/{strategy_id}")
def remove_portfolio_member(portfolio_id: str, strategy_id: str) -> dict:
    engine = make_engine(get_config().vault_db_path)
    with Session(engine) as session:
        membership = session.get(PortfolioStrategy, {"portfolio_id": portfolio_id, "strategy_id": strategy_id})
        if membership is None:
            raise HTTPException(status_code=404, detail="That strategy is not in this portfolio")
        session.delete(membership)
        session.commit()
    return {"portfolio_id": portfolio_id, "strategy_id": strategy_id, "removed": True}


@router.post("/vault/{strategy_id}/add-to-portfolio/{portfolio_id}")
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
