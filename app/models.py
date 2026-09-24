from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base

# Schema per design.md B.6. Migrations are versioned scripts tracked in
# `schema_migrations`, applied transactionally with automatic pre-migration
# backup — see REQ-SET-6 / tasks.md 3.2 (not implemented yet, Phase 3).


class Strategy(Base):
    __tablename__ = "strategies"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    data_source: Mapped[str] = mapped_column(String, nullable=False)
    asset: Mapped[str] = mapped_column(String, nullable=False)
    timeframe: Mapped[str] = mapped_column(String, nullable=False)
    trades: Mapped[int | None] = mapped_column(Integer)
    return_pct: Mapped[float | None]
    cagr_pct: Mapped[float | None]
    max_dd_pct: Mapped[float | None]
    # Worst single day, peak-to-trough against that day's opening equity.
    # A different rule from max_dd_pct and the one a prop firm enforces
    # most aggressively.
    max_daily_dd_pct: Mapped[float | None]
    archetype: Mapped[str | None] = mapped_column(String)
    robust_pct: Mapped[float | None]
    # The 10th/90th percentile of the sampled noise degradations, expressed
    # as robustness percentages. robust_pct on its own is a point estimate
    # of a distribution that is measurably bimodal — "72%" and "72% (25–73)"
    # are very different claims about the same strategy.
    robust_low_pct: Mapped[float | None]
    robust_high_pct: Mapped[float | None]
    icir: Mapped[float | None]
    # REQ-AUTO-2 - the out-of-sample win rate, kept so a live health check
    # has something to compare against that needs no currency conversion.
    # Live profit is money and the Vault's return is a percentage; win rate
    # is a ratio on both sides, which is what makes it comparable at all.
    win_rate_pct: Mapped[float | None]
    avg_hold_hours: Mapped[float | None]
    score: Mapped[int | None] = mapped_column(Integer)
    grade: Mapped[str | None] = mapped_column(String)
    genome_json: Mapped[str] = mapped_column(Text, nullable=False)
    equity_is: Mapped[str | None] = mapped_column(Text)
    equity_oos: Mapped[str | None] = mapped_column(Text)
    deployed: Mapped[bool] = mapped_column(Boolean, default=False)
    active_mt5: Mapped[bool] = mapped_column(Boolean, default=False)
    # REQ-SET-4.3 — purged strategies go to a Recycle Bin recoverable for 30
    # days, so a purge is never an irreversible delete.
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime)
    delete_reason: Mapped[str | None] = mapped_column(Text)
    # REQ-VAULT-1.1 PERSISTENCE — share of walk-forward folds the strategy
    # was profitable in, with the per-fold detail behind it.
    persistence_pct: Mapped[float | None]
    walk_forward: Mapped[str | None] = mapped_column(Text)  # JSON
    # REQ-PORT-3 — the stage this bot has actually been promoted to. Null
    # means it has never been promoted, i.e. INCUBATION. Kept separate from
    # what its live record would *allow*, so a promotion is a decision that
    # sticks rather than a number recomputed on every page load.
    incubation_stage: Mapped[str | None] = mapped_column(String)


class StrategyTrade(Base):
    """One backtested trade behind a Vault row.

    The Vault stored aggregates and an equity curve, which is enough to
    rank a strategy and not enough to interrogate one. Every question
    worth asking after the fact - which hours it actually makes money in,
    whether it survives without its three best trades, what the exits
    really looked like - needed a full re-simulation, and the tick
    verification had to re-run the whole out-of-sample window before it
    could check a single fill.

    Timestamps rather than bar indices, because an index only means
    something against the exact bar series it came from, and that series
    is a cache that grows every day.
    """

    __tablename__ = "strategy_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_id: Mapped[str] = mapped_column(ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    opened_at: Mapped[datetime | None] = mapped_column(DateTime)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime)
    direction: Mapped[int] = mapped_column(Integer, nullable=False)  # 1 long, -1 short
    entry_price: Mapped[float] = mapped_column(nullable=False)
    exit_price: Mapped[float] = mapped_column(nullable=False)
    stop_price: Mapped[float] = mapped_column(nullable=False)
    target_price: Mapped[float] = mapped_column(nullable=False)
    r_multiple: Mapped[float] = mapped_column(nullable=False)
    pnl: Mapped[float] = mapped_column(nullable=False)
    exit_reason: Mapped[str] = mapped_column(String, nullable=False)


class Portfolio(Base):
    __tablename__ = "portfolios"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    description: Mapped[str | None] = mapped_column(Text)


class PortfolioStrategy(Base):
    __tablename__ = "portfolio_strategies"

    portfolio_id: Mapped[str] = mapped_column(ForeignKey("portfolios.id"), primary_key=True)
    strategy_id: Mapped[str] = mapped_column(ForeignKey("strategies.id"), primary_key=True)


class Campaign(Base):
    __tablename__ = "campaigns"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    asset: Mapped[str | None] = mapped_column(String)
    timeframe: Mapped[str | None] = mapped_column(String)
    phase: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str | None] = mapped_column(String)  # running/completed/cancelled/crashed
    survivors_per_phase: Mapped[str | None] = mapped_column(Text)  # JSON
    checkpoint_generation: Mapped[int | None] = mapped_column(Integer)
    checkpoint_population: Mapped[str | None] = mapped_column(Text)  # JSON
    last_checkpoint_at: Mapped[datetime | None] = mapped_column(DateTime)


class LiveTrade(Base):
    """One closed deal reported by the Pilot EA — REQ-PORT-4.

    `strategy_id` is resolved from the deal's magic number, which the
    generated EA derives from the strategy's Hash ID. A deal whose magic
    matches no Vault strategy is still stored, with a null strategy_id:
    dropping it would hide a bot running from a strategy that was deleted.
    """

    __tablename__ = "live_trades"

    deal_ticket: Mapped[str] = mapped_column(String, primary_key=True)
    strategy_id: Mapped[str | None] = mapped_column(String)
    magic: Mapped[int] = mapped_column(Integer, nullable=False)
    symbol: Mapped[str] = mapped_column(String, nullable=False)
    closed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    volume: Mapped[float | None]
    profit: Mapped[float] = mapped_column(nullable=False)
    exit_price: Mapped[float | None]
    # The account server the deal came from. REQ-AUTO-3 turns on this one
    # field: the same strategy losing everywhere is alpha decay, losing on
    # one broker is that broker's problem.
    broker: Mapped[str | None] = mapped_column(String)
    imported_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class SchemaMigration(Base):
    __tablename__ = "schema_migrations"

    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    applied_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    description: Mapped[str | None] = mapped_column(Text)
