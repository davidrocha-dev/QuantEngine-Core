"""Versioned schema migrations — design.md B.6.

Hand-written DDL kept in lockstep with `design.md` B.6 (the authoritative
schema text) rather than reflected off the SQLAlchemy models, so the
migration history stays an explicit, reviewable record of schema changes.
`app/models.py` (the ORM layer used for queries/writes) must match what
these migrations produce.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Migration:
    version: int
    description: str
    statements: tuple[str, ...]


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        description="Initial schema: strategies, portfolios, portfolio_strategies, campaigns, schema_migrations",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS strategies (
                id TEXT PRIMARY KEY, created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                data_source TEXT NOT NULL, asset TEXT NOT NULL, timeframe TEXT NOT NULL,
                trades INTEGER, return_pct REAL, cagr_pct REAL, max_dd_pct REAL,
                archetype TEXT, robust_pct REAL, icir REAL, avg_hold_hours REAL,
                score INTEGER, grade TEXT, genome_json TEXT NOT NULL,
                equity_is TEXT, equity_oos TEXT,
                deployed BOOLEAN DEFAULT FALSE, active_mt5 BOOLEAN DEFAULT FALSE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS portfolios (
                id TEXT PRIMARY KEY, name TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP, description TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS portfolio_strategies (
                portfolio_id TEXT REFERENCES portfolios(id),
                strategy_id  TEXT REFERENCES strategies(id),
                PRIMARY KEY (portfolio_id, strategy_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS campaigns (
                id TEXT PRIMARY KEY, started_at DATETIME, finished_at DATETIME,
                asset TEXT, timeframe TEXT, phase INTEGER,
                status TEXT,
                survivors_per_phase TEXT,
                checkpoint_generation INTEGER,
                checkpoint_population TEXT,
                last_checkpoint_at DATETIME
            )
            """,
        ),
    ),
    Migration(
        version=2,
        description="Recycle Bin: soft-delete column for purged strategies (REQ-SET-4.3)",
        statements=(
            "ALTER TABLE strategies ADD COLUMN deleted_at DATETIME",
            "ALTER TABLE strategies ADD COLUMN delete_reason TEXT",
        ),
    ),
)

CURRENT_SCHEMA_VERSION = max(m.version for m in MIGRATIONS)
