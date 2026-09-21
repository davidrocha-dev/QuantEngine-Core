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
    Migration(
        version=3,
        description="Walk-forward validation: persistence and per-fold results (REQ-VAULT-1.1 PERSISTENCE)",
        statements=(
            "ALTER TABLE strategies ADD COLUMN persistence_pct REAL",
            "ALTER TABLE strategies ADD COLUMN walk_forward TEXT",
        ),
    ),
    Migration(
        version=4,
        description="Live trades reported by the QuantEngine Pilot EA (REQ-PORT-4)",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS live_trades (
                deal_ticket TEXT PRIMARY KEY,
                strategy_id TEXT,
                magic INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                closed_at DATETIME NOT NULL,
                volume REAL,
                profit REAL NOT NULL,
                exit_price REAL,
                imported_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_live_trades_strategy ON live_trades (strategy_id, closed_at)",
        ),
    ),
    Migration(
        version=5,
        description="Which broker a live trade came from, for multi-broker health checks (REQ-AUTO-3)",
        statements=("ALTER TABLE live_trades ADD COLUMN broker TEXT",),
    ),
    Migration(
        version=6,
        description="Incubation stage per deployed strategy (REQ-PORT-3)",
        statements=("ALTER TABLE strategies ADD COLUMN incubation_stage TEXT",),
    ),
    Migration(
        version=7,
        description="Backtest win rate, so a health check can compare magnitudes and not only signs (REQ-AUTO-2)",
        statements=("ALTER TABLE strategies ADD COLUMN win_rate_pct REAL",),
    ),
)

CURRENT_SCHEMA_VERSION = max(m.version for m in MIGRATIONS)
