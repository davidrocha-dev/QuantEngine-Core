"""Migration runner — REQ-SET-6.

Safety model: an automatic timestamped backup of `vault.db` is taken
before any migration runs (REQ-SET-6.1, REQ-SET-5.3); migrations apply
inside a single SQLite transaction (REQ-SET-6.2); on any failure the
transaction is rolled back AND the database file itself is restored from
the pre-migration backup, so a partially-applied DDL statement (SQLite
does not roll back all DDL uniformly) can never leave the file on disk in
an intermediate state (REQ-SET-6.3).
"""

from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.migrations.definitions import MIGRATIONS

_SCHEMA_MIGRATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    description TEXT
)
"""


class MigrationError(RuntimeError):
    """Raised when a migration fails; the database has already been
    restored to its pre-migration state by the time this is raised."""


def backup_database(db_path: Path) -> Path | None:
    """REQ-SET-5.1/6.1 — timestamped copy of vault.db. Returns None (no
    backup needed or possible) if the database doesn't exist yet."""
    if not db_path.exists():
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = db_path.with_name(f"{db_path.stem}.backup-{timestamp}{db_path.suffix}")
    shutil.copy2(db_path, backup_path)
    return backup_path


def export_vault(db_path: Path, destination_dir: Path) -> Path:
    """REQ-SET-5.1 — EXPORT VAULT: timestamped copy of vault.db at a
    user-chosen location. `destination_dir` is resolved by the caller's
    native save/folder dialog (Electron main process)."""
    if not db_path.exists():
        raise FileNotFoundError(f"No vault database found at {db_path}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination_path = destination_dir / f"vault-export-{timestamp}.db"
    shutil.copy2(db_path, destination_path)
    return destination_path


def restore_database(db_path: Path, backup_path: Path) -> None:
    """REQ-SET-5.2 — destructive: overwrites the current vault.db. Callers
    (Electron main / Settings UI) must obtain explicit user confirmation
    before invoking this outside of the automatic rollback-on-failure path."""
    shutil.copy2(backup_path, db_path)


def get_schema_version(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_SCHEMA_MIGRATIONS_TABLE_SQL)
        row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
        return row[0] or 0
    finally:
        conn.close()


def run_migrations(db_path: Path, auto_backup: bool = True) -> list[int]:
    """Applies every migration newer than the current schema version.
    Returns the list of versions applied (empty if already up to date)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    current_version = get_schema_version(db_path)

    pending = [m for m in MIGRATIONS if m.version > current_version]
    if not pending:
        return []

    backup_path = backup_database(db_path) if auto_backup else None

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_SCHEMA_MIGRATIONS_TABLE_SQL)
        conn.execute("BEGIN")
        applied = []
        for migration in pending:
            for statement in migration.statements:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_migrations (version, description) VALUES (?, ?)",
                (migration.version, migration.description),
            )
            applied.append(migration.version)
        conn.commit()
    except Exception as exc:
        conn.rollback()
        conn.close()
        if backup_path is not None:
            restore_database(db_path, backup_path)
        raise MigrationError(
            f"Migration failed and the database was rolled back to its pre-migration state: {exc}"
        ) from exc
    else:
        conn.close()
        return applied
