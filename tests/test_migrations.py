import sqlite3
from pathlib import Path

import pytest

from app.migrations.definitions import CURRENT_SCHEMA_VERSION, MIGRATIONS
from app.migrations.runner import (
    MigrationError,
    backup_database,
    export_vault,
    get_schema_version,
    restore_database,
    run_migrations,
)


def test_get_schema_version_zero_for_nonexistent_db(tmp_path):
    assert get_schema_version(tmp_path / "does_not_exist.db") == 0


def test_run_migrations_creates_all_tables(tmp_path):
    db_path = tmp_path / "vault.db"
    applied = run_migrations(db_path)

    assert applied == [m.version for m in MIGRATIONS]
    conn = sqlite3.connect(db_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    for expected in ("strategies", "portfolios", "portfolio_strategies", "campaigns", "schema_migrations"):
        assert expected in tables


def test_run_migrations_is_idempotent(tmp_path):
    db_path = tmp_path / "vault.db"
    run_migrations(db_path)
    second_run = run_migrations(db_path)
    assert second_run == []
    assert get_schema_version(db_path) == CURRENT_SCHEMA_VERSION


def test_run_migrations_no_backup_on_fresh_database(tmp_path):
    db_path = tmp_path / "vault.db"
    run_migrations(db_path, auto_backup=True)
    # A fresh (previously nonexistent) database has nothing to back up.
    backups = list(tmp_path.glob("*.backup-*"))
    assert backups == []


def test_run_migrations_backs_up_existing_database_before_upgrading(tmp_path, monkeypatch):
    db_path = tmp_path / "vault.db"
    run_migrations(db_path)  # v1 already applied

    # Simulate a future migration being added.
    from app.migrations import definitions

    next_version = CURRENT_SCHEMA_VERSION + 1
    fake_migration = definitions.Migration(version=next_version, description="test migration", statements=("CREATE TABLE IF NOT EXISTS t2 (id INTEGER)",))
    monkeypatch.setattr(definitions, "MIGRATIONS", MIGRATIONS + (fake_migration,))
    monkeypatch.setattr("app.migrations.runner.MIGRATIONS", MIGRATIONS + (fake_migration,))

    applied = run_migrations(db_path, auto_backup=True)

    assert applied == [next_version]
    backups = list(tmp_path.glob("vault.backup-*.db"))
    assert len(backups) == 1


def test_run_migrations_rolls_back_and_restores_on_failure(tmp_path, monkeypatch):
    db_path = tmp_path / "vault.db"
    run_migrations(db_path)  # establish v1, valid baseline

    original_bytes = db_path.read_bytes()

    from app.migrations import definitions

    broken_migration = definitions.Migration(version=CURRENT_SCHEMA_VERSION + 1, description="broken", statements=("THIS IS NOT VALID SQL",))
    monkeypatch.setattr("app.migrations.runner.MIGRATIONS", MIGRATIONS + (broken_migration,))

    with pytest.raises(MigrationError):
        run_migrations(db_path, auto_backup=True)

    # The database file must be back to its exact pre-migration state.
    assert db_path.read_bytes() == original_bytes
    assert get_schema_version(db_path) == CURRENT_SCHEMA_VERSION
    # No stray backup files left behind masquerading as the live db.
    assert db_path.exists()


def test_run_migrations_without_auto_backup_still_rolls_back_transaction(tmp_path):
    db_path = tmp_path / "vault.db"
    run_migrations(db_path)

    from app.migrations import definitions
    import app.migrations.runner as runner_module

    broken_migration = definitions.Migration(version=CURRENT_SCHEMA_VERSION + 1, description="broken", statements=("THIS IS NOT VALID SQL",))
    original = runner_module.MIGRATIONS
    runner_module.MIGRATIONS = MIGRATIONS + (broken_migration,)
    try:
        with pytest.raises(MigrationError):
            run_migrations(db_path, auto_backup=False)
    finally:
        runner_module.MIGRATIONS = original

    # Without a backup, the transaction rollback alone must still leave
    # schema_migrations at the last successfully committed version.
    assert get_schema_version(db_path) == CURRENT_SCHEMA_VERSION


def test_backup_database_returns_none_for_nonexistent_file(tmp_path):
    assert backup_database(tmp_path / "nope.db") is None


def test_backup_and_restore_round_trip(tmp_path):
    db_path = tmp_path / "vault.db"
    db_path.write_text("original content")

    backup_path = backup_database(db_path)
    assert backup_path is not None
    assert backup_path.exists()

    db_path.write_text("corrupted content")
    restore_database(db_path, backup_path)

    assert db_path.read_text() == "original content"


def test_export_vault_creates_timestamped_copy(tmp_path):
    db_path = tmp_path / "vault.db"
    db_path.write_text("vault data")
    destination = tmp_path / "exports"

    exported_path = export_vault(db_path, destination)

    assert exported_path.exists()
    assert exported_path.parent == destination
    assert exported_path.name.startswith("vault-export-")
    assert exported_path.read_text() == "vault data"


def test_export_vault_raises_when_no_database_exists(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_vault(tmp_path / "missing.db", tmp_path / "exports")
