from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Local Vault database — design.md B.6. The Python engine is the sole
# writer; the Electron renderer reads the same file via better-sqlite3
# (see app/src/lib/db.ts) for display only.


class Base(DeclarativeBase):
    pass


def make_engine(vault_db_path: Path):
    engine = create_engine(f"sqlite:///{vault_db_path}", future=True)

    @event.listens_for(engine, "connect")
    def _enforce_foreign_keys(dbapi_connection, _connection_record) -> None:
        """SQLite ignores every FOREIGN KEY it is given unless asked not to.

        The pragma defaults to off, per connection, so the ON DELETE
        CASCADE declared on strategy_trades and portfolio_strategies was
        decoration. Demonstrated: a strategy whose recycle-bin window
        expires is deleted by empty_expired_recycle_bin and its 25 trade
        rows stay behind, unreachable and permanent. Nothing had noticed
        because nothing had yet been hard-deleted - the bin holds rows
        for thirty days.
        """
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def make_session_factory(vault_db_path: Path):
    engine = make_engine(vault_db_path)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
