from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Local Vault database — design.md B.6. The Python engine is the sole
# writer; the Electron renderer reads the same file via better-sqlite3
# (see app/src/lib/db.ts) for display only.


class Base(DeclarativeBase):
    pass


def make_engine(vault_db_path: Path):
    return create_engine(f"sqlite:///{vault_db_path}", future=True)


def make_session_factory(vault_db_path: Path):
    engine = make_engine(vault_db_path)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
