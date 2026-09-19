from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.db import Base, make_engine
from app.engine.purge import (
    RECYCLE_BIN_RETENTION_DAYS,
    empty_expired_recycle_bin,
    restore_from_recycle_bin,
    run_purge,
)
from app.models import Campaign, Portfolio, PortfolioStrategy, Strategy
from app.settings_store import DEFAULT_APP_SETTINGS


@pytest.fixture
def session(tmp_path: Path) -> Session:
    engine = make_engine(tmp_path / "vault_test.db")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def make_strategy(session: Session, strategy_id: str, **overrides) -> Strategy:
    defaults = dict(
        id=strategy_id, data_source="dukascopy", asset="EURUSD", timeframe="H1",
        trades=200, return_pct=25.0, max_dd_pct=10.0, robust_pct=90.0, icir=1.2,
        avg_hold_hours=5.0, score=70, grade="GOOD", genome_json="{}",
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    row = Strategy(**defaults)
    session.add(row)
    session.commit()
    return row


def settings_with(**criteria) -> dict:
    settings = {
        "vault_maintenance": {
            "protect_portfolio": True,
            "purge_criteria": {
                key: dict(value) for key, value in DEFAULT_APP_SETTINGS["vault_maintenance"]["purge_criteria"].items()
            },
        }
    }
    for key, threshold in criteria.items():
        settings["vault_maintenance"]["purge_criteria"][key] = {"enabled": True, "threshold": threshold}
    return settings


def test_purge_soft_deletes_strategies_failing_a_criterion(session):
    make_strategy(session, "#WEAK", score=30)
    make_strategy(session, "#GOOD", score=80)

    result = run_purge(session, settings_with(min_score=57.0))

    assert result.purged == 1
    assert session.get(Strategy, "#WEAK").deleted_at is not None
    assert session.get(Strategy, "#WEAK").delete_reason == "score below minimum"
    assert session.get(Strategy, "#GOOD").deleted_at is None


def test_purge_never_hard_deletes(session):
    make_strategy(session, "#WEAK", score=10)
    run_purge(session, settings_with(min_score=57.0))
    # REQ-SET-4.3 — still recoverable, not gone.
    assert session.get(Strategy, "#WEAK") is not None


def test_protect_portfolio_shields_members(session):
    make_strategy(session, "#WEAK", score=10)
    session.add(Portfolio(id="p1", name="Core"))
    session.add(PortfolioStrategy(portfolio_id="p1", strategy_id="#WEAK"))
    session.commit()

    result = run_purge(session, settings_with(min_score=57.0))

    assert result.purged == 0
    assert result.protected == 1
    assert session.get(Strategy, "#WEAK").deleted_at is None


def test_purge_is_deferred_while_a_campaign_runs(session):
    make_strategy(session, "#WEAK", score=10)
    session.add(Campaign(id="c1", status="running"))
    session.commit()

    result = run_purge(session, settings_with(min_score=57.0))

    assert result.purged == 0
    assert result.skipped_reason == "a campaign is running"
    assert session.get(Strategy, "#WEAK").deleted_at is None


def test_drawdown_greater_than_return_criterion(session):
    make_strategy(session, "#BAD", return_pct=5.0, max_dd_pct=20.0)
    settings = settings_with()
    settings["vault_maintenance"]["purge_criteria"]["dd_gt_return"] = {"enabled": True, "threshold": None}

    assert run_purge(session, settings).purged == 1


def test_built_in_data_bred_criterion_uses_data_source(session):
    make_strategy(session, "#BUNDLED", data_source="bundled")
    make_strategy(session, "#REAL", data_source="dukascopy")
    settings = settings_with()
    settings["vault_maintenance"]["purge_criteria"]["built_in_data_bred"] = {"enabled": True, "threshold": None}

    result = run_purge(session, settings)

    assert result.purged == 1
    assert session.get(Strategy, "#REAL").deleted_at is None


def test_nothing_is_purged_when_no_criterion_is_enabled(session):
    make_strategy(session, "#ANY", score=1, return_pct=-50.0, max_dd_pct=99.0)
    assert run_purge(session, settings_with()).purged == 0


def test_restore_brings_a_strategy_back(session):
    make_strategy(session, "#WEAK", score=10)
    run_purge(session, settings_with(min_score=57.0))

    assert restore_from_recycle_bin(session, "#WEAK") is True
    assert session.get(Strategy, "#WEAK").deleted_at is None
    assert restore_from_recycle_bin(session, "#WEAK") is False  # no longer in the bin


def test_recycle_bin_expires_after_the_retention_window(session):
    old = datetime.now(timezone.utc) - timedelta(days=RECYCLE_BIN_RETENTION_DAYS + 1)
    make_strategy(session, "#OLD", deleted_at=old)
    make_strategy(session, "#RECENT", deleted_at=datetime.now(timezone.utc))

    removed = empty_expired_recycle_bin(session)

    assert removed == 1
    assert session.get(Strategy, "#OLD") is None
    assert session.get(Strategy, "#RECENT") is not None
