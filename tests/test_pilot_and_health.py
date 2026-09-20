from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.broker.pilot import import_live_trades
from app.codegen.mql5 import _magic_number
from app.db import Base, make_engine
from app.engine.health import (
    FULL_RISK_TRADES,
    MIN_TRADES_FOR_VERDICT,
    PROVING_TRADES,
    check_strategy,
    ladder,
    run_health_check,
)
from app.models import LiveTrade, Strategy


@pytest.fixture
def session(tmp_path: Path) -> Session:
    engine = make_engine(tmp_path / "vault.db")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def make_strategy(session: Session, strategy_id: str = "#LIVE01", return_pct: float = 20.0, deployed: bool = True) -> Strategy:
    row = Strategy(
        id=strategy_id, data_source="dukascopy", asset="EURUSD", timeframe="M15",
        genome_json="{}", trades=100, return_pct=return_pct, score=70, grade="GOOD", deployed=deployed,
    )
    session.add(row)
    session.commit()
    return row


def write_report(path: Path, rows: list[tuple[str, str, int, float]]) -> Path:
    lines = ["deal_ticket,closed_at,magic,symbol,volume,profit,entry_price,exit_price"]
    for ticket, closed_at, magic, profit in rows:
        lines.append(f"{ticket},{closed_at},{magic},EURUSD,0.10,{profit},0.00000,1.10000")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def add_live_trades(session: Session, strategy_id: str, profits: list[float], start: datetime | None = None) -> None:
    start = start or datetime.now(timezone.utc) - timedelta(days=20)
    for index, profit in enumerate(profits):
        session.add(
            LiveTrade(
                deal_ticket=f"{strategy_id}-{index}", strategy_id=strategy_id, magic=_magic_number(strategy_id),
                symbol="EURUSD", closed_at=start + timedelta(hours=index), volume=0.1, profit=profit,
            )
        )
    session.commit()


def test_import_attributes_deals_to_the_strategy_that_generated_them(session, tmp_path):
    strategy = make_strategy(session)
    magic = _magic_number(strategy.id)
    report = write_report(tmp_path / "live.csv", [("1", "2026.01.05 10:00:00", magic, 12.5)])

    result = import_live_trades(session, report)

    assert result.imported == 1
    trade = session.get(LiveTrade, "1")
    assert trade.strategy_id == strategy.id
    assert trade.profit == 12.5


def test_import_is_idempotent(session, tmp_path):
    strategy = make_strategy(session)
    report = write_report(tmp_path / "live.csv", [("1", "2026.01.05 10:00:00", _magic_number(strategy.id), 5.0)])

    first = import_live_trades(session, report)
    second = import_live_trades(session, report)

    assert first.imported == 1
    assert second.imported == 0
    assert second.skipped_existing == 1


def test_a_deal_from_an_unknown_strategy_is_kept_not_dropped(session, tmp_path):
    # A bot running from a strategy that was deleted is exactly what you
    # want to see, not hide.
    make_strategy(session)
    report = write_report(tmp_path / "live.csv", [("9", "2026.01.05 10:00:00", 123456789, -4.0)])

    result = import_live_trades(session, report)

    assert result.imported == 1
    assert result.unattributed == 1
    assert session.get(LiveTrade, "9").strategy_id is None


def test_a_missing_report_says_what_to_do(session):
    result = import_live_trades(session, Path("nowhere.csv"))
    assert result.imported == 0
    assert "Pilot" in result.detail


def test_a_strategy_with_no_live_trades_reports_no_data(session):
    strategy = make_strategy(session)
    assert check_strategy(session, strategy).status == "no_data"


def test_too_few_live_trades_is_too_early_not_healthy(session):
    strategy = make_strategy(session)
    add_live_trades(session, strategy.id, [5.0] * (MIN_TRADES_FOR_VERDICT - 1))

    verdict = check_strategy(session, strategy)

    assert verdict.status == "too_early"
    assert str(MIN_TRADES_FOR_VERDICT) in verdict.detail


def test_a_profitable_backtest_losing_live_is_degraded(session):
    strategy = make_strategy(session, return_pct=25.0)
    add_live_trades(session, strategy.id, [-3.0] * 12)

    verdict = check_strategy(session, strategy)

    assert verdict.status == "degraded"
    assert verdict.live_profit < 0


def test_a_profitable_live_run_is_healthy(session):
    strategy = make_strategy(session, return_pct=25.0)
    add_live_trades(session, strategy.id, [4.0, -2.0] * 8)

    assert check_strategy(session, strategy).status == "healthy"


def test_run_health_check_summarises_only_deployed_strategies(session):
    deployed = make_strategy(session, "#DEP", deployed=True)
    make_strategy(session, "#NOTDEP", deployed=False)
    add_live_trades(session, deployed.id, [3.0] * 12)

    verdicts = run_health_check(session)

    assert [verdict.strategy_id for verdict in verdicts] == ["#DEP"]


def test_a_fresh_bot_starts_in_incubation_at_a_quarter_risk(session):
    make_strategy(session)
    entry = ladder(session)[0]

    assert entry.stage == "INCUBATION"
    assert entry.risk_share == 0.25


def test_enough_healthy_trades_moves_a_bot_up_the_ladder(session):
    strategy = make_strategy(session)
    add_live_trades(session, strategy.id, [2.0] * PROVING_TRADES)
    assert ladder(session)[0].stage == "PROVING"

    add_live_trades(
        session, strategy.id + "x", [], start=None
    )  # no-op, keeps the helper's ticket namespace distinct
    for index in range(FULL_RISK_TRADES):
        session.add(
            LiveTrade(
                deal_ticket=f"extra-{index}", strategy_id=strategy.id, magic=_magic_number(strategy.id),
                symbol="EURUSD", closed_at=datetime.now(timezone.utc), volume=0.1, profit=1.0,
            )
        )
    session.commit()

    assert ladder(session)[0].stage == "FULL RISK"


def test_an_underwater_bot_is_not_promoted_however_many_trades_it_has(session):
    # REQ-PORT-3.4 — the ladder exists to stop a bad deploy reaching size.
    strategy = make_strategy(session)
    add_live_trades(session, strategy.id, [-2.0] * PROVING_TRADES)

    entry = ladder(session)[0]

    assert entry.can_promote is False
    assert "underwater" in entry.blocked_reason


def add_broker_trades(session: Session, strategy_id: str, broker: str, profits: list[float]) -> None:
    for index, profit in enumerate(profits):
        session.add(
            LiveTrade(
                deal_ticket=f"{strategy_id}-{broker}-{index}", strategy_id=strategy_id,
                magic=_magic_number(strategy_id), symbol="EURUSD",
                closed_at=datetime.now(timezone.utc) - timedelta(hours=index), volume=0.1,
                profit=profit, broker=broker,
            )
        )
    session.commit()


def test_one_broker_alone_cannot_tell_the_two_causes_apart(session):
    from app.engine.health import check_across_brokers

    strategy = make_strategy(session)
    add_broker_trades(session, strategy.id, "BrokerA", [-2.0] * 15)

    verdict = check_across_brokers(session, strategy)

    assert verdict.classification == "inconclusive"
    assert "two or more brokers" in verdict.detail


def test_losing_on_every_broker_is_alpha_decay(session):
    from app.engine.health import check_across_brokers

    strategy = make_strategy(session)
    add_broker_trades(session, strategy.id, "BrokerA", [-2.0] * 12)
    add_broker_trades(session, strategy.id, "BrokerB", [-1.5] * 12)

    verdict = check_across_brokers(session, strategy)

    assert verdict.classification == "alpha_decay"
    assert "ALPHA DECAY" in verdict.detail


def test_losing_on_one_broker_while_others_hold_names_that_broker(session):
    from app.engine.health import check_across_brokers

    strategy = make_strategy(session)
    add_broker_trades(session, strategy.id, "GoodBroker", [3.0] * 12)
    add_broker_trades(session, strategy.id, "BadBroker", [-3.0] * 12)

    verdict = check_across_brokers(session, strategy)

    assert verdict.classification == "broker_warning"
    assert "BadBroker" in verdict.detail


def test_profitable_everywhere_is_healthy(session):
    from app.engine.health import check_across_brokers

    strategy = make_strategy(session)
    add_broker_trades(session, strategy.id, "BrokerA", [2.0] * 12)
    add_broker_trades(session, strategy.id, "BrokerB", [1.0] * 12)

    assert check_across_brokers(session, strategy).classification == "healthy"


def test_per_broker_comparison_normalises_by_trade_count(session):
    # A bigger account should not look like a better broker.
    from app.engine.health import check_across_brokers

    strategy = make_strategy(session)
    add_broker_trades(session, strategy.id, "Small", [1.0] * 12)
    add_broker_trades(session, strategy.id, "Large", [10.0] * 12)

    verdict = check_across_brokers(session, strategy)

    assert verdict.per_broker["Small"]["profit_per_trade"] == 1.0
    assert verdict.per_broker["Large"]["profit_per_trade"] == 10.0
    assert verdict.classification == "healthy"
