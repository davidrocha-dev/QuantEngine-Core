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
from app.models import LiveTrade, Strategy, StrategyTrade


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


def test_evidence_makes_a_bot_eligible_but_does_not_move_it_on_its_own(session):
    # REQ-PORT-3.6 — with AUTO off, a promotion is a decision someone made,
    # not a number that quietly changed.
    strategy = make_strategy(session)
    add_live_trades(session, strategy.id, [2.0] * PROVING_TRADES)

    entry = ladder(session)[0]

    assert entry.stage == "INCUBATION"
    assert entry.eligible_stage == "PROVING"
    assert entry.can_promote is True


def test_graduating_moves_the_bot_and_its_risk_share(session):
    from app.engine.health import promote

    strategy = make_strategy(session)
    add_live_trades(session, strategy.id, [2.0] * PROVING_TRADES)

    ok, detail, entry = promote(session, strategy.id)

    assert ok is True
    assert entry.stage == "PROVING"
    assert entry.risk_share == 0.5
    assert "PROVING" in detail
    assert ladder(session)[0].stage == "PROVING"


def test_a_bot_cannot_skip_a_stage_in_one_promotion(session):
    from app.engine.health import promote

    strategy = make_strategy(session)
    add_live_trades(session, strategy.id, [2.0] * FULL_RISK_TRADES)

    promote(session, strategy.id)

    assert ladder(session)[0].stage == "PROVING"


def test_auto_promotes_everything_eligible_and_skips_the_rest(session):
    from app.engine.health import auto_promote

    ready = make_strategy(session, "#READY")
    add_live_trades(session, ready.id, [2.0] * PROVING_TRADES)
    make_strategy(session, "#FRESH")

    promoted = auto_promote(session)

    assert [entry.strategy_id for entry in promoted] == ["#READY"]
    stages = {entry.strategy_id: entry.stage for entry in ladder(session)}
    assert stages["#READY"] == "PROVING"
    assert stages["#FRESH"] == "INCUBATION"


def test_risk_shares_are_published_per_strategy(session):
    from app.engine.health import promote, risk_shares

    strategy = make_strategy(session)
    add_live_trades(session, strategy.id, [2.0] * PROVING_TRADES)

    assert risk_shares(session)[strategy.id] == 0.25
    promote(session, strategy.id)
    assert risk_shares(session)[strategy.id] == 0.5


def test_the_risk_file_lists_each_bot_by_its_magic_number(session, tmp_path, monkeypatch):
    from app.broker import pilot
    from app.engine.health import risk_shares

    monkeypatch.setattr(pilot, "common_files_dir", lambda: tmp_path)
    make_strategy(session)

    path = pilot.write_risk_shares(risk_shares(session))

    assert path is not None
    assert path.read_text(encoding="utf-8").strip() == f"{_magic_number('#LIVE01')},0.2500"


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


# --- the magnitude check (REQ-AUTO-2) ------------------------------------


def test_a_bot_winning_far_less_often_is_degraded_even_while_profitable(session):
    """A bot can be net positive and still be a shadow of what was
    approved. Before this, the only thing that counted as degradation was
    losing money outright, so a strategy winning a third as often as its
    backtest read as healthy for as long as the balance crept up."""
    strategy = make_strategy(session)
    strategy.win_rate_pct = 60.0
    session.commit()
    # 2 winners in 12 (17%), but big enough to stay net positive.
    add_live_trades(session, strategy.id, [200.0, 200.0] + [-20.0] * 10)

    verdict = check_strategy(session, strategy)

    assert verdict.status == "degraded"
    assert verdict.live_profit > 0
    assert verdict.deviation_pct is not None and verdict.deviation_pct > 50


def test_a_bot_holding_its_win_rate_is_healthy(session):
    strategy = make_strategy(session)
    strategy.win_rate_pct = 60.0
    session.commit()
    add_live_trades(session, strategy.id, [50.0] * 7 + [-40.0] * 5)  # 58%

    verdict = check_strategy(session, strategy)

    assert verdict.status == "healthy"
    assert "expected" in verdict.detail


def test_a_strategy_bred_before_win_rate_was_recorded_is_not_called_degraded(session):
    # An absent number is not evidence of decay, and every strategy bred
    # before this column existed has one.
    strategy = make_strategy(session)
    assert strategy.win_rate_pct is None
    add_live_trades(session, strategy.id, [10.0, 10.0] + [-1.0] * 10)

    verdict = check_strategy(session, strategy)

    assert verdict.status == "healthy"
    assert verdict.deviation_pct is None


def test_the_win_rate_check_still_needs_enough_live_trades(session):
    strategy = make_strategy(session)
    strategy.win_rate_pct = 60.0
    session.commit()
    add_live_trades(session, strategy.id, [-20.0] * (MIN_TRADES_FOR_VERDICT - 1))

    assert check_strategy(session, strategy).status == "too_early"


# --- The magnitude test has to fire before the money is gone ------------


def test_degradation_is_measured_against_the_strategys_own_breakeven():
    """A fixed relative shortfall cannot serve every geometry.

    Measured across a real Vault, the win rate at which a 50% shortfall
    fires sits 11 to 27 points below the win rate at which the strategy
    stops making money — so it only ever triggered after live profit per
    trade had already turned negative and the sign check had called the
    bot degraded. A magnitude test that fires only after the cheaper test
    has fired is not a second opinion.
    """
    from app.engine.health import breakeven_win_rate_pct, check_strategy

    engine = make_engine(":memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        # Backtest: wins average 60, losses average 100, so the edge dies
        # below 100 / 160 = 62.5% — and the backtest won 75%.
        strategy = Strategy(
            id="#DECAY", data_source="dukascopy", asset="XAUUSD", timeframe="M15",
            score=70, grade="GOOD", deployed=True, trades=40, return_pct=20.0, win_rate_pct=75.0,
            genome_json="{}",
        )
        session.add(strategy)
        for i in range(40):
            session.add(
                StrategyTrade(
                    strategy_id="#DECAY", sequence=i, direction=1,
                    entry_price=1.0, exit_price=1.0, stop_price=0.9, target_price=1.1,
                    r_multiple=1.0, pnl=60.0 if i < 30 else -100.0, exit_reason="target",
                )
            )
        # Live: winning 55% — under the 62.5% it needs, but still showing a
        # positive balance, and nowhere near a 50% shortfall off 75%.
        add_live_trades(session, "#DECAY", [60.0] * 11 + [-55.0] * 9)

        breakeven = breakeven_win_rate_pct(session, "#DECAY")
        assert breakeven is not None
        assert 62.0 < breakeven < 63.0, breakeven

        verdict = check_strategy(session, strategy)

        assert verdict.live_profit > 0, "this fixture is meant to still look profitable"
        shortfall = (75.0 - 55.0) / 75.0 * 100
        assert shortfall < 50.0, "and nowhere near the old fixed threshold"
        assert verdict.status == "degraded", verdict.detail
        assert "break even" in verdict.detail


def test_breakeven_is_unknown_on_too_few_recorded_trades():
    from app.engine.health import MIN_TRADES_FOR_BREAKEVEN, breakeven_win_rate_pct

    engine = make_engine(":memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Strategy(id="#THIN", data_source="dukascopy", asset="EURUSD", timeframe="H1", genome_json="{}"))
        for i in range(MIN_TRADES_FOR_BREAKEVEN - 1):
            session.add(
                StrategyTrade(
                    strategy_id="#THIN", sequence=i, direction=1,
                    entry_price=1.0, exit_price=1.0, stop_price=0.9, target_price=1.1,
                    r_multiple=1.0, pnl=10.0 if i % 2 else -5.0, exit_reason="target",
                )
            )
        session.commit()

        assert breakeven_win_rate_pct(session, "#THIN") is None
