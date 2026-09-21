"""The Parity Test — guide_parity_test.md.

This button shipped with a guide explaining it and an endpoint that
answered "not implemented yet". The question it answers is not the one
the health check answers: a decayed edge is a market problem and the
strategy should be retired, while an EA on the wrong chart is an operator
problem and the strategy was never wrong. Both look like disappointing
live results and the fixes are opposites.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.db import Base, make_engine
from app.engine.parity import MIN_TRADES_FOR_PARITY, check_parity
from app.models import LiveTrade, Strategy


@pytest.fixture
def session(tmp_path: Path) -> Session:
    engine = make_engine(tmp_path / "vault.db")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def make_strategy(session: Session, trades: int = 100, avg_hold_hours: float | None = 6.0) -> Strategy:
    row = Strategy(
        id="#PARITY", data_source="dukascopy", asset="EURUSD", timeframe="M15",
        genome_json="{}", trades=trades, return_pct=20.0, score=70, grade="GOOD",
        avg_hold_hours=avg_hold_hours, deployed=True,
    )
    session.add(row)
    session.commit()
    return row


def add_trades(session: Session, strategy_id: str, count: int, over_days: float) -> None:
    """`count` deals spread evenly across `over_days`."""
    start = datetime.now(timezone.utc) - timedelta(days=over_days)
    step = timedelta(days=over_days / max(count - 1, 1))
    for index in range(count):
        session.add(
            LiveTrade(
                deal_ticket=f"{strategy_id}-{index}", strategy_id=strategy_id, magic=1,
                symbol="EURUSD", closed_at=start + step * index, volume=0.1, profit=1.0,
            )
        )
    session.commit()


def test_nothing_reported_points_at_the_terminal(session):
    strategy = make_strategy(session)

    verdict = check_parity(session, strategy)

    assert verdict.status == "no_data"
    assert "algo trading" in verdict.detail


def test_a_handful_of_trades_is_not_a_rate(session):
    strategy = make_strategy(session)
    add_trades(session, strategy.id, MIN_TRADES_FOR_PARITY - 1, over_days=3)

    assert check_parity(session, strategy).status == "too_early"


def test_an_ea_executing_as_designed_is_aligned(session):
    # 100 trades averaging 6h each implies about 4 a day; live matches.
    strategy = make_strategy(session, trades=100, avg_hold_hours=6.0)
    add_trades(session, strategy.id, 40, over_days=10)

    verdict = check_parity(session, strategy)

    assert verdict.status == "aligned"
    assert verdict.expected_trades_per_day == pytest.approx(4.0, abs=0.1)
    assert verdict.live_trades_per_day == pytest.approx(4.0, abs=0.5)


def test_an_ea_barely_trading_points_at_the_chart_not_the_strategy(session):
    """The failure this test exists for.

    Wrong timeframe, wrong symbol, algo trading off, terminal closed
    overnight - every one of them shows up first as trading far less than
    the backtest, and none of them means the strategy is broken.
    """
    strategy = make_strategy(session, trades=100, avg_hold_hours=6.0)  # ~4/day expected
    add_trades(session, strategy.id, 10, over_days=10)  # 1/day live

    verdict = check_parity(session, strategy)

    assert verdict.status == "trading_less"
    assert verdict.rate_deviation_pct < -40
    assert "right symbol and timeframe" in verdict.detail


def test_an_ea_trading_far_too_much_suggests_two_charts(session):
    strategy = make_strategy(session, trades=100, avg_hold_hours=6.0)
    add_trades(session, strategy.id, 100, over_days=10)  # 10/day against 4

    verdict = check_parity(session, strategy)

    assert verdict.status == "trading_more"
    assert "two charts" in verdict.detail


def test_normal_drift_is_not_reported_as_a_problem(session):
    # Live is never exactly as frequent as a simulation. A band that
    # narrow would cry wolf on every strategy, every week.
    strategy = make_strategy(session, trades=100, avg_hold_hours=6.0)
    add_trades(session, strategy.id, 30, over_days=10)  # 3/day against 4

    assert check_parity(session, strategy).status == "aligned"


def test_a_strategy_with_no_recorded_hold_time_says_so_instead_of_guessing(session):
    strategy = make_strategy(session, trades=100, avg_hold_hours=None)
    add_trades(session, strategy.id, 40, over_days=10)

    verdict = check_parity(session, strategy)

    assert verdict.status == "unknown_expectation"
    assert verdict.live_trades_per_day is not None  # what is known is still reported
    assert "REFRESH" in verdict.detail
