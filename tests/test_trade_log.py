"""The trades behind a Vault row — kept, not re-simulated.

The Vault stored aggregates and an equity curve. That ranks a strategy
and cannot answer a question about one: which sessions it makes money in,
whether it survives without its best three trades, what the exits
actually looked like. All of it meant re-running the whole out-of-sample
window first.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy.orm import Session

from app.db import Base, make_engine
from app.engine.backtest import Trade
from app.engine.trade_log import replace_trades
from app.models import Strategy, StrategyTrade


@pytest.fixture
def session(tmp_path: Path) -> Session:
    engine = make_engine(tmp_path / "vault.db")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        s.add(
            Strategy(
                id="#TRADES", data_source="dukascopy", asset="EURUSD", timeframe="M15",
                genome_json="{}", trades=3, score=70, grade="GOOD",
            )
        )
        s.commit()
        yield s


@pytest.fixture
def bars() -> pd.DataFrame:
    start = datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)
    return pd.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=15 * i) for i in range(20)],
            "open": [1.08] * 20, "high": [1.081] * 20, "low": [1.079] * 20,
            "close": [1.08] * 20, "volume": [1.0] * 20,
        }
    )


def make_trade(entry_idx: int, exit_idx: int, r: float = 2.0, reason: str = "target") -> Trade:
    return Trade(
        entry_idx=entry_idx, exit_idx=exit_idx, direction=1, entry_price=1.0800,
        exit_price=1.0820, stop_price=1.0790, target_price=1.0820,
        r_multiple=r, pnl=r * 100, exit_reason=reason,
    )


def test_trades_are_stored_against_their_strategy(session, bars):
    written = replace_trades(session, "#TRADES", [make_trade(0, 4), make_trade(6, 9)], bars)
    session.commit()

    rows = session.query(StrategyTrade).order_by(StrategyTrade.sequence).all()

    assert written == 2
    assert [row.sequence for row in rows] == [0, 1]
    assert rows[0].exit_reason == "target"
    assert rows[0].r_multiple == 2.0


def test_a_trade_carries_the_time_it_happened_not_a_bar_index(session, bars):
    """An index only means something against the exact series it came
    from, and that series is a cache that grows every day."""
    replace_trades(session, "#TRADES", [make_trade(2, 5)], bars)
    session.commit()

    row = session.query(StrategyTrade).one()

    assert row.opened_at == datetime(2026, 3, 2, 9, 30)
    assert row.closed_at == datetime(2026, 3, 2, 10, 15)


def test_rerunning_replaces_rather_than_appends(session, bars):
    # A REFRESH re-runs the same genome over newer data; keeping both runs
    # would leave a history belonging to no single backtest.
    replace_trades(session, "#TRADES", [make_trade(0, 2), make_trade(3, 5)], bars)
    session.commit()

    replace_trades(session, "#TRADES", [make_trade(1, 4, r=-1.0, reason="stop")], bars)
    session.commit()

    rows = session.query(StrategyTrade).all()
    assert len(rows) == 1
    assert rows[0].exit_reason == "stop"


def test_one_strategy_replacing_its_trades_leaves_another_alone(session, bars):
    session.add(
        Strategy(
            id="#OTHER", data_source="dukascopy", asset="GBPUSD", timeframe="M15",
            genome_json="{}", trades=1, score=60, grade="GOOD",
        )
    )
    session.commit()
    replace_trades(session, "#OTHER", [make_trade(0, 3)], bars)
    replace_trades(session, "#TRADES", [make_trade(0, 3), make_trade(4, 7)], bars)
    session.commit()

    replace_trades(session, "#TRADES", [], bars)
    session.commit()

    assert session.query(StrategyTrade).filter_by(strategy_id="#OTHER").count() == 1
    assert session.query(StrategyTrade).filter_by(strategy_id="#TRADES").count() == 0


def test_a_trade_beyond_the_end_of_the_bars_stores_no_time_rather_than_crashing(session, bars):
    # max_hold can land the exit on the final bar; an off-by-one here
    # should not take down a whole campaign's Vault write.
    replace_trades(session, "#TRADES", [make_trade(18, 99)], bars)
    session.commit()

    row = session.query(StrategyTrade).one()

    assert row.opened_at is not None
    assert row.closed_at is None
