"""Persisting the trades behind a Vault row.

The Vault stored aggregates and an equity curve, which is enough to rank
a strategy and not enough to interrogate one. Anything worth asking after
the fact - which sessions it actually makes money in, whether it survives
without its best three trades, what the exits really looked like - meant
re-simulating the whole out-of-sample window first.

Timestamps rather than bar indices, because an index only means something
against the exact series it came from, and that series is a cache that
grows every day.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.engine.backtest import Trade
from app.models import StrategyTrade


def _at(bars: pd.DataFrame, index: int):
    if index < 0 or index >= len(bars):
        return None
    return pd.Timestamp(bars["timestamp"].iloc[index]).to_pydatetime().replace(tzinfo=None)


def replace_trades(session: Session, strategy_id: str, trades: list[Trade], bars: pd.DataFrame) -> int:
    """Writes this strategy's trade list, replacing whatever was there.

    Replacing rather than appending: a REFRESH re-runs the same genome
    over newer data, and keeping both runs would leave a history that
    belongs to no single backtest.
    """
    session.execute(delete(StrategyTrade).where(StrategyTrade.strategy_id == strategy_id))
    rows = [
        StrategyTrade(
            strategy_id=strategy_id,
            sequence=sequence,
            opened_at=_at(bars, trade.entry_idx),
            closed_at=_at(bars, trade.exit_idx),
            direction=trade.direction,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            stop_price=trade.stop_price,
            target_price=trade.target_price,
            r_multiple=trade.r_multiple,
            pnl=trade.pnl,
            exit_reason=trade.exit_reason,
        )
        for sequence, trade in enumerate(trades)
    ]
    session.add_all(rows)
    return len(rows)
