"""The Parity Test — is the EA executing what was approved?

A health check asks whether a strategy still has an edge. This asks a
narrower and more mechanical question: is the robot on the chart doing
the same *amount* of trading, in the same direction of outcome, as the
backtest that approved it?

They are different failures with different fixes. An edge that decayed is
a market problem and the answer is to retire the strategy. An EA on the
wrong timeframe, on the wrong chart, or with algo trading switched off is
an operator problem, and the answer is to go and fix the terminal - the
strategy was never wrong. `guide_parity_test.md` lists exactly those
causes, and every one of them shows up first as trading too little.

What this deliberately does not claim: it cannot measure per-fill
slippage. That would need the moment each position *opened*, and the
Pilot reports closed deals - `closed_at` is a close time. Reporting an
invented slippage figure would be worse than admitting the gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import LiveTrade, Strategy, StrategyTrade

# Below this many live trades, a rate computed from them is noise.
MIN_TRADES_FOR_PARITY = 8
# How far the live trading rate may drift from the backtest's before it
# is worth telling someone. Live trading is never exactly as frequent as
# a simulation - spreads widen, a session is missed, the terminal
# restarts - so a narrow band would cry wolf constantly.
RATE_TOLERANCE_PCT = 40.0


@dataclass
class ParityVerdict:
    strategy_id: str
    status: str  # aligned | trading_less | trading_more | too_early | no_data | unknown_expectation
    live_trades: int
    live_days: float
    live_trades_per_day: float | None
    expected_trades_per_day: float | None
    rate_deviation_pct: float | None
    detail: str

    def to_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _expected_rate(session: Session, strategy: Strategy) -> float | None:
    """Backtest trades per day, measured over the window they happened in.

    Taken from the stored trade log (migration v8), which carries the
    opening and closing timestamp of every backtested trade, so the
    denominator is the span those trades actually covered.

    It used to be derived from the row's aggregates:

        busy_days = trades * avg_hold_hours / 24
        rate      = trades / busy_days

    which cancels to 24 / avg_hold_hours - the trade count drops out
    entirely. What is left assumes the strategy is in a position every
    hour of every day, back to back with no gaps, which is the
    theoretical maximum rather than anything it did. Measured across this
    Vault it inflated the expectation by 1.4x to 3.8x, and six of eight
    strategies would have been reported as "trading less" while executing
    at exactly their backtested rate - each one telling the user to go
    and check a terminal that was working. The comment justifying it had
    the direction backwards: raising the expectation makes the
    trading-less branch fire more eagerly, not less.
    """
    rows = list(
        session.execute(
            select(StrategyTrade.opened_at, StrategyTrade.closed_at).where(
                StrategyTrade.strategy_id == strategy.id
            )
        )
    )
    stamps = [value for row in rows for value in row if value is not None]
    if len(stamps) < 2 or len(rows) < MIN_TRADES_FOR_PARITY:
        return None
    span_days = (max(stamps) - min(stamps)) / timedelta(days=1)
    if span_days <= 0:
        return None
    return len(rows) / span_days


def check_parity(session: Session, strategy: Strategy, now: datetime | None = None) -> ParityVerdict:
    now = now or datetime.now(timezone.utc)
    trades = list(
        session.scalars(
            select(LiveTrade).where(LiveTrade.strategy_id == strategy.id).order_by(LiveTrade.closed_at)
        )
    )

    if not trades:
        return ParityVerdict(
            strategy.id, "no_data", 0, 0.0, None, None, None,
            "No live trades reported yet. Check the Pilot EA is attached to a chart and algo trading is on.",
        )

    if len(trades) < MIN_TRADES_FOR_PARITY:
        return ParityVerdict(
            strategy.id, "too_early", len(trades), 0.0, None, None, None,
            f"{len(trades)} live trades so far — parity needs {MIN_TRADES_FOR_PARITY} before a rate means anything.",
        )

    first, last = _aware(trades[0].closed_at), _aware(trades[-1].closed_at)
    live_days = max((last - first) / timedelta(days=1), 1 / 24)
    live_rate = len(trades) / live_days

    expected = _expected_rate(session, strategy)
    if expected is None:
        return ParityVerdict(
            strategy.id, "unknown_expectation", len(trades), round(live_days, 2), round(live_rate, 2), None, None,
            (
                f"Trading {live_rate:.1f} times a day live. This strategy has no stored backtest trade log to "
                "compare against — run REFRESH on it and the comparison becomes available."
            ),
        )

    deviation = (live_rate - expected) / expected * 100
    common = dict(
        strategy_id=strategy.id, live_trades=len(trades), live_days=round(live_days, 2),
        live_trades_per_day=round(live_rate, 2), expected_trades_per_day=round(expected, 2),
        rate_deviation_pct=round(deviation, 1),
    )

    if deviation < -RATE_TOLERANCE_PCT:
        return ParityVerdict(
            **common, status="trading_less",
            detail=(
                f"Trading {live_rate:.1f} times a day against {expected:.1f} in the backtest — "
                f"{abs(deviation):.0f}% less. That is usually the terminal, not the strategy: check the EA is on "
                "the right symbol and timeframe, that algo trading is enabled, and that the chart has been open."
            ),
        )

    if deviation > RATE_TOLERANCE_PCT:
        return ParityVerdict(
            **common, status="trading_more",
            detail=(
                f"Trading {live_rate:.1f} times a day against {expected:.1f} in the backtest — "
                f"{deviation:.0f}% more. Check the EA is not attached to two charts at once, and that the "
                "symbol's spread has not widened enough to trigger entries the backtest never took."
            ),
        )

    return ParityVerdict(
        **common, status="aligned",
        detail=(
            f"Trading {live_rate:.1f} times a day against {expected:.1f} in the backtest. The EA is executing "
            "about as often as it was designed to — whatever else is true, it is on the right chart and running."
        ),
    )
