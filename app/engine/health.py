"""Health checks and the Incubation ladder — REQ-AUTO-2, REQ-PORT-3.

Both answer the same question from live results: is this bot behaving like
the backtest that approved it?

A health check compares live profit per trade against the out-of-sample
expectation stored with the strategy. The comparison is deliberately
coarse — a handful of live trades cannot resolve a small edge — so a bot
is only called degraded once it has enough trades for the answer to mean
something, and until then it is reported as too early rather than healthy.

The Incubation ladder (REQ-PORT-3) starts every deployed bot at a quarter
of its target risk and raises it on live evidence, not on time alone. A
bot that is underwater, or trading far less than its backtest predicted,
does not advance — the ladder exists precisely to stop a bad deploy from
reaching full size.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import LiveTrade, Strategy

# Below this many live trades, live numbers say more about luck than edge.
MIN_TRADES_FOR_VERDICT = 10
# Live profit per trade this far below the backtest's is degradation.
DEGRADATION_THRESHOLD_PCT = 50.0

# REQ-PORT-3.1-3.3 — stages, and what it takes to leave each one.
STAGES = ("INCUBATION", "PROVING", "FULL RISK")
STAGE_RISK_SHARE = {"INCUBATION": 0.25, "PROVING": 0.5, "FULL RISK": 1.0}
PROVING_TRADES = 10
FULL_RISK_TRADES = 30
PROVING_DAYS = 14
FULL_RISK_DAYS = 42
# REQ-PORT-3.5 — an unusually strong opening run can skip the wait.
HOT_START_TRADES = 8
HOT_START_WIN_RATE = 0.8


@dataclass
class HealthVerdict:
    strategy_id: str
    live_trades: int
    live_profit: float
    live_profit_per_trade: float | None
    expected_profit_per_trade: float | None
    deviation_pct: float | None
    status: str  # healthy | degraded | too_early | no_data
    detail: str

    def to_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}


@dataclass
class LadderEntry:
    strategy_id: str
    stage: str
    risk_share: float
    live_trades: int
    winning_trades: int
    days_live: int
    can_promote: bool
    blocked_reason: str | None
    next_stage: str | None

    def to_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}


def _live_trades(session: Session, strategy_id: str) -> list[LiveTrade]:
    return list(
        session.scalars(
            select(LiveTrade).where(LiveTrade.strategy_id == strategy_id).order_by(LiveTrade.closed_at)
        )
    )


def _expected_profit_per_trade(strategy: Strategy, live_profit_total: float, live_trades: int) -> float | None:
    """The backtest's return per trade, expressed in the live account's own
    money so the two are comparable.

    The Vault stores percentages, not currency, so the expectation is
    scaled by what the live trades themselves imply about position size —
    which is why this is a ratio check, not a precise P&L forecast.
    """
    if not strategy.trades or strategy.return_pct is None or strategy.trades == 0:
        return None
    if live_trades == 0 or live_profit_total == 0:
        return None
    # Backtest return per trade, as a fraction of the backtest's own equity.
    return strategy.return_pct / strategy.trades


def check_strategy(session: Session, strategy: Strategy) -> HealthVerdict:
    trades = _live_trades(session, strategy.id)
    if not trades:
        return HealthVerdict(
            strategy_id=strategy.id, live_trades=0, live_profit=0.0, live_profit_per_trade=None,
            expected_profit_per_trade=None, deviation_pct=None, status="no_data",
            detail="No live trades reported yet — is the Pilot attached to a chart?",
        )

    live_profit = sum(trade.profit for trade in trades)
    per_trade = live_profit / len(trades)

    if len(trades) < MIN_TRADES_FOR_VERDICT:
        return HealthVerdict(
            strategy_id=strategy.id, live_trades=len(trades), live_profit=round(live_profit, 2),
            live_profit_per_trade=round(per_trade, 2), expected_profit_per_trade=None, deviation_pct=None,
            status="too_early",
            detail=f"{len(trades)} live trades — needs {MIN_TRADES_FOR_VERDICT} before the comparison means anything.",
        )

    # Sign agreement is the robust part of the comparison: a backtest that
    # made money and a live run that loses it is the signal that matters,
    # long before the size of the gap is measurable.
    expected_sign = 1 if (strategy.return_pct or 0) > 0 else -1
    live_sign = 1 if per_trade > 0 else -1

    if expected_sign > 0 and live_sign < 0:
        return HealthVerdict(
            strategy_id=strategy.id, live_trades=len(trades), live_profit=round(live_profit, 2),
            live_profit_per_trade=round(per_trade, 2),
            expected_profit_per_trade=_expected_profit_per_trade(strategy, live_profit, len(trades)),
            deviation_pct=100.0, status="degraded",
            detail=f"Losing live ({round(live_profit, 2)}) on a strategy whose backtest was profitable.",
        )

    win_rate = sum(1 for trade in trades if trade.profit > 0) / len(trades)
    return HealthVerdict(
        strategy_id=strategy.id, live_trades=len(trades), live_profit=round(live_profit, 2),
        live_profit_per_trade=round(per_trade, 2),
        expected_profit_per_trade=_expected_profit_per_trade(strategy, live_profit, len(trades)),
        deviation_pct=None, status="healthy",
        detail=f"{len(trades)} live trades, {round(win_rate * 100)}% winners, {round(live_profit, 2)} net.",
    )


def run_health_check(session: Session, strategy_ids: list[str] | None = None) -> list[HealthVerdict]:
    """REQ-AUTO-2/5 — every deployed strategy, or the ones named."""
    statement = select(Strategy).where(Strategy.deleted_at.is_(None))
    if strategy_ids:
        statement = statement.where(Strategy.id.in_(strategy_ids))
    else:
        statement = statement.where(Strategy.deployed.is_(True))
    return [check_strategy(session, strategy) for strategy in session.scalars(statement)]


@dataclass
class BrokerVerdict:
    strategy_id: str
    classification: str  # alpha_decay | broker_warning | healthy | inconclusive
    per_broker: dict[str, dict]
    detail: str

    def to_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}


def check_across_brokers(session: Session, strategy: Strategy) -> BrokerVerdict:
    """REQ-AUTO-3 — tells a dead edge apart from one broker's problem.

    Returns are normalised to profit per trade before comparison, because
    the same strategy will be sized differently on different accounts and
    raw profit would make the biggest account look like the best broker.
    """
    trades = [trade for trade in _live_trades(session, strategy.id) if trade.broker]
    by_broker: dict[str, list] = {}
    for trade in trades:
        by_broker.setdefault(trade.broker, []).append(trade)

    per_broker = {
        broker: {
            "trades": len(rows),
            "profit": round(sum(t.profit for t in rows), 2),
            "profit_per_trade": round(sum(t.profit for t in rows) / len(rows), 2),
        }
        for broker, rows in by_broker.items()
    }

    measurable = {b: stats for b, stats in per_broker.items() if stats["trades"] >= MIN_TRADES_FOR_VERDICT}
    if len(measurable) < 2:
        return BrokerVerdict(
            strategy_id=strategy.id, classification="inconclusive", per_broker=per_broker,
            detail=(
                "Needs the same strategy running on two or more brokers, each with "
                f"{MIN_TRADES_FOR_VERDICT}+ live trades, before the two causes can be told apart."
            ),
        )

    losing = [broker for broker, stats in measurable.items() if stats["profit_per_trade"] < 0]
    if not losing:
        return BrokerVerdict(
            strategy_id=strategy.id, classification="healthy", per_broker=per_broker,
            detail=f"Profitable on all {len(measurable)} brokers reporting.",
        )
    if len(losing) == len(measurable):
        # REQ-AUTO-3.2 — every terminal declining together is the edge.
        return BrokerVerdict(
            strategy_id=strategy.id, classification="alpha_decay", per_broker=per_broker,
            detail="ALPHA DECAY — losing on every broker reporting. Retire the bot.",
        )
    # REQ-AUTO-3.3 — one terminal out of step is an execution problem.
    return BrokerVerdict(
        strategy_id=strategy.id, classification="broker_warning", per_broker=per_broker,
        detail=f"BROKER WARNING — losing only on {', '.join(losing)} while the others hold.",
    )


def check_all_brokers(session: Session) -> list[BrokerVerdict]:
    deployed = session.scalars(
        select(Strategy).where(Strategy.deleted_at.is_(None)).where(Strategy.deployed.is_(True))
    )
    return [check_across_brokers(session, strategy) for strategy in deployed]


def ladder_entry(session: Session, strategy: Strategy, now: datetime | None = None) -> LadderEntry:
    """REQ-PORT-3 — where this bot stands, and what is holding it back."""
    now = now or datetime.now(timezone.utc)
    trades = _live_trades(session, strategy.id)
    winning = sum(1 for trade in trades if trade.profit > 0)
    net = sum(trade.profit for trade in trades)

    first = trades[0].closed_at if trades else None
    if first is not None and first.tzinfo is None:
        first = first.replace(tzinfo=timezone.utc)
    days_live = (now - first).days if first else 0

    hot_start = len(trades) >= HOT_START_TRADES and winning / max(len(trades), 1) >= HOT_START_WIN_RATE
    qualifies_full = len(trades) >= FULL_RISK_TRADES or days_live >= FULL_RISK_DAYS
    qualifies_proving = len(trades) >= PROVING_TRADES or days_live >= PROVING_DAYS or hot_start

    if qualifies_full:
        stage = "FULL RISK"
    elif qualifies_proving:
        stage = "PROVING"
    else:
        stage = "INCUBATION"

    # REQ-PORT-3.4 — evidence can qualify a bot for the next stage and
    # still not earn it: losing money is a hard stop.
    blocked_reason = None
    if net < 0:
        blocked_reason = f"underwater ({round(net, 2)}) — promotion is blocked until it recovers"
        stage = "INCUBATION" if stage == "PROVING" else stage
    elif not trades:
        blocked_reason = "no live trades reported yet"

    next_stage = STAGES[STAGES.index(stage) + 1] if stage != "FULL RISK" else None
    return LadderEntry(
        strategy_id=strategy.id,
        stage=stage,
        risk_share=STAGE_RISK_SHARE[stage],
        live_trades=len(trades),
        winning_trades=winning,
        days_live=days_live,
        can_promote=next_stage is not None and blocked_reason is None,
        blocked_reason=blocked_reason,
        next_stage=next_stage,
    )


def ladder(session: Session, now: datetime | None = None) -> list[LadderEntry]:
    deployed = session.scalars(
        select(Strategy).where(Strategy.deleted_at.is_(None)).where(Strategy.deployed.is_(True))
    )
    return [ladder_entry(session, strategy, now) for strategy in deployed]
