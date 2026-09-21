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
# A live win rate this far below the backtest's is degradation.
#
# Win rate rather than profit: live results are money and the Vault stores
# percentages, and converting between them needs an account balance this
# server never sees. A ratio is comparable on both sides without inventing
# a conversion. What it will not catch on its own is a strategy that keeps
# winning as often but earns less when it does - the sign check below is
# what catches that, once the losses add up.
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
    stage: str  # where the bot actually is
    eligible_stage: str  # where its live record would allow it to be
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
    """The backtest's return per trade, as a percentage of its own equity.

    This is shown, not compared. It is a percentage and live profit is
    money, and turning one into the other needs the account balance behind
    those trades, which nothing here records. Presenting it as a like-for
    -like expectation would be inventing a conversion; the magnitude test
    is done on win rate instead, where both sides are ratios.
    """
    if not strategy.trades or strategy.return_pct is None or strategy.trades == 0:
        return None
    if live_trades == 0 or live_profit_total == 0:
        return None
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

    live_win_rate_pct = sum(1 for trade in trades if trade.profit > 0) / len(trades) * 100

    # The magnitude check. A bot can be net positive and still be a shadow
    # of the thing that was approved - winning a third as often as the
    # backtest did is decay, even while the balance creeps up.
    expected_win_rate_pct = strategy.win_rate_pct
    if expected_win_rate_pct:
        shortfall_pct = (expected_win_rate_pct - live_win_rate_pct) / expected_win_rate_pct * 100
        if shortfall_pct >= DEGRADATION_THRESHOLD_PCT:
            return HealthVerdict(
                strategy_id=strategy.id, live_trades=len(trades), live_profit=round(live_profit, 2),
                live_profit_per_trade=round(per_trade, 2),
                expected_profit_per_trade=_expected_profit_per_trade(strategy, live_profit, len(trades)),
                deviation_pct=round(shortfall_pct, 1), status="degraded",
                detail=(
                    f"Winning {round(live_win_rate_pct)}% of live trades against "
                    f"{round(expected_win_rate_pct)}% in the backtest - {round(shortfall_pct)}% below it."
                ),
            )

    # A strategy bred before the win rate was recorded has nothing to
    # compare against, and an absent number is not evidence of decay.
    basis = (
        f" vs {round(expected_win_rate_pct)}% expected" if expected_win_rate_pct else ""
    )
    return HealthVerdict(
        strategy_id=strategy.id, live_trades=len(trades), live_profit=round(live_profit, 2),
        live_profit_per_trade=round(per_trade, 2),
        expected_profit_per_trade=_expected_profit_per_trade(strategy, live_profit, len(trades)),
        deviation_pct=None, status="healthy",
        detail=(
            f"{len(trades)} live trades, {round(live_win_rate_pct)}% winners{basis}, "
            f"{round(live_profit, 2)} net."
        ),
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

    # REQ-PORT-3.5 — an unusually strong opening run can skip the wait.
    hot_start = len(trades) >= HOT_START_TRADES and winning / max(len(trades), 1) >= HOT_START_WIN_RATE
    if len(trades) >= FULL_RISK_TRADES or days_live >= FULL_RISK_DAYS:
        eligible_stage = "FULL RISK"
    elif len(trades) >= PROVING_TRADES or days_live >= PROVING_DAYS or hot_start:
        eligible_stage = "PROVING"
    else:
        eligible_stage = "INCUBATION"

    # Where the bot actually is: a promotion is a recorded decision, not a
    # number recomputed from its record on every read (REQ-PORT-3.6 — with
    # AUTO off, evidence alone must not move it).
    stage = strategy.incubation_stage if strategy.incubation_stage in STAGES else STAGES[0]
    next_stage = STAGES[STAGES.index(stage) + 1] if stage != "FULL RISK" else None

    # REQ-PORT-3.4 — evidence can qualify a bot and still not earn it.
    blocked_reason = None
    if not trades:
        blocked_reason = "no live trades reported yet"
    elif net < 0:
        blocked_reason = f"underwater ({round(net, 2)}) — promotion is blocked until it recovers"
    elif next_stage is not None and STAGES.index(eligible_stage) <= STAGES.index(stage):
        blocked_reason = f"not enough live evidence for {next_stage} yet"

    return LadderEntry(
        strategy_id=strategy.id,
        stage=stage,
        eligible_stage=eligible_stage,
        risk_share=STAGE_RISK_SHARE[stage],
        live_trades=len(trades),
        winning_trades=winning,
        days_live=days_live,
        can_promote=next_stage is not None and blocked_reason is None,
        blocked_reason=blocked_reason,
        next_stage=next_stage,
    )


def promote(session: Session, strategy_id: str, now: datetime | None = None) -> tuple[bool, str, LadderEntry | None]:
    """REQ-PORT-3.6 — the manual GRADUATE action.

    Promotion is only ever *allowed* by evidence, never forced by it: the
    same blocking rules apply whether a human clicks the button or AUTO
    does it, so this is the single path both take.
    """
    strategy = session.get(Strategy, strategy_id)
    if strategy is None or strategy.deleted_at is not None:
        return False, f"Unknown strategy: {strategy_id}", None

    entry = ladder_entry(session, strategy, now)
    if entry.next_stage is None:
        return False, "Already at full target risk.", entry
    if entry.blocked_reason:
        return False, entry.blocked_reason, entry
    if not entry.can_promote:
        return False, "Not enough live evidence to promote yet.", entry

    strategy.incubation_stage = entry.next_stage
    session.commit()

    promoted = ladder_entry(session, strategy, now)
    return True, f"Promoted to {promoted.stage} ({int(promoted.risk_share * 100)}% of target risk).", promoted


def auto_promote(session: Session, now: datetime | None = None) -> list[LadderEntry]:
    """REQ-PORT-3.6 with AUTO on — promotes everything eligible, silently
    skipping everything that is not. Returns what actually changed."""
    promoted: list[LadderEntry] = []
    for entry in ladder(session, now):
        if entry.can_promote:
            ok, _, updated = promote(session, entry.strategy_id, now)
            if ok and updated is not None:
                promoted.append(updated)
    return promoted


def risk_shares(session: Session, now: datetime | None = None) -> dict[str, float]:
    """What each deployed bot should currently risk, as a share of target."""
    return {entry.strategy_id: entry.risk_share for entry in ladder(session, now)}


def ladder(session: Session, now: datetime | None = None) -> list[LadderEntry]:
    deployed = session.scalars(
        select(Strategy).where(Strategy.deleted_at.is_(None)).where(Strategy.deployed.is_(True))
    )
    return [ladder_entry(session, strategy, now) for strategy in deployed]
