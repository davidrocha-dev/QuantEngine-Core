"""Health checks and the Incubation ladder — REQ-AUTO-2, REQ-PORT-3.

Both answer the same question from live results: is this bot behaving like
the backtest that approved it?

A health check compares live profit per trade against the out-of-sample
expectation stored with the strategy. The comparison is deliberately
coarse — a handful of live trades cannot resolve a small edge — so a bot
is only called degraded once it has enough trades for the answer to mean
something, and until then it is reported as too early rather than healthy.

The Incubation ladder (REQ-PORT-3) starts every deployed bot at a quarter
of its target risk and raises it on live evidence. Enough time live can
qualify a bot as well as enough trades, but it never overrides the
blockers: a bot that is underwater, flagged degraded, or trading far less
than its backtest predicted does not advance — the ladder exists precisely to stop a bad deploy from
reaching full size.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import LiveTrade, Strategy, StrategyTrade

# Below this many live trades, live numbers say more about luck than edge.
MIN_TRADES_FOR_VERDICT = 10
# Fewest backtest trades worth resampling, or reading a breakeven win
# rate from (average loss / (average win + average loss)).
MIN_TRADES_FOR_BREAKEVEN = 20

# A live record is called degraded only when it would be unlikely from
# the strategy the backtest describes - below this probability. Losing
# money over the first ten trades is not that: a strategy winning 45% of
# the time at 1.8R does it in roughly one run in four by chance alone,
# and each of those was retired by auto-deploy as if its edge had died.
DEGRADATION_P_VALUE = 0.05
BOOTSTRAP_SAMPLES = 4000
# With no trade log to resample, a live mean this many standard errors
# below zero is the fallback (one-sided p of about 4% at ten trades).
LOSING_T_STAT = -2.0

# REQ-PORT-3.1-3.3 — stages, and what it takes to leave each one.
STAGES = ("INCUBATION", "PROVING", "FULL RISK")
STAGE_RISK_SHARE = {"INCUBATION": 0.25, "PROVING": 0.5, "FULL RISK": 1.0}
PROVING_TRADES = 10
FULL_RISK_TRADES = 30
PROVING_DAYS = 14
FULL_RISK_DAYS = 42
# A bot this unlikely to be trading so rarely, given its backtest's pace,
# is not promoted (guide_incubation.md: "low activity is not silence -
# it is a warning signal"). Judged only after this many days live.
LOW_ACTIVITY_MIN_DAYS = 14
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


def _backtest_pnls(session: Session, strategy_id: str) -> list[float]:
    return list(session.scalars(select(StrategyTrade.pnl).where(StrategyTrade.strategy_id == strategy_id)))


def _probability_of_so_little(live_pnls: list[float], backtest_pnls: list[float]) -> float | None:
    """How often the backtest's own trades, drawn at random, add up to a
    result as poor as the live one.

    Both sides are in money, on accounts of different sizes, so each is
    measured in units of its own average trade size (mean |P&L|): what is
    compared is the shape - how often it wins, by how much against how
    much it loses - which is what an edge is. None without a trade log
    long enough to resample.
    """
    if len(backtest_pnls) < MIN_TRADES_FOR_BREAKEVEN or not live_pnls:
        return None
    backtest = np.asarray(backtest_pnls, dtype=float)
    live = np.asarray(live_pnls, dtype=float)
    backtest_scale, live_scale = np.abs(backtest).mean(), np.abs(live).mean()
    if backtest_scale <= 0 or live_scale <= 0:
        return None
    observed = live.sum() / live_scale
    # Fixed seed: the same record gets the same verdict on every read.
    draws = np.random.default_rng(0).choice(backtest / backtest_scale, size=(BOOTSTRAP_SAMPLES, live.size))
    return float((draws.sum(axis=1) <= observed).mean())


def _significantly_losing(live_pnls: list[float]) -> bool:
    """Without a backtest trade log: is the live mean clearly below zero,
    rather than a losing streak a profitable strategy would also have?"""
    n = len(live_pnls)
    mean = sum(live_pnls) / n
    if mean >= 0:
        return False
    spread = math.sqrt(sum((x - mean) ** 2 for x in live_pnls) / (n - 1)) if n > 1 else 0.0
    if spread == 0:
        return True  # every trade lost the same amount
    return mean / (spread / math.sqrt(n)) <= LOSING_T_STAT


def _probability_of_so_few_wins(wins: int, trades: int, expected_win_rate_pct: float) -> float:
    """P(at most `wins` winners in `trades`) for the backtest's win rate."""
    p = min(max(expected_win_rate_pct / 100, 0.0), 1.0)
    return sum(math.comb(trades, k) * p**k * (1 - p) ** (trades - k) for k in range(wins + 1))


def breakeven_win_rate_pct(session: Session, strategy_id: str) -> float | None:
    """The win rate at which this strategy's backtested trades break even.

    Read from its own stored trade log (migration v8): with an average
    win of W and an average loss of L, expectancy is zero at
    L / (W + L). Returns None when too few trades are on record for the
    two averages to mean anything, in which case the caller falls back to
    the fixed relative shortfall.
    """
    rows = _backtest_pnls(session, strategy_id)
    if len(rows) < MIN_TRADES_FOR_BREAKEVEN:
        return None
    wins = [pnl for pnl in rows if pnl > 0]
    losses = [-pnl for pnl in rows if pnl < 0]
    if not wins or not losses:
        return None
    avg_win = sum(wins) / len(wins)
    avg_loss = sum(losses) / len(losses)
    if avg_win + avg_loss <= 0:
        return None
    return avg_loss / (avg_win + avg_loss) * 100


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

    live_pnls = [trade.profit for trade in trades]
    backtest_pnls = _backtest_pnls(session, strategy.id)
    wins = sum(1 for profit in live_pnls if profit > 0)
    live_win_rate_pct = wins / len(trades) * 100
    expected_win_rate_pct = strategy.win_rate_pct

    def degraded(detail: str, deviation_pct: float | None) -> HealthVerdict:
        return HealthVerdict(
            strategy_id=strategy.id, live_trades=len(trades), live_profit=round(live_profit, 2),
            live_profit_per_trade=round(per_trade, 2),
            expected_profit_per_trade=_expected_profit_per_trade(strategy, live_profit, len(trades)),
            deviation_pct=deviation_pct, status="degraded", detail=detail,
        )

    # Every test below asks the same question: would the strategy the
    # backtest describes produce a live record this poor by chance? Only
    # when that is unlikely is the bot degraded. The old rules fired on
    # the sign of ten trades, or on any win rate under breakeven, and a
    # profitable strategy trips those routinely before its edge shows.

    # Money: resample the backtest's own trades.
    if (strategy.return_pct or 0) > 0:
        chance = _probability_of_so_little(live_pnls, backtest_pnls)
        if chance is not None and chance < DEGRADATION_P_VALUE:
            return degraded(
                f"{round(live_profit, 2)} over {len(trades)} live trades. The backtest's own trades, drawn at "
                f"random, do this badly {chance * 100:.1f}% of the time - its edge is not showing live.",
                100.0 if live_profit < 0 else None,
            )
        if chance is None and _significantly_losing(live_pnls):
            return degraded(
                f"Losing live ({round(live_profit, 2)} over {len(trades)} trades) on a strategy whose backtest "
                "was profitable - more consistently than a losing streak would.",
                100.0,
            )

    # Win rate: how often the backtest's win rate yields this few winners.
    if expected_win_rate_pct:
        chance = _probability_of_so_few_wins(wins, len(trades), expected_win_rate_pct)
        if chance < DEGRADATION_P_VALUE:
            shortfall_pct = (expected_win_rate_pct - live_win_rate_pct) / expected_win_rate_pct * 100
            breakeven_pct = breakeven_win_rate_pct(session, strategy.id)
            needs = (
                f" It needs {round(breakeven_pct)}% to break even on its own win/loss sizes."
                if breakeven_pct is not None else ""
            )
            return degraded(
                f"Winning {round(live_win_rate_pct)}% of live trades against {round(expected_win_rate_pct)}% in "
                f"the backtest - that few winners happens {chance * 100:.1f}% of the time by chance.{needs}",
                round(shortfall_pct, 1),
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


def _expected_trades_per_day(strategy: Strategy) -> float | None:
    """The backtest's pace: out-of-sample trades over the out-of-sample
    span, read from the stored OOS equity curve's first and last point."""
    if not strategy.trades or not strategy.equity_oos:
        return None
    try:
        points = json.loads(strategy.equity_oos)
        span = datetime.fromisoformat(points[-1]["t"]) - datetime.fromisoformat(points[0]["t"])
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    days = span.total_seconds() / 86_400
    return strategy.trades / days if days >= 1 else None


def _probability_of_so_few_trades(observed: int, expected: float) -> float:
    """P(at most `observed` trades) when `expected` were due (Poisson)."""
    term = total = math.exp(-expected)
    for k in range(1, observed + 1):
        term *= expected / k
        total += term
    return min(total, 1.0)


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
    # Time alone can qualify one (guide_incubation.md), which is only safe
    # because every one of these still has to be clear: a quiet bot that
    # has placed three trades in six weeks has not proven anything by
    # having been switched on for six weeks.
    blocked_reason = None
    per_day = _expected_trades_per_day(strategy)
    expected = per_day * days_live if per_day is not None else None
    if not trades:
        blocked_reason = "no live trades reported yet"
    elif net < 0:
        blocked_reason = f"underwater ({round(net, 2)}) — promotion is blocked until it recovers"
    elif (verdict := check_strategy(session, strategy)).status == "degraded":
        blocked_reason = f"decay warning: {verdict.detail}"
    elif (
        expected is not None
        and days_live >= LOW_ACTIVITY_MIN_DAYS
        and _probability_of_so_few_trades(len(trades), expected) < DEGRADATION_P_VALUE
    ):
        blocked_reason = (
            f"trading far less than its backtest: {len(trades)} trades in {days_live} days, "
            f"about {round(expected)} expected"
        )
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
