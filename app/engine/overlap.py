"""Is this strategy the same bet as one already running? — REQ-PORT-2.

Correlation was only ever checked inside a portfolio, which means it was
checked in the one place somebody had already decided to think about
diversification. The deploy button did not consult it at all, so three
strategies that are the same idea in different clothes could go live one
after another, and the account would carry three times the intended risk
on a single bet while every screen showed three separate bots behaving
themselves.

This is the check the deploy path was missing. It is deliberately a
warning rather than a veto: correlation is measured from a finite
out-of-sample window and there are good reasons to run two related
strategies on purpose. What it must not do is stay quiet.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.engine.portfolio import MIN_OVERLAP_DAYS, daily_returns
from app.models import Strategy
import pandas as pd

# Above this, two strategies are close enough to the same bet that
# deploying both is a position-sizing decision rather than a
# diversification one.
HIGH_CORRELATION = 0.8


@dataclass(frozen=True)
class OverlapWarning:
    strategy_id: str
    against_id: str
    correlation: float
    same_market: bool
    detail: str

    def to_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}


def check_against_deployed(session: Session, strategy: Strategy) -> OverlapWarning | None:
    """The strongest overlap with anything already deployed, if any.

    Returns the worst offender rather than a list: the question at the
    deploy button is "is this a duplicate bet", and one clear example
    answers it better than a table nobody reads.
    """
    deployed = list(
        session.scalars(
            select(Strategy).where(
                Strategy.deployed.is_(True),
                Strategy.deleted_at.is_(None),
                Strategy.id != strategy.id,
            )
        )
    )
    if not deployed:
        return None

    candidate_returns = daily_returns(strategy)
    if candidate_returns is None:
        return None

    worst: OverlapWarning | None = None
    for other in deployed:
        other_returns = daily_returns(other)
        if other_returns is None:
            continue
        aligned = pd.concat([candidate_returns, other_returns], axis=1, join="inner").dropna()
        if len(aligned) < MIN_OVERLAP_DAYS:
            # Not enough shared days to tell: silence here means "cannot
            # say", which is why this returns None rather than zero.
            continue
        if aligned.iloc[:, 0].std() == 0 or aligned.iloc[:, 1].std() == 0:
            continue

        correlation = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        if correlation < HIGH_CORRELATION:
            continue
        if worst is not None and correlation <= worst.correlation:
            continue

        same_market = (strategy.asset, strategy.timeframe) == (other.asset, other.timeframe)
        worst = OverlapWarning(
            strategy_id=strategy.id,
            against_id=other.id,
            correlation=round(correlation, 2),
            same_market=same_market,
            detail=(
                f"This moves {correlation:.0%} in step with {other.id}, which is already deployed"
                + (f" on the same {strategy.asset} {strategy.timeframe}." if same_market else ".")
                + " Running both is closer to doubling one position than to diversifying —"
                " size them as one bet, or deploy only the stronger."
            ),
        )

    return worst
