"""Portfolio analytics — REQ-PORT-1.

A portfolio's risk is not the sum of its members' risks: two strategies
that lose money on the same days are one strategy with twice the size.
Everything here exists to make that visible before capital is committed.

Correlation is computed from the stored out-of-sample equity curves
(`strategies.equity_oos`, written by the vault writer), resampled onto a
common daily grid so strategies on different timeframes are comparable.
Members bred on non-overlapping history have no shared days, and their
correlation is reported as unknown rather than as zero — "we cannot tell"
and "they are independent" are opposite conclusions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from app.models import Strategy

# Fewer shared days than this makes a correlation coefficient noise.
MIN_OVERLAP_DAYS = 10

# REQ-PORT-1.3 — Diversity Grade bands, worst sub-score decides the grade.
GRADE_THRESHOLDS = ((90, "A"), (75, "B"), (60, "C"), (45, "D"), (30, "E"))


@dataclass
class DiversityDimension:
    name: str
    score: float  # 0-100
    detail: str


@dataclass
class PortfolioHealth:
    strategy_count: int
    avg_correlation: float | None
    max_correlation: float | None
    diversity_grade: str
    diversity_score: float
    dimensions: list[DiversityDimension] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    combined_equity: list[dict] = field(default_factory=list)
    combined_return_pct: float | None = None
    combined_max_dd_pct: float | None = None
    combined_sharpe: float | None = None

    def to_dict(self) -> dict:
        return {
            "strategy_count": self.strategy_count,
            "avg_correlation": self.avg_correlation,
            "max_correlation": self.max_correlation,
            "diversity_grade": self.diversity_grade,
            "diversity_score": self.diversity_score,
            "dimensions": [vars(d) for d in self.dimensions],
            "recommendations": self.recommendations,
            "combined_equity": self.combined_equity,
            "combined_return_pct": self.combined_return_pct,
            "combined_max_dd_pct": self.combined_max_dd_pct,
            "combined_sharpe": self.combined_sharpe,
        }


def daily_returns(strategy: Strategy) -> pd.Series | None:
    """Daily percentage returns from a strategy's stored OOS equity curve."""
    if not strategy.equity_oos:
        return None
    try:
        points = json.loads(strategy.equity_oos)
    except (json.JSONDecodeError, TypeError):
        return None
    if not points:
        return None

    series = pd.Series(
        [float(point["equity"]) for point in points],
        index=pd.DatetimeIndex([point["t"] for point in points]),
    )
    # Equity is a step function between trades; the last value of each day
    # is that day's closing equity.
    daily = series.resample("1D").last().dropna()
    returns = daily.pct_change().dropna()
    return returns if len(returns) >= 2 else None


def pairwise_correlations(strategies: list[Strategy]) -> dict[tuple[str, str], float]:
    """Correlation for every pair with enough overlapping days to judge."""
    returns = {s.id: daily_returns(s) for s in strategies}
    correlations: dict[tuple[str, str], float] = {}

    for first, second in combinations(strategies, 2):
        left, right = returns.get(first.id), returns.get(second.id)
        if left is None or right is None:
            continue
        aligned = pd.concat([left, right], axis=1, join="inner").dropna()
        if len(aligned) < MIN_OVERLAP_DAYS:
            continue
        if aligned.iloc[:, 0].std() == 0 or aligned.iloc[:, 1].std() == 0:
            continue
        correlations[(first.id, second.id)] = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))

    return correlations


def _spread_score(values: list[str]) -> tuple[float, str]:
    """How evenly a portfolio is spread across one dimension.

    100 means every member differs; it falls towards 0 as one value
    dominates. Concentration, not variety, is what hurts: five strategies
    on three symbols with one symbol holding three of them is worse than
    five on five.
    """
    if not values:
        return 0.0, "no members"
    counts = pd.Series(values).value_counts()
    dominant_share = counts.iloc[0] / len(values)
    score = round(100 * (1 - (dominant_share - 1 / len(counts)) / max(1 - 1 / len(counts), 1e-9)), 1) if len(counts) > 1 else 0.0
    detail = f"{len(counts)} distinct, {counts.index[0]} holds {counts.iloc[0]}/{len(values)}"
    return max(0.0, min(100.0, score)), detail


def _correlation_score(correlations: dict[tuple[str, str], float]) -> tuple[float, str]:
    if not correlations:
        return 50.0, "not enough overlapping history to measure"
    average = float(np.mean(list(correlations.values())))
    # +1.0 correlated -> 0, uncorrelated -> 100, negatively correlated is
    # capped at 100 rather than rewarded beyond it.
    score = round(min(100.0, max(0.0, (1 - average) * 100)), 1)
    return score, f"average {average:+.2f} across {len(correlations)} pairs"


def _size_score(count: int) -> tuple[float, str]:
    # Five members is where diversification starts to do real work; one
    # member is a bet, not a portfolio.
    score = round(min(100.0, 100 * count / 5), 1)
    return score, f"{count} strategies"


def compute_health(strategies: list[Strategy]) -> PortfolioHealth:
    if not strategies:
        return PortfolioHealth(
            strategy_count=0,
            avg_correlation=None,
            max_correlation=None,
            diversity_grade="F",
            diversity_score=0.0,
            recommendations=["Add strategies from the Vault to build this portfolio."],
        )

    correlations = pairwise_correlations(strategies)
    correlation_values = list(correlations.values())
    avg_correlation = float(np.mean(correlation_values)) if correlation_values else None
    max_correlation = float(np.max(correlation_values)) if correlation_values else None

    correlation_score, correlation_detail = _correlation_score(correlations)
    size_score, size_detail = _size_score(len(strategies))
    symbol_score, symbol_detail = _spread_score([s.asset for s in strategies])
    timeframe_score, timeframe_detail = _spread_score([s.timeframe for s in strategies])
    archetype_score, archetype_detail = _spread_score([s.archetype or "UNKNOWN" for s in strategies])

    dimensions = [
        DiversityDimension("correlation", correlation_score, correlation_detail),
        DiversityDimension("size", size_score, size_detail),
        DiversityDimension("symbol", symbol_score, symbol_detail),
        DiversityDimension("timeframe", timeframe_score, timeframe_detail),
        DiversityDimension("archetype", archetype_score, archetype_detail),
    ]

    # The worst dimension carries more weight than the average: a portfolio
    # diversified in every way except that every member trades one symbol
    # is not diversified.
    scores = [d.score for d in dimensions]
    diversity_score = round(0.6 * float(np.mean(scores)) + 0.4 * min(scores), 1)

    combined = combined_equity_curve(strategies)
    return PortfolioHealth(
        strategy_count=len(strategies),
        avg_correlation=round(avg_correlation, 2) if avg_correlation is not None else None,
        max_correlation=round(max_correlation, 2) if max_correlation is not None else None,
        diversity_grade=_grade(diversity_score),
        diversity_score=diversity_score,
        dimensions=dimensions,
        recommendations=_recommendations(strategies, dimensions, max_correlation),
        combined_equity=combined["curve"],
        combined_return_pct=combined["return_pct"],
        combined_max_dd_pct=combined["max_dd_pct"],
        combined_sharpe=combined["sharpe"],
    )


def _grade(score: float) -> str:
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"


def _recommendations(
    strategies: list[Strategy], dimensions: list[DiversityDimension], max_correlation: float | None
) -> list[str]:
    """REQ-PORT-1.4 — each one names the specific thing to change."""
    advice: list[str] = []
    by_name = {d.name: d for d in dimensions}

    if len(strategies) < 5:
        advice.append(f"GROW TO 5+ STRATEGIES (currently {len(strategies)})")
    if max_correlation is not None and max_correlation > 0.7:
        advice.append(f"REDUCE OVERLAP — two members move together ({max_correlation:+.2f})")

    symbols = pd.Series([s.asset for s in strategies]).value_counts()
    if len(strategies) >= 3 and symbols.iloc[0] / len(strategies) > 0.5:
        advice.append(f"REDUCE {symbols.index[0]} EXPOSURE ({symbols.iloc[0]} of {len(strategies)})")

    if by_name["timeframe"].score < 40 and len(strategies) >= 3:
        advice.append("SPREAD ACROSS TIMEFRAMES")
    if by_name["archetype"].score < 40 and len(strategies) >= 3:
        advice.append("MIX ARCHETYPES — members share one mechanism")

    return advice


def combined_equity_curve(strategies: list[Strategy], initial_balance: float = 10_000.0) -> dict:
    """Equal-weight combined equity of the members' OOS curves.

    This is an aggregation of separately-simulated strategies, not a single
    simulated account: it has no symbol lock and no aggregate risk cap, so
    it says how the members' returns combine, not what a broker would have
    allowed (that is TRUE SIM — REQ-PORT-2, not built yet).
    """
    series = [r for r in (daily_returns(s) for s in strategies) if r is not None]
    if not series:
        return {"curve": [], "return_pct": None, "max_dd_pct": None, "sharpe": None}

    frame = pd.concat(series, axis=1, join="outer").sort_index().fillna(0.0)
    portfolio_returns = frame.mean(axis=1)
    equity = initial_balance * (1 + portfolio_returns).cumprod()

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    return {
        "curve": [
            {"t": timestamp.isoformat(), "equity": round(float(value), 2)}
            for timestamp, value in equity.items()
        ],
        "return_pct": round(float((equity.iloc[-1] - initial_balance) / initial_balance * 100), 2),
        "max_dd_pct": round(float(-drawdown.min() * 100), 2),
        # Annualised from daily portfolio returns (252 trading days).
        "sharpe": (
            round(float(portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)), 2)
            if len(portfolio_returns) > 2 and portfolio_returns.std() > 0
            else None
        ),
    }
