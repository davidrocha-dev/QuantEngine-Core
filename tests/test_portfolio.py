import json
from datetime import datetime, timedelta, timezone

import numpy as np
from pytest import approx

from app.engine.portfolio import (
    MIN_OVERLAP_DAYS,
    combined_equity_curve,
    compute_health,
    daily_returns,
    pairwise_correlations,
)
from app.models import Strategy


def equity_curve(returns: list[float], start: datetime | None = None, initial: float = 10_000.0) -> str:
    start = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
    equity, points = initial, [{"t": start.isoformat(), "equity": initial}]
    for day, value in enumerate(returns, start=1):
        equity *= 1 + value
        points.append({"t": (start + timedelta(days=day)).isoformat(), "equity": round(equity, 2)})
    return json.dumps(points)


def make_strategy(strategy_id: str, returns: list[float] | None = None, **overrides) -> Strategy:
    defaults = dict(
        id=strategy_id, data_source="dukascopy", asset="EURUSD", timeframe="M15",
        archetype="MULTI_SIGNAL", genome_json="{}", score=70,
        equity_oos=equity_curve(returns) if returns is not None else None,
    )
    defaults.update(overrides)
    return Strategy(**defaults)


def test_daily_returns_come_from_the_stored_oos_curve():
    strategy = make_strategy("#A", [0.01, -0.005, 0.02])
    returns = daily_returns(strategy)

    assert returns is not None
    assert len(returns) == 3
    assert returns.iloc[0] == approx(0.01)


def test_a_strategy_without_an_equity_curve_has_no_returns():
    assert daily_returns(make_strategy("#A")) is None


def test_identical_strategies_are_perfectly_correlated():
    series = [0.01, -0.02, 0.015, -0.005] * 5
    correlations = pairwise_correlations([make_strategy("#A", series), make_strategy("#B", series)])

    assert correlations[("#A", "#B")] > 0.99


def test_mirrored_strategies_are_negatively_correlated():
    series = [0.01, -0.02, 0.015, -0.005] * 5
    mirrored = [-value for value in series]
    correlations = pairwise_correlations([make_strategy("#A", series), make_strategy("#B", mirrored)])

    assert correlations[("#A", "#B")] < -0.99


def test_pairs_without_enough_overlap_are_unknown_not_zero():
    # "We cannot tell" and "they are independent" are opposite conclusions.
    short = [0.01] * (MIN_OVERLAP_DAYS - 3)
    correlations = pairwise_correlations([make_strategy("#A", short), make_strategy("#B", short)])

    assert correlations == {}


def test_health_of_an_empty_portfolio_says_what_to_do():
    health = compute_health([])

    assert health.strategy_count == 0
    assert health.diversity_grade == "F"
    assert health.avg_correlation is None
    assert health.recommendations


def test_correlated_single_symbol_portfolio_grades_badly_and_says_why():
    series = [0.01, -0.02, 0.015, -0.005] * 6
    members = [make_strategy(f"#{i}", series, asset="EURUSD", timeframe="M15") for i in range(3)]

    health = compute_health(members)

    assert health.avg_correlation is not None and health.avg_correlation > 0.9
    assert health.diversity_grade in ("D", "E", "F")
    assert any("OVERLAP" in r for r in health.recommendations)
    assert any("EURUSD" in r for r in health.recommendations)


def test_diverse_portfolio_grades_better_than_a_concentrated_one():
    rng = np.random.default_rng(3)
    independent = [list(rng.normal(0.001, 0.01, 40)) for _ in range(5)]
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "AUDUSD"]
    timeframes = ["M15", "H1", "H4", "M30", "H1"]
    archetypes = ["MULTI_SIGNAL", "MEAN_REVERTER", "PATTERN_SPOTTER", "MEAN_REVERTER", "MULTI_SIGNAL"]
    diverse = [
        make_strategy(f"#D{i}", independent[i], asset=symbols[i], timeframe=timeframes[i], archetype=archetypes[i])
        for i in range(5)
    ]
    concentrated = [make_strategy(f"#C{i}", independent[0], asset="EURUSD", timeframe="M15") for i in range(3)]

    assert compute_health(diverse).diversity_score > compute_health(concentrated).diversity_score


def test_a_small_portfolio_is_told_to_grow():
    health = compute_health([make_strategy("#A", [0.01, 0.02, -0.01] * 5)])
    assert any("GROW TO 5+" in r for r in health.recommendations)


def test_combined_equity_is_the_equal_weighted_blend():
    up = [0.02] * 20
    down = [-0.01] * 20
    combined = combined_equity_curve([make_strategy("#A", up), make_strategy("#B", down)])

    assert combined["curve"]
    # Equal weights: +2% and -1% a day average +0.5% a day, compounded.
    expected = ((1.005) ** 20 - 1) * 100
    assert abs(combined["return_pct"] - expected) < 1.0
    assert combined["max_dd_pct"] >= 0


def test_combined_equity_is_empty_without_member_curves():
    combined = combined_equity_curve([make_strategy("#A")])
    assert combined == {"curve": [], "return_pct": None, "max_dd_pct": None, "sharpe": None}


def test_a_member_that_did_not_exist_yet_is_not_counted_as_a_flat_day():
    """The combined curve used to fill missing days with 0.0 and divide by
    the full member count, so a member bred on a different window dragged
    every other member's day towards zero.

    Measured: two strategies each returning +81.50% over their own 200
    days, run back to back, combined to +81.58% — the second one's whole
    contribution erased. The correlation path in this same module already
    refuses that substitution, and the module docstring says why.
    """
    import json

    import pandas as pd

    from app.engine.portfolio import combined_equity_curve

    def member(strategy_id: str, start: str, daily_pct: float, days: int = 200) -> Strategy:
        equity, value = [], 10_000.0
        for stamp in pd.date_range(start, periods=days, freq="1D", tz="UTC"):
            value *= 1 + daily_pct
            equity.append({"t": stamp.isoformat(), "equity": round(value, 2)})
        return Strategy(
            id=strategy_id, data_source="dukascopy", asset="XAUUSD", timeframe="M15",
            genome_json="{}", equity_oos=json.dumps(equity),
        )

    alone = combined_equity_curve([member("#A", "2026-01-01", 0.003)])
    back_to_back = combined_equity_curve(
        [member("#A", "2026-01-01", 0.003), member("#B", "2026-07-20", 0.003)]
    )
    overlapping = combined_equity_curve(
        [member("#A", "2026-01-01", 0.003), member("#B", "2026-01-01", 0.003)]
    )

    # Two good runs one after the other compound; they do not average away.
    assert back_to_back["return_pct"] > alone["return_pct"] * 2
    # Two identical members held at the same time are still equal weight.
    assert overlapping["return_pct"] == approx(alone["return_pct"], rel=1e-6)
