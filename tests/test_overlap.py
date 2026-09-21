"""Deploying the same bet twice — REQ-PORT-2.

Correlation was only ever checked inside a portfolio, which is the one
place somebody had already decided to think about diversification. The
deploy button never asked, so three strategies that are one idea in
different clothes could go live one after another: three bots on three
screens, one position's worth of risk taken three times.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.db import Base, make_engine
from app.engine.overlap import HIGH_CORRELATION, check_against_deployed
from app.models import Strategy


@pytest.fixture
def session(tmp_path: Path) -> Session:
    engine = make_engine(tmp_path / "vault.db")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def equity_curve(daily_moves: list[float], start_equity: float = 10_000.0) -> str:
    """One point per day, compounding the given daily returns."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    equity = start_equity
    points = []
    for index, move in enumerate(daily_moves):
        equity *= 1 + move
        points.append({"t": (start + timedelta(days=index)).isoformat(), "equity": round(equity, 2)})
    return json.dumps(points)


def make_strategy(
    session: Session, strategy_id: str, moves: list[float], deployed: bool = False, asset: str = "EURUSD"
) -> Strategy:
    row = Strategy(
        id=strategy_id, data_source="dukascopy", asset=asset, timeframe="M15",
        genome_json="{}", trades=120, return_pct=18.0, score=70, grade="GOOD",
        deployed=deployed, equity_oos=equity_curve(moves),
    )
    session.add(row)
    session.commit()
    return row


# Forty days of moves, and the same forty with a little noise on top.
BASE = [0.01, -0.005, 0.008, -0.002, 0.012, -0.009, 0.004, 0.006] * 5
NEARLY_THE_SAME = [move + (0.0004 if index % 3 else -0.0003) for index, move in enumerate(BASE)]
OPPOSITE = [-move for move in BASE]
# Perturbed with noise rather than rescaled, so it really is less
# correlated (about 0.83) and not merely smaller.
_NOISE = [
    0.00021, 0.00816, 0.00735, -0.00306, -0.00179, -0.00316, 0.00342, -0.00034,
    -0.00412, 0.00551, -0.00223, 0.00618, -0.00505, 0.00187, 0.00402, -0.00661,
    0.00294, -0.00518, 0.00631, -0.00147, 0.00233, -0.00389, 0.00512, -0.00276,
    -0.00604, 0.00358, 0.00119, -0.00443, 0.00267, 0.00594, -0.00321, 0.00186,
    0.00455, -0.00572, 0.00208, -0.00239, 0.00374, 0.00641, -0.00415, 0.00163,
]
LOOSELY_RELATED = [move + _NOISE[index] for index, move in enumerate(BASE)]


def test_nothing_deployed_means_nothing_to_collide_with(session):
    candidate = make_strategy(session, "#NEW", BASE)

    assert check_against_deployed(session, candidate) is None


def test_a_near_duplicate_of_a_running_bot_is_caught(session):
    make_strategy(session, "#RUNNING", BASE, deployed=True)
    candidate = make_strategy(session, "#TWIN", NEARLY_THE_SAME)

    warning = check_against_deployed(session, candidate)

    assert warning is not None
    assert warning.against_id == "#RUNNING"
    assert warning.correlation >= HIGH_CORRELATION
    assert warning.same_market is True
    assert "doubling one position" in warning.detail


def test_something_genuinely_different_passes(session):
    make_strategy(session, "#RUNNING", BASE, deployed=True)
    candidate = make_strategy(session, "#OTHER", OPPOSITE)

    assert check_against_deployed(session, candidate) is None


def test_a_strategy_is_not_compared_against_itself(session):
    candidate = make_strategy(session, "#SELF", BASE, deployed=True)

    assert check_against_deployed(session, candidate) is None


def test_the_worst_offender_is_the_one_reported(session):
    # One clear example answers "is this a duplicate bet" better than a
    # table nobody reads. LOOSE is perturbed rather than scaled, because
    # correlation is scale-invariant: half of BASE plus a constant is
    # still perfectly correlated with BASE.
    make_strategy(session, "#LOOSE", LOOSELY_RELATED, deployed=True)
    make_strategy(session, "#TIGHT", NEARLY_THE_SAME, deployed=True)
    candidate = make_strategy(session, "#NEW", BASE)

    warning = check_against_deployed(session, candidate)

    assert warning is not None and warning.against_id == "#TIGHT"


def test_too_little_shared_history_says_nothing_rather_than_zero(session):
    # "We cannot tell" is not the same as "these are unrelated", and
    # reporting the second would be worse than staying quiet.
    make_strategy(session, "#RUNNING", BASE[:3], deployed=True)
    candidate = make_strategy(session, "#NEW", BASE[:3])

    assert check_against_deployed(session, candidate) is None


def test_a_correlated_pair_on_different_markets_is_still_a_warning(session):
    # Two markets that happen to move together are still one bet.
    make_strategy(session, "#RUNNING", BASE, deployed=True, asset="EURUSD")
    candidate = make_strategy(session, "#NEW", NEARLY_THE_SAME, asset="GBPUSD")

    warning = check_against_deployed(session, candidate)

    assert warning is not None
    assert warning.same_market is False


def test_a_strategy_with_no_equity_curve_cannot_be_judged(session):
    make_strategy(session, "#RUNNING", BASE, deployed=True)
    candidate = Strategy(
        id="#NOCURVE", data_source="dukascopy", asset="EURUSD", timeframe="M15",
        genome_json="{}", trades=120, score=70, grade="GOOD",
    )
    session.add(candidate)
    session.commit()

    assert check_against_deployed(session, candidate) is None
