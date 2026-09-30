"""REASSESS has to give the same answer twice.

Measured before the fix, on a real strategy in a real Vault, six
identical runs — same genome, same bars, same settings, nothing changed
in between — produced scores of 64, 64, 77, 78, 78 and 80, robustness
from 36.6% to 100%, and a `still_passes` verdict that came back False
once and True five times. Two sources of randomness were being drawn
unseeded on the decision path: the noise perturbation and the bootstrap.
"""

import json
from types import SimpleNamespace

import pytest

from app.engine.reassess import reassess_strategy, refresh_strategy
from app.genome.gene_pool import GenePoolConfig
from app.genome.genome import GeneChoice, Genome
from app.models import Strategy
from tests.helpers import make_synthetic_ohlc


def make_config() -> SimpleNamespace:
    return SimpleNamespace(
        initial_balance=10_000.0,
        point_size=0.0001,
        # Zero cost: this fixture is about determinism, and a genome
        # that only just breaks even would bail out before scoring.
        cost_points=0.0,
        commission_per_lot=0.0,
        contract_size=100_000.0,
        risk_pct=0.01,
    )


def make_strategy(genome: Genome) -> Strategy:
    return Strategy(
        id="#STABLE",
        data_source="dukascopy",
        asset="EURUSD",
        timeframe="H1",
        score=60,
        grade="GOOD",
        archetype="MULTI_SIGNAL",
        genome_json=json.dumps(genome.to_dict()),
    )


def trading_genome() -> Genome:
    return Genome(
        signal=GeneChoice("MACD", {"fast": 12, "slow": 26, "signal": 9}),
        bias=None,
        filters=[],
        execution=GeneChoice("MARKET2", {"confirm_bars": 1}),
        management=None,
        # This geometry is deliberately noise-sensitive: on the fixture
        # below a 1-bar entry displacement costs 0.0% and a 2-bar one
        # costs 18.9%, both inside the 25% band the robustness score
        # is graded on. Without that the test would pass whatever the
        # randomness did, because the component saturates outside it.
        stop_atr_mult=2.0,
        target_atr_mult=4.0,
    )


@pytest.fixture(scope="module")
def bars():
    # Long enough and trending enough to clear MIN_OOS_TRADES, so the
    # assessment reaches the scoring path instead of bailing out early.
    return make_synthetic_ohlc(30_000, seed=47, drift=0.00015, volatility=0.0035)


def test_two_identical_reassessments_agree(bars):
    pool = GenePoolConfig()
    config = make_config()

    first = reassess_strategy(make_strategy(trading_genome()), bars, pool, config)
    second = reassess_strategy(make_strategy(trading_genome()), bars, pool, config)

    assert first.score is not None, f"the run bailed out early: {first.failed_gate}"
    assert first.score == second.score
    assert first.robust_pct == second.robust_pct
    assert first.still_passes == second.still_passes
    assert first.persistence_pct == second.persistence_pct


def test_the_stored_robustness_carries_its_range(bars):
    pool = GenePoolConfig()
    strategy = make_strategy(trading_genome())

    result = reassess_strategy(strategy, bars, pool, make_config())

    assert result.score is not None, f"the run bailed out early: {result.failed_gate}"
    assert strategy.robust_low_pct is not None
    assert strategy.robust_high_pct is not None
    assert strategy.robust_low_pct <= strategy.robust_pct <= strategy.robust_high_pct


def test_refresh_is_deterministic_too(bars):
    pool = GenePoolConfig()
    config = make_config()

    first = refresh_strategy(make_strategy(trading_genome()), bars, pool, config)
    second = refresh_strategy(make_strategy(trading_genome()), bars, pool, config)

    assert first.score == second.score
    assert first.return_pct == second.return_pct


def test_refresh_measures_the_campaigns_windows_and_its_holdout():
    """REASSESS used to split the whole history 70/30, so its out-of-sample
    window swallowed the final holdout and part of the search's own
    in-sample period."""
    from app.engine.campaign import MIN_IS_BARS
    from app.engine.phases import split_is_oos, split_search_holdout

    df = make_synthetic_ohlc(6000, seed=4, drift=0.0003, volatility=0.002)
    strategy = make_strategy(trading_genome())
    refresh_strategy(strategy, df, GenePoolConfig(), make_config())

    search_df, holdout_df = split_search_holdout(df, MIN_IS_BARS)
    _, oos_df = split_is_oos(search_df)
    assert holdout_df is not None
    assert strategy.holdout_start == holdout_df["timestamp"].iloc[0].isoformat()
    assert strategy.holdout_trades is not None
    # The out-of-sample curve ends where the holdout begins.
    last_oos_point = json.loads(strategy.equity_oos)[-1]["t"]
    assert last_oos_point == oos_df["timestamp"].iloc[-1].isoformat()
