"""PAUSE and SKIP — REQ-BREED-1.2.

Both buttons existed and both did nothing: the endpoints behind them
answered "not supported" while the campaign carried on regardless. These
tests are about them doing what the label says, and about doing it
without losing the generation that was in flight.
"""

import pandas as pd
import pytest

from app.engine.control import CampaignControl, CampaignPaused
from app.engine.phases import CandidateResult, run_generational_phase
from app.genome.gene_pool import GenePoolConfig
from app.genome.genome import random_genome
from app.engine.metrics import TradeStats


def stats(profit_factor: float = 1.5) -> TradeStats:
    return TradeStats(
        trade_count=120, profit_factor=profit_factor, win_rate=0.5, sharpe_ratio=1.0,
        max_dd_pct=10.0, net_profit_pct=20.0, cagr_pct=15.0, payoff_ratio=1.5,
        icir=1.0, avg_hold_hours=4.0,
    )


@pytest.fixture
def seed():
    pool = GenePoolConfig()
    return [CandidateResult(genome=random_genome(pool), stats=stats()) for _ in range(6)], pool


@pytest.fixture
def bars():
    # Enough to backtest against, small enough to evolve quickly.
    index = pd.date_range("2026-01-01", periods=400, freq="15min", tz="UTC")
    close = pd.Series(range(400), dtype=float) * 0.0001 + 1.08
    return pd.DataFrame(
        {
            "timestamp": index,
            "open": close, "high": close + 0.0005, "low": close - 0.0005, "close": close,
            "volume": [100.0] * 400,
        }
    )


def test_a_control_with_nothing_asked_of_it_does_nothing():
    control = CampaignControl()

    control.checkpoint()  # must not raise
    assert control.take_skip() is False


def test_a_pause_raises_at_the_next_checkpoint():
    control = CampaignControl()
    control.request_pause()

    with pytest.raises(CampaignPaused):
        control.checkpoint()


def test_a_pause_is_not_consumed_by_reading_it():
    # It is cleared once the campaign has actually stopped, so a pause
    # arriving during teardown cannot be swallowed.
    control = CampaignControl()
    control.request_pause()

    with pytest.raises(CampaignPaused):
        control.checkpoint()
    with pytest.raises(CampaignPaused):
        control.checkpoint()

    control.clear()
    control.checkpoint()


def test_a_skip_applies_once_and_only_once():
    # One click must end one phase, not every phase that follows it.
    control = CampaignControl()
    control.request_skip()

    assert control.take_skip() is True
    assert control.take_skip() is False


def test_pausing_stops_the_phase_after_the_generation_that_was_running(seed, bars):
    survivors, pool = seed
    control = CampaignControl()
    generations_seen = []

    def on_generation(generation, population):
        generations_seen.append(generation)
        control.request_pause()  # asked for during the first generation

    with pytest.raises(CampaignPaused):
        run_generational_phase(
            survivors, lambda s: s.profit_factor, "A", pool, bars, generations=5, top_n=3,
            on_generation_complete=on_generation, control=control,
            initial_balance=10_000.0, risk_pct=0.01,
        )

    # It finished the generation it was in, and did not start another.
    assert generations_seen == [0]


def test_the_checkpoint_is_written_before_the_pause_takes_effect(seed, bars):
    """This is what makes a pause free rather than destructive.

    The callback that writes the checkpoint runs first; only then is the
    flag read. Reverse the order and a resume would restart from a
    population that no longer matches what was saved.
    """
    survivors, pool = seed
    control = CampaignControl()
    checkpointed = []

    def on_generation(generation, population):
        checkpointed.append((generation, len(population)))
        control.request_pause()

    with pytest.raises(CampaignPaused):
        run_generational_phase(
            survivors, lambda s: s.profit_factor, "A", pool, bars, generations=5, top_n=3,
            on_generation_complete=on_generation, control=control,
            initial_balance=10_000.0, risk_pct=0.01,
        )

    assert checkpointed and checkpointed[0][1] > 0


def test_skipping_ends_the_phase_early_but_keeps_its_survivors(seed, bars):
    survivors, pool = seed
    control = CampaignControl()
    generations_seen = []

    def on_generation(generation, population):
        generations_seen.append(generation)
        control.request_skip()

    outcome = run_generational_phase(
        survivors, lambda s: s.profit_factor, "A", pool, bars, generations=6, top_n=3,
        on_generation_complete=on_generation, control=control,
        initial_balance=10_000.0, risk_pct=0.01,
    )

    assert generations_seen == [0]          # stopped after the first
    assert outcome.passed is True           # a skip is not a failure
    assert len(outcome.survivors) > 0       # and its work carries forward


def test_without_a_control_the_phase_runs_every_generation(seed, bars):
    survivors, pool = seed
    generations_seen = []

    run_generational_phase(
        survivors, lambda s: s.profit_factor, "A", pool, bars, generations=3, top_n=3,
        on_generation_complete=lambda generation, population: generations_seen.append(generation),
        initial_balance=10_000.0, risk_pct=0.01,
    )

    assert generations_seen == [0, 1, 2]
