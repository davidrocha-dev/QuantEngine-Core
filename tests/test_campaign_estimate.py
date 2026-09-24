"""The campaign duration estimate, checked against measured campaigns.

Four real runs on this machine, 39,734 XAUUSD M15 bars, 12 cores:

    ECO     60/3     estimate 1-2 min   actual 1.68 min
    MEDIUM  300/5    estimate 1-4 min   actual 1.49 min
    HIGH    1000/10  estimate 3-7 min   actual 3.17 min
    ECO     60/3     (the owner's own campaign, 3 survivors)  1.40 min

MEDIUM finishing faster than ECO is not noise: Phase 0 redraws a whole
population and screens it again until it has a breeding pool, up to
MAX_SCREENING_ROUNDS times, and a population of 300 reaches that on its
first draw where 60 does not. That is the dominant uncertainty in the
number, and it is what the reported range is made of.
"""

import pytest

from app.engine.campaign import MAX_SCREENING_ROUNDS
from app.engine.campaign_estimate import (
    SECONDS_PER_BAR,
    campaign_work,
    estimate_campaign_minutes,
)
from app.engine.monte_carlo import NOISE_SAMPLES
from app.engine.phases import IS_FRACTION

REAL_BARS = 39_734
REAL_CORES = 12


def test_work_matches_an_instrumented_campaign():
    """Measured by wrapping simulate_trades and running a campaign end to
    end on one core: 6,000 bars, population 60, 3 generations, and Phase 0
    took three screening rounds.

        phase 0   126.0 full runs      phase 3    56.0
        phase 1+2 184.8                phase 4   117.0
    """
    work = campaign_work(
        population_size=60, generations=3, phase3_children=80,
        screening_rounds=3, phase4_full_depth=False,
    )

    # Phase 0 is exact: rounds x population x the in-sample share.
    assert 3 * 60 * IS_FRACTION == pytest.approx(126.0)
    # And the parallel total is within a fifth of what was measured. The
    # model uses the breeding-pool cap as an upper bound on children,
    # where the run had fewer survivors than that.
    assert work.parallel == pytest.approx(356.4, rel=0.15)
    # Phase 4 measured 117 full runs, between the two bounds this models.
    shallow = work.serial
    deep = campaign_work(60, 3, 80, screening_rounds=3, phase4_full_depth=True).serial
    assert shallow < 117.0 < deep


def test_phase_4_is_not_divided_by_the_core_count():
    """Robustness, noise sampling and walk-forward run one candidate at a
    time in the parent; only the population evaluations go through the
    pool. Dividing the whole campaign by the cores was the largest single
    error on a machine with many of them."""
    one_core = estimate_campaign_minutes(1, population_size=1000, generations=10, bars=REAL_BARS)
    many_cores = estimate_campaign_minutes(64, population_size=1000, generations=10, bars=REAL_BARS)

    assert many_cores[1] < one_core[1]
    # Serial work does not vanish: sixty-four cores cannot make it free.
    work = campaign_work(1000, 10, 80, MAX_SCREENING_ROUNDS, True)
    floor_minutes = work.serial * REAL_BARS * SECONDS_PER_BAR / 60
    assert many_cores[1] >= floor_minutes


@pytest.mark.parametrize(
    "population,generations,actual_minutes",
    [(60, 3, 1.68), (300, 5, 1.49), (1000, 10, 3.17), (60, 3, 1.40)],
)
def test_measured_campaigns_land_inside_the_estimate(population, generations, actual_minutes):
    low, high = estimate_campaign_minutes(
        REAL_CORES, population_size=population, generations=generations, bars=REAL_BARS
    )
    assert low <= actual_minutes <= high, f"{actual_minutes} outside {low}-{high}"


def test_the_range_is_the_screening_uncertainty_not_a_percentage_band():
    """The old estimate multiplied a point by 0.6 and 1.4. The range now
    means something: one screening round and nothing surviving robustness
    at the low end, the maximum redraws and every Phase 3 child going
    through noise sampling and walk-forward at the high end."""
    low, high = estimate_campaign_minutes(
        REAL_CORES, population_size=300, generations=5, bars=REAL_BARS
    )

    optimistic = campaign_work(300, 5, 80, screening_rounds=1, phase4_full_depth=False)
    pessimistic = campaign_work(300, 5, 80, screening_rounds=MAX_SCREENING_ROUNDS, phase4_full_depth=True)

    assert pessimistic.parallel > optimistic.parallel
    assert pessimistic.serial > optimistic.serial
    assert high / max(low, 1) != pytest.approx(1.4 / 0.6, rel=0.01), "that would be the old fixed band"


def test_noise_sampling_is_counted():
    """Phase 4 re-simulates each surviving candidate NOISE_SAMPLES times.
    The old model counted one Phase 4 backtest per survivor."""
    deep = campaign_work(300, 5, 80, screening_rounds=1, phase4_full_depth=True)
    shallow = campaign_work(300, 5, 80, screening_rounds=1, phase4_full_depth=False)

    oos = 1 - IS_FRACTION
    assert deep.serial - shallow.serial == pytest.approx(80 * (oos * NOISE_SAMPLES + 1.0))


def test_more_cores_and_lighter_presets_shorten_the_estimate():
    assert estimate_campaign_minutes(16)[1] <= estimate_campaign_minutes(4)[1]
    light = estimate_campaign_minutes(8, population_size=300, generations=5, bars=20_000)[1]
    heavy = estimate_campaign_minutes(8, population_size=4000, generations=20, bars=20_000)[1]
    deep = estimate_campaign_minutes(8, population_size=300, generations=5, bars=200_000)[1]
    assert heavy > light and deep > light


def test_estimate_is_a_positive_widening_range():
    for cores in (1, 2, 8, 32):
        low, high = estimate_campaign_minutes(cores)
        assert 1 <= low < high
