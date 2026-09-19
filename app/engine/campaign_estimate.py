"""Campaign duration estimate — REQ-DASH-1.5.

The estimate is derived from the work the campaign will actually do, not
from the core count alone: number of backtests (Phase 0's population, plus
each tribe's children per generation, plus Phase 3's crossover children)
multiplied by the cost of one backtest, divided across cores.

`SECONDS_PER_BAR_PER_BACKTEST` is measured, not guessed: a full genome
evaluation over a 14,212-bar M1 series takes ~0.127s on this machine's
single core, i.e. ~9e-6 s/bar. It is a coefficient to re-measure when the
backtester's hot path changes, not a constant of nature.

Pure function so the Dashboard's "before you confirm launch" estimate can
be unit-tested without spinning up a real campaign.
"""

from __future__ import annotations

SECONDS_PER_BAR_PER_BACKTEST = 9.0e-6
# Phases 1 and 2 each breed `children_per_generation` candidates per
# generation; the breeding pool is capped (phases.py DEFAULT_MAX_BREEDING_
# POPULATION) and 80% of it is replaced each generation (elite fraction 0.2).
TRIBES = 2
CHILD_FRACTION = 0.8
# Parallel efficiency is never 1.0 — pickling, pool startup and the
# sequential Phase 4 all eat into it.
PARALLEL_EFFICIENCY = 0.75
ESTIMATE_BAND = 0.4  # +/- 40% around the point estimate


def campaign_backtest_count(population_size: int, generations: int, breeding_population: int, phase3_children: int) -> int:
    """Total genome evaluations a campaign performs, across all 5 phases."""
    children_per_generation = max(1, int(breeding_population * CHILD_FRACTION))
    return (
        population_size  # Phase 0
        + TRIBES * generations * children_per_generation  # Phases 1 and 2
        + phase3_children  # Phase 3
        + phase3_children  # Phase 4 re-runs survivors against OOS
    )


def estimate_campaign_minutes(
    cpu_cores: int,
    population_size: int = 300,
    generations: int = 5,
    bars: int = 20_000,
    breeding_population: int = 200,
    phase3_children: int = 80,
) -> tuple[int, int]:
    """Return an inclusive (min, max) minute range for this configuration.

    Defaults match the MEDIUM POWER preset so a caller with no settings in
    hand still gets a sane number rather than a fixed 15-60 placeholder.
    """
    cores = max(1, cpu_cores)
    backtests = campaign_backtest_count(population_size, generations, breeding_population, phase3_children)
    serial_seconds = backtests * bars * SECONDS_PER_BAR_PER_BACKTEST
    parallel_seconds = serial_seconds / (cores * PARALLEL_EFFICIENCY)

    minutes = parallel_seconds / 60
    low = max(1, round(minutes * (1 - ESTIMATE_BAND)))
    high = max(low + 1, round(minutes * (1 + ESTIMATE_BAND)))
    return low, high
