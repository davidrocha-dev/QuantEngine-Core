"""Campaign duration estimate — REQ-DASH-1.5.

The estimate is derived from the work a campaign actually does, measured
by instrumenting `simulate_trades` and running real campaigns end to end
on one core. What that measurement corrected:

  * **Screening repeats.** Phase 0 redraws a whole fresh population and
    screens it again, up to `MAX_SCREENING_ROUNDS` times, until the
    tribes have a pool worth evolving (campaign.py). The old model
    counted one round. Measured: three rounds on a synthetic series, and
    the count is data-dependent, which is most of the honest uncertainty
    in this number.
  * **Phase 4 was counted as one backtest per survivor.** It is the
    out-of-sample run, plus `NOISE_SAMPLES` re-simulations of it, plus
    the walk-forward windows, which together cover the history once more.
  * **Phase 4 is serial.** Only the population evaluations in Phases 0 to
    3 go through the process pool; robustness, walk-forward and the Vault
    write happen one candidate at a time in the parent. Dividing the
    whole campaign by the core count was the largest single error for a
    machine with many cores.
  * **Windows differ per phase.** A final holdout (`HOLDOUT_FRACTION`)
    is kept back from the search entirely. Of what remains, Phases 0 to 3
    run on the in-sample slice, Phase 4's baseline and noise runs on the
    out-of-sample remainder, walk-forward across the searched history;
    the holdout is run once per approved candidate.
  * **Phase 4 sees more than the hybrids.** The tribes' own elites
    (`PHASE_4_ELITES_PER_TRIBE` each) go to robustness alongside the Phase
    3 children that clear screening.

Work is counted in "full runs": one backtest over the whole series. A
Phase 0 evaluation is therefore 0.7 of a full run, and the totals below
are directly comparable to the measurements in `tests/`.

`SECONDS_PER_BAR` is a campaign-level coefficient, not the cost of a bare
backtest — it absorbs process-pool startup, the per-candidate statistics,
checkpointing and the Vault write. It is calibrated against a measured
campaign and is the first thing to re-measure when the hot path changes.

Pure functions, so the Dashboard's "before you confirm launch" estimate
can be unit-tested without spinning up a real campaign.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.engine.campaign import MAX_SCREENING_ROUNDS
from app.engine.monte_carlo import NOISE_SAMPLES
from app.engine.phases import (
    DEFAULT_MAX_BREEDING_POPULATION,
    HOLDOUT_FRACTION,
    IS_FRACTION,
    PHASE_4_ELITES_PER_TRIBE,
)

# Seconds per bar of every full-run-equivalent, at campaign level.
# Calibrated against an ECO campaign on this machine: 39,734 XAUUSD M15
# bars, 12 cores, 84 seconds wall clock.
SECONDS_PER_BAR = 1.25e-5

TRIBES = 2
# Phases 1 and 2 replace this share of the breeding pool each generation
# (elite fraction 0.2, phases.py).
CHILD_FRACTION = 0.8
# Pool startup, pickling the bar frame to each worker and the sequential
# stretches inside an otherwise parallel phase.
PARALLEL_EFFICIENCY = 0.75


@dataclass(frozen=True)
class CampaignWork:
    """Work a campaign performs, in full-run equivalents."""

    parallel: float  # Phases 0-3: population evaluations, across the pool
    serial: float  # Phase 4: robustness, noise samples and walk-forward

    @property
    def total(self) -> float:
        return self.parallel + self.serial


def campaign_work(
    population_size: int,
    generations: int,
    phase3_children: int,
    screening_rounds: int,
    phase4_full_depth: bool,
    max_breeding_population: int = DEFAULT_MAX_BREEDING_POPULATION,
) -> CampaignWork:
    """Full-run equivalents for one configuration.

    `screening_rounds` and `phase4_full_depth` are the two things that
    cannot be known before the run: how many times Phase 0 has to redraw
    before it has a breeding pool, and how many Phase 3 children survive
    robustness far enough to reach the noise sampling and the
    walk-forward. Callers pass the optimistic and the pessimistic case
    and report the pair.
    """
    breeding_population = min(population_size, max_breeding_population)
    children = max(1, int(breeding_population * CHILD_FRACTION))
    searched = 1 - HOLDOUT_FRACTION
    is_fraction = searched * IS_FRACTION
    oos_fraction = searched * (1 - IS_FRACTION)

    parallel = (
        screening_rounds * population_size * is_fraction  # Phase 0, redrawn each round
        + TRIBES * generations * children * is_fraction  # Phases 1 and 2
        + phase3_children * is_fraction  # Phase 3
    )

    # Phase 4, per candidate: the out-of-sample baseline always, then the
    # noise re-simulations, the walk-forward windows and the holdout only
    # for those that clear robustness. The walk-forward windows partition
    # the searched history, so together they are one pass over it.
    # Candidates: the screened hybrids (at most every child) plus each
    # tribe's elites.
    candidates = phase3_children + TRIBES * PHASE_4_ELITES_PER_TRIBE
    per_candidate = oos_fraction
    if phase4_full_depth:
        per_candidate += oos_fraction * NOISE_SAMPLES + searched + HOLDOUT_FRACTION
    serial = candidates * per_candidate

    return CampaignWork(parallel=parallel, serial=serial)


def _raw_minutes(
    cpu_cores: int,
    population_size: int,
    generations: int,
    bars: int,
    phase3_children: int,
    max_breeding_population: int,
) -> tuple[float, float]:
    cores = max(1, cpu_cores)

    def minutes(rounds: int, full_depth: bool) -> float:
        work = campaign_work(
            population_size, generations, phase3_children, rounds, full_depth, max_breeding_population
        )
        effective_runs = work.parallel / (cores * PARALLEL_EFFICIENCY) + work.serial
        return effective_runs * bars * SECONDS_PER_BAR / 60

    return minutes(1, False), minutes(MAX_SCREENING_ROUNDS, True)


def estimate_campaign_minutes(
    cpu_cores: int,
    population_size: int = 300,
    generations: int = 5,
    bars: int = 20_000,
    phase3_children: int = 80,
    max_breeding_population: int = DEFAULT_MAX_BREEDING_POPULATION,
    speed_factor: float = 1.0,
) -> tuple[int, int]:
    """An inclusive (min, max) minute range for this configuration.

    The range is the real uncertainty, not a percentage band bolted on
    afterwards: the low end assumes screening finds a breeding pool on
    its first draw and nothing survives robustness, the high end assumes
    it redraws the maximum number of times and every Phase 3 child goes
    through the full noise sampling and walk-forward.

    `speed_factor` is this machine's measured pace against the reference
    one (load_speed_factor): the amount of work is the model's, the time
    each unit of it takes is the computer's.

    Defaults match the MEDIUM POWER preset so a caller with no settings
    in hand still gets a sane number.
    """
    low, high = _raw_minutes(
        cpu_cores, population_size, generations, bars, phase3_children, max_breeding_population
    )
    low, high = low * speed_factor, high * speed_factor
    low_minutes = max(1, round(low))
    return low_minutes, max(low_minutes + 1, round(high))


# --- Learning this machine's pace --------------------------------------
#
# SECONDS_PER_BAR was measured on one computer. The work model is right
# (an instrumented MEDIUM run did 1,146 full-run equivalents where it
# predicts 1,178 to 1,892), but the seconds each unit takes are the
# machine's: on a 4-core Linux container a MEDIUM campaign on 39,734 bars
# took 1.3 minutes against an estimate of 4 to 8. No constant is right
# for every computer, so every finished campaign records how long it took
# against what the model predicted, and the estimate is scaled by the
# median of the recent ratios.

CALIBRATION_FILE = "campaign_timing.json"
CALIBRATION_HISTORY = 10
SPEED_FACTOR_BOUNDS = (0.05, 20.0)


def record_campaign_duration(
    data_dir,
    wall_seconds: float,
    cpu_cores: int,
    population_size: int,
    generations: int,
    bars: int,
    phase3_children: int = 80,
) -> float | None:
    """Stores actual / predicted for a finished campaign. Returns it."""
    import json
    from pathlib import Path

    low, high = _raw_minutes(
        cpu_cores, population_size, generations, bars, phase3_children, DEFAULT_MAX_BREEDING_POPULATION
    )
    predicted = (low * high) ** 0.5  # the middle of the range, on a log scale
    if predicted <= 0 or wall_seconds <= 0:
        return None
    ratio = (wall_seconds / 60) / predicted
    path = Path(data_dir) / CALIBRATION_FILE
    try:
        ratios = json.loads(path.read_text(encoding="utf-8")).get("ratios", [])
    except (OSError, ValueError):
        ratios = []
    ratios = [*ratios, round(ratio, 4)][-CALIBRATION_HISTORY:]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"ratios": ratios}), encoding="utf-8")
    except OSError:
        return None  # an estimate that stays uncalibrated is not worth failing a campaign over
    return ratio


def load_speed_factor(data_dir) -> tuple[float, int]:
    """(factor, campaigns it was learned from); (1.0, 0) with no history."""
    import json
    import statistics
    from pathlib import Path

    try:
        ratios = json.loads((Path(data_dir) / CALIBRATION_FILE).read_text(encoding="utf-8")).get("ratios", [])
        ratios = [float(r) for r in ratios if float(r) > 0]
    except (OSError, ValueError, TypeError):
        ratios = []
    if not ratios:
        return 1.0, 0
    low, high = SPEED_FACTOR_BOUNDS
    return min(high, max(low, statistics.median(ratios))), len(ratios)
