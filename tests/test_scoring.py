import pytest

from app.engine.metrics import TradeStats
from app.engine.monte_carlo import RobustnessResult
from app.engine.scoring import classify_archetype, compute_grade, compute_score
from app.genome.genome import GeneChoice, Genome


def make_stats(**overrides) -> TradeStats:
    defaults = dict(
        trade_count=150, profit_factor=2.0, win_rate=0.6, sharpe_ratio=1.5,
        max_dd_pct=10.0, net_profit_pct=30.0, cagr_pct=15.0, payoff_ratio=1.8,
        icir=1.0, avg_hold_hours=4.0,
    )
    defaults.update(overrides)
    return TradeStats(**defaults)


def make_robustness(mc_degradation=5.0, noise_degradation=5.0) -> RobustnessResult:
    return RobustnessResult(
        baseline_return_pct=20.0,
        monte_carlo_median_return_pct=19.0,
        monte_carlo_worst_return_pct=12.0,
        monte_carlo_degradation_pct=mc_degradation,
        monte_carlo_worst_drawdown_pct=8.0,
        noise_return_pct=19.0,
        noise_degradation_pct=noise_degradation,
        passed=True,
    )


def test_classify_archetype_mean_reverter():
    genome = Genome(signal=GeneChoice("FADE", {}), execution=GeneChoice("STOP", {}))
    assert classify_archetype(genome) == "MEAN_REVERTER"


def test_classify_archetype_pattern_spotter():
    genome = Genome(signal=GeneChoice("ENGULFING", {}), execution=GeneChoice("STOP", {}))
    assert classify_archetype(genome) == "PATTERN_SPOTTER"


def test_classify_archetype_multi_signal_default():
    genome = Genome(signal=GeneChoice("BREAKOUT", {}), execution=GeneChoice("STOP", {}))
    assert classify_archetype(genome) == "MULTI_SIGNAL"


def test_compute_score_high_quality_strategy_scores_high():
    score = compute_score(make_stats(), make_robustness())
    assert score >= 57  # GOOD grade territory for a strong strategy


def test_compute_score_weak_strategy_scores_low():
    weak_stats = make_stats(profit_factor=0.5, win_rate=0.2, sharpe_ratio=0.1, max_dd_pct=45.0)
    weak_robustness = make_robustness(mc_degradation=24.0, noise_degradation=24.0)
    score = compute_score(weak_stats, weak_robustness)
    assert score < 57


def test_compute_score_bounded_0_100():
    score = compute_score(make_stats(profit_factor=float("inf"), sharpe_ratio=10.0), make_robustness(0.0, 0.0))
    assert 0 <= score <= 100


def test_compute_grade_boundary():
    assert compute_grade(57) == "GOOD"
    assert compute_grade(56) == "WEAK"


# --- how much evidence is behind the score --------------------------------


def test_a_thin_sample_scores_below_the_same_result_on_a_thick_one():
    """The gap this closed.

    A strategy with 30 out-of-sample trades and one with 400 used to
    score identically, and the Vault is sorted by score — so the thin one
    could sit at the top of the list and be the one someone deploys.
    """
    thin = compute_score(make_stats(trade_count=30), make_robustness())
    thick = compute_score(make_stats(trade_count=400), make_robustness())

    assert thin < thick
    assert thick - thin >= 5  # enough to change where it sorts


def test_the_sample_component_stops_rewarding_past_the_point_it_matters():
    from app.engine.scoring import FULL_CONFIDENCE_TRADES, sample_adequacy

    assert sample_adequacy(FULL_CONFIDENCE_TRADES) == 1.0
    assert sample_adequacy(FULL_CONFIDENCE_TRADES * 10) == 1.0
    # Square root, because that is how the precision of an estimate scales
    # with its sample: four times the trades is twice the confidence.
    assert sample_adequacy(FULL_CONFIDENCE_TRADES // 4) == pytest.approx(0.5, abs=0.01)


def test_no_trades_is_no_confidence():
    from app.engine.scoring import sample_adequacy

    assert sample_adequacy(0) == 0.0


def test_a_strong_result_on_a_thin_sample_is_never_graded_good():
    """A dozen trades is a fluke until proven otherwise, however good it
    looks - and one component in six cannot hold back a result that is
    excellent on every other axis. Only a floor can."""
    from app.engine.scoring import MIN_TRADES_FOR_GOOD

    stats = make_stats(trade_count=12)
    score = compute_score(stats, make_robustness())

    assert score >= 57  # it still looks good on every other measure
    assert compute_grade(score, stats.trade_count) == "WEAK"
    assert compute_grade(score, MIN_TRADES_FOR_GOOD) == "GOOD"


def test_grading_without_a_trade_count_still_works():
    # Callers that only have a score keep the old behaviour.
    assert compute_grade(70) == "GOOD"
