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
    return RobustnessResult(20.0, 19.0, mc_degradation, 19.0, noise_degradation, True)


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
