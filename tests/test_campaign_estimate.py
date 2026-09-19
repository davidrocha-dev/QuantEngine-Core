from app.engine.campaign_estimate import campaign_backtest_count, estimate_campaign_minutes


def test_backtest_count_covers_every_phase():
    # Phase 0 population + 2 tribes x generations x children + Phase 3 children
    # (evaluated once in Phase 3, then again against OOS in Phase 4).
    assert campaign_backtest_count(300, 5, 200, 80) == 300 + 2 * 5 * 160 + 80 + 80


def test_more_cores_shortens_the_estimate():
    low_4, high_4 = estimate_campaign_minutes(4)
    low_16, high_16 = estimate_campaign_minutes(16)
    assert low_16 <= low_4
    assert high_16 <= high_4


def test_heavier_preset_and_deeper_history_lengthen_the_estimate():
    _, light = estimate_campaign_minutes(8, population_size=300, generations=5, bars=20_000)
    _, heavy = estimate_campaign_minutes(8, population_size=4000, generations=20, bars=20_000)
    _, deep = estimate_campaign_minutes(8, population_size=300, generations=5, bars=200_000)
    assert heavy > light
    assert deep > light


def test_estimate_is_a_positive_widening_range():
    for cores in (1, 2, 8, 32):
        low, high = estimate_campaign_minutes(cores)
        assert 1 <= low < high
