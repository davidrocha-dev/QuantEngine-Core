from app.engine.campaign_estimate import estimate_campaign_minutes


def test_low_core_count_saturates_at_max():
    low, high = estimate_campaign_minutes(1)
    assert (low, high) == (30, 60)


def test_high_core_count_saturates_at_min():
    low, high = estimate_campaign_minutes(32)
    assert (low, high) == (15, 30)


def test_estimate_always_within_req_dash_1_5_bounds():
    for cores in range(1, 65):
        low, high = estimate_campaign_minutes(cores)
        assert 15 <= low <= high <= 60


def test_estimate_narrows_and_shifts_down_as_cores_increase():
    low_4, high_4 = estimate_campaign_minutes(4)
    low_8, high_8 = estimate_campaign_minutes(8)
    assert low_8 <= low_4
    assert high_8 <= high_4
