import pytest

from app.settings_store import DEFAULT_APP_SETTINGS, load_settings, save_settings, update_section


def test_load_settings_returns_defaults_when_missing(tmp_path):
    path = tmp_path / "settings.json"
    settings = load_settings(path)
    assert settings == DEFAULT_APP_SETTINGS


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "settings.json"
    settings = load_settings(path)
    settings["risk"]["max_drawdown_pct"] = 15.0
    save_settings(path, settings)

    reloaded = load_settings(path)
    assert reloaded["risk"]["max_drawdown_pct"] == 15.0


def test_load_settings_deep_merges_new_default_fields(tmp_path):
    # Forward compatibility — a file saved before a new field existed must
    # still pick up that field's default rather than KeyError downstream.
    path = tmp_path / "settings.json"
    path.write_text('{"risk": {"max_drawdown_pct": 25.0}}')

    settings = load_settings(path)
    assert settings["risk"]["max_drawdown_pct"] == 25.0
    assert settings["risk"]["per_trade_risk_pct"] == DEFAULT_APP_SETTINGS["risk"]["per_trade_risk_pct"]
    assert settings["campaign"] == DEFAULT_APP_SETTINGS["campaign"]


def test_update_section_merges_and_persists(tmp_path):
    path = tmp_path / "settings.json"
    updated = update_section(path, "campaign", {"name": "First run", "power": "HIGH"})

    assert updated["name"] == "First run"
    assert updated["power"] == "HIGH"
    assert updated["cores"] == DEFAULT_APP_SETTINGS["campaign"]["cores"]

    reloaded = load_settings(path)
    assert reloaded["campaign"]["name"] == "First run"


def test_update_section_unknown_section_raises(tmp_path):
    path = tmp_path / "settings.json"
    try:
        update_section(path, "not_a_section", {})
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_manual_spread_falls_back_per_symbol():
    """One spread figure cannot serve the catalogue.

    A point is 1/point_divisor, and the divisors span two orders of
    magnitude: 10 points is one pip on EURUSD (divisor 100,000) and one
    cent on XAUUSD (divisor 1,000) — about a seventeenth of what gold
    costs to cross. Campaigns bred with no terminal open charged that
    cent.
    """
    from app.main import _manual_spread_points

    costs = {
        "manual_spread_points": 10.0,
        "manual_spread_by_symbol": {"XAUUSD": 150.0},
    }

    assert _manual_spread_points(costs, "XAUUSD") == 150.0
    assert _manual_spread_points(costs, "EURUSD") == 10.0
    # A settings file written before the map existed still resolves.
    assert _manual_spread_points({"manual_spread_points": 10.0}, "XAUUSD") == 10.0


def test_the_account_size_and_the_volume_ladder_reach_the_engine():
    """Both were declared and never read.

    initial_balance was hard-wired to 10,000 whatever Settings said, and
    broker_caps held a max_lot_size nothing consumed. Neither is cosmetic:
    fixed-fractional sizing asks for a fraction of a lot, and whether that
    clears the broker's minimum depends entirely on the account. 1% of
    10,000 on XAUUSD is about 0.034 lots and loses a seventh of itself to
    rounding; 1% of 100,000 is 0.34 and loses a seventieth.
    """
    defaults = DEFAULT_APP_SETTINGS
    assert defaults["risk"]["account_balance"] == 10_000.0
    assert defaults["broker_caps"]["min_lot_size"] == 0.01
    assert defaults["broker_caps"]["lot_step"] == 0.01


def test_auto_spread_says_when_it_undercuts_the_configured_one(caplog):
    """AUTO reads one live quote and charges it against years of history.

    Campaigns get started in calm moments, and the spread in a calm moment
    is not the spread over a news release or a rollover, so AUTO is
    optimistic by construction. It was also silent: the owner configured
    600 points for XAUUSD, AUTO read 0.15 of a dollar off a connected
    terminal, and the campaign that produced three GOOD strategies charged
    150 - a quarter of what the broker had actually been measured taking.
    """
    from app.main import _spread_points

    costs = {
        "spread_mode": "AUTO",
        "manual_spread_points": 10.0,
        "manual_spread_by_symbol": {"XAUUSD": 600},
        "commission_per_lot": 7.0,
        "slippage_points": 2.0,
    }

    class _Quote:
        spread_points = 0.15

    class _Bridge:
        def is_connected(self):
            return True

        def connect(self):
            return True

        def get_live_quote(self, symbol):
            return _Quote()

    class _Instrument:
        point_divisor = 1000

    import app.main as main

    original = main._mt5_bridge
    main._mt5_bridge = _Bridge()
    try:
        with caplog.at_level("WARNING", logger="quantengine"):
            charged = _spread_points(costs, "XAUUSD", _Instrument())
    finally:
        main._mt5_bridge = original

    # AUTO still wins - it is what the setting asks for - but not quietly.
    assert charged == pytest.approx(150.0)
    assert any("below the 600" in record.getMessage() for record in caplog.records)
