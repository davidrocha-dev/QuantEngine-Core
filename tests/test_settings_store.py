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
