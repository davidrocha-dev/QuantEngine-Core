import json

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app

client = TestClient(app)


def test_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json()["state"] == "idle"


def test_vault_empty(tmp_path, monkeypatch):
    # Isolated from the real dev vault.db — otherwise this depends on
    # nothing having ever bred a real strategy on this machine.
    from app.migrations.runner import run_migrations

    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    run_migrations(fake_config.vault_db_path)

    response = client.get("/vault")
    assert response.status_code == 200
    assert response.json() == []


def test_mt5_status_disconnected(monkeypatch):
    # REQ-DASH-1.4 — no terminal reachable must report disconnected, never raise.
    monkeypatch.setattr(main_module._mt5_bridge, "is_connected", lambda: False)
    monkeypatch.setattr(main_module._mt5_bridge, "connect", lambda: False)

    response = client.get("/mt5/status")
    assert response.status_code == 200
    body = response.json()
    assert body["connected"] is False
    assert body["server"] is None
    assert body["latency_ms"] is None


def test_mt5_status_connected_flags_metaquotes_demo(monkeypatch):
    # REQ-DASH-1.2, 1.3 — server name/latency surfaced, MetaQuotes-Demo flagged.
    monkeypatch.setattr(main_module._mt5_bridge, "is_connected", lambda: True)
    monkeypatch.setattr(
        main_module._mt5_bridge,
        "get_server_info",
        lambda: {"server": "MetaQuotes-Demo", "is_demo": True, "is_metaquotes_demo": True},
    )

    response = client.get("/mt5/status")
    assert response.status_code == 200
    body = response.json()
    assert body["connected"] is True
    assert body["server"] == "MetaQuotes-Demo"
    assert body["is_metaquotes_demo"] is True
    assert isinstance(body["latency_ms"], float)


def test_campaign_estimate_within_req_dash_1_5_bounds():
    response = client.get("/campaign/estimate")
    assert response.status_code == 200
    body = response.json()
    assert 15 <= body["min_minutes"] <= body["max_minutes"] <= 60
    assert body["cpu_cores"] >= 1


def test_data_instruments_lists_full_catalog():
    response = client.get("/data/instruments")
    assert response.status_code == 200
    body = response.json()
    assert "EURUSD" in body["symbols"]
    assert "XAGUSD" in body["symbols"]
    assert "M15" in body["timeframes"]


def test_vault_purge_reports_zero_on_empty_vault():
    response = client.post("/vault/purge")
    assert response.status_code == 200
    assert response.json() == {"purged": 0}


def test_system_info_reports_paths():
    response = client.get("/system/info")
    assert response.status_code == 200
    body = response.json()
    assert "data_dir" in body
    assert "vault_db_path" in body


class _FakeConfig:
    def __init__(self, tmp_path):
        self.settings_path = tmp_path / "settings.json"
        self.gene_pool_path = tmp_path / "gene_pool.json"
        self.vault_db_path = tmp_path / "vault.db"


def test_app_settings_defaults_and_patch_round_trip(tmp_path, monkeypatch):
    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)

    defaults = client.get("/settings/app")
    assert defaults.status_code == 200
    assert defaults.json()["risk"]["max_drawdown_pct"] == 20.0

    patched = client.post("/settings/app/campaign", json={"name": "First run", "power": "HIGH"})
    assert patched.status_code == 200
    assert patched.json()["name"] == "First run"
    assert patched.json()["power"] == "HIGH"

    reread = client.get("/settings/app")
    assert reread.json()["campaign"]["name"] == "First run"


def test_app_settings_patch_unknown_section_404(tmp_path, monkeypatch):
    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)

    response = client.post("/settings/app/not_a_section", json={})
    assert response.status_code == 404


def test_gene_pool_defaults_all_enabled_46_genes(tmp_path, monkeypatch):
    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)

    response = client.get("/settings/gene-pool")
    assert response.status_code == 200
    body = response.json()
    assert len(body["genes"]) == 46
    assert all(body["genes"].values())
    assert body["layer_counts"]["signal"] == {"enabled": 16, "total": 16}


def test_gene_pool_toggle_persists(tmp_path, monkeypatch):
    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)

    response = client.post("/settings/gene-pool", json={"genes": {"signal::MACD": False}})
    assert response.status_code == 200
    assert response.json()["genes"]["signal::MACD"] is False
    assert response.json()["layer_counts"]["signal"] == {"enabled": 15, "total": 16}

    reread = client.get("/settings/gene-pool")
    assert reread.json()["genes"]["signal::MACD"] is False


def test_gene_pool_unknown_gene_400(tmp_path, monkeypatch):
    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)

    response = client.post("/settings/gene-pool", json={"genes": {"signal::NOT_A_GENE": False}})
    assert response.status_code == 400


class _NoOpThread:
    """threading.Thread stand-in — records the target/args without running
    them, so campaign tests never actually execute the (15-60 min) GA run."""

    def __init__(self, target=None, args=(), daemon=None):
        self.target = target
        self.args = args

    def start(self):
        pass


def test_campaign_start_creates_running_campaign_row(tmp_path, monkeypatch):
    from app.migrations.runner import run_migrations

    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    monkeypatch.setattr(main_module, "_active_campaign_id", None)
    monkeypatch.setattr(main_module.threading, "Thread", _NoOpThread)
    run_migrations(fake_config.vault_db_path)

    response = client.post("/campaign/start")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    assert body["campaign_id"]

    status = client.get("/status")
    assert status.json()["state"] == "running"
    assert status.json()["campaign"]["id"] == body["campaign_id"]
    assert status.json()["campaign"]["status"] == "running"


def test_campaign_start_409_when_already_running(tmp_path, monkeypatch):
    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    monkeypatch.setattr(main_module, "_active_campaign_id", "already-running-id")

    response = client.post("/campaign/start")
    assert response.status_code == 409


def test_campaign_pause_and_skip_are_honest_about_being_unsupported():
    assert client.post("/campaign/pause").json()["status"] == "not_supported"
    assert client.post("/campaign/skip").json()["status"] == "not_supported"


def test_vault_lists_strategies(tmp_path, monkeypatch):
    from app.migrations.runner import run_migrations
    from app.models import Strategy

    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    run_migrations(fake_config.vault_db_path)

    engine = main_module.make_engine(fake_config.vault_db_path)
    from sqlalchemy.orm import Session as OrmSession

    with OrmSession(engine) as session:
        session.add(
            Strategy(
                id="abc123",
                data_source="dukascopy",
                asset="EURUSD",
                timeframe="H1",
                score=66,
                grade="GOOD",
                archetype="multi-signal",
                genome_json="{}",
            )
        )
        session.commit()

    response = client.get("/vault")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == "abc123"
    assert body[0]["score"] == 66
    # genome_json="{}" — no signal layer to parse, so this must degrade
    # gracefully (REQ-VAULT-1.4 "active genes" is best-effort, never a 500).
    assert body[0]["active_genes"] == []
    assert body[0]["mechanism"] is None


def test_neural_map_reports_not_ready_below_minimum_population(tmp_path, monkeypatch):
    from app.migrations.runner import run_migrations
    from app.models import Strategy

    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    run_migrations(fake_config.vault_db_path)

    engine = main_module.make_engine(fake_config.vault_db_path)
    from sqlalchemy.orm import Session as OrmSession

    with OrmSession(engine) as session:
        session.add(
            Strategy(
                id="only-one", data_source="dukascopy", asset="EURUSD", timeframe="H1",
                score=66, grade="GOOD", archetype="MULTI_SIGNAL", genome_json="{}",
            )
        )
        session.commit()

    response = client.get("/neural-map")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is False
    assert body["strategy_count"] == 1
    assert body["nodes"] == []
    assert body["clusters"] == []


def _seed_strategy(session, **overrides) -> None:
    from app.genome.genome import Genome, GeneChoice
    from app.models import Strategy

    genome = Genome(
        signal=GeneChoice("BREAKOUT", {}),
        execution=GeneChoice("STOP", {}),
        bias=GeneChoice("RSI", {}),
        filters=[GeneChoice("DOJI", {})],
        management=GeneChoice("EOD CLOSE", {}),
    )
    defaults = dict(
        id="strat1",
        data_source="dukascopy",
        asset="EURUSD",
        timeframe="H1",
        trades=150,
        score=70,
        grade="GOOD",
        archetype="MULTI_SIGNAL",
        genome_json=json.dumps(genome.to_dict()),
    )
    defaults.update(overrides)
    session.add(Strategy(**defaults))
    session.commit()


def test_vault_full_shape_includes_mechanism_and_active_genes(tmp_path, monkeypatch):
    from app.migrations.runner import run_migrations
    from sqlalchemy.orm import Session as OrmSession

    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    run_migrations(fake_config.vault_db_path)

    with OrmSession(main_module.make_engine(fake_config.vault_db_path)) as session:
        _seed_strategy(session)

    body = client.get("/vault").json()
    assert body[0]["mechanism"] == "BREAKOUT"
    layers = {g["layer"] for g in body[0]["active_genes"]}
    assert layers == {"signal", "bias", "filter", "execution", "management"}


def test_delete_strategy_removes_row(tmp_path, monkeypatch):
    from app.migrations.runner import run_migrations
    from sqlalchemy.orm import Session as OrmSession

    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    run_migrations(fake_config.vault_db_path)

    with OrmSession(main_module.make_engine(fake_config.vault_db_path)) as session:
        _seed_strategy(session)

    response = client.delete("/vault/strat1")
    assert response.status_code == 200
    assert client.get("/vault").json() == []


def test_delete_unknown_strategy_404(tmp_path, monkeypatch):
    from app.migrations.runner import run_migrations

    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    run_migrations(fake_config.vault_db_path)

    assert client.delete("/vault/nope").status_code == 404


def test_refresh_reassess_parity_test_are_honest_about_being_unsupported():
    assert client.post("/vault/x/refresh").json()["status"] == "not_supported"
    assert client.post("/vault/x/reassess").json()["status"] == "not_supported"
    assert client.post("/vault/x/parity-test").json()["status"] == "not_supported"
    assert client.post("/deploy/x").json()["status"] == "not_supported"


def test_portfolio_create_list_and_add_strategy(tmp_path, monkeypatch):
    from app.migrations.runner import run_migrations
    from sqlalchemy.orm import Session as OrmSession

    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    run_migrations(fake_config.vault_db_path)

    with OrmSession(main_module.make_engine(fake_config.vault_db_path)) as session:
        _seed_strategy(session)

    created = client.post("/portfolios", json={"name": "Core", "description": "core bots"})
    assert created.status_code == 200
    portfolio_id = created.json()["id"]

    listed = client.get("/portfolios").json()
    assert len(listed) == 1
    assert listed[0]["name"] == "Core"

    added = client.post(f"/vault/strat1/add-to-portfolio/{portfolio_id}")
    assert added.status_code == 200

    # Idempotent — adding the same strategy twice must not raise.
    added_again = client.post(f"/vault/strat1/add-to-portfolio/{portfolio_id}")
    assert added_again.status_code == 200


def test_add_to_portfolio_unknown_strategy_404(tmp_path, monkeypatch):
    from app.migrations.runner import run_migrations

    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    run_migrations(fake_config.vault_db_path)

    created = client.post("/portfolios", json={"name": "Core"})
    portfolio_id = created.json()["id"]

    response = client.post(f"/vault/does-not-exist/add-to-portfolio/{portfolio_id}")
    assert response.status_code == 404


def test_best_fitness_extracts_max_from_checkpoint():
    checkpoint = '{"active_population": [{"fitness": 0.4}, {"fitness": 0.9}, {"fitness": null}]}'
    assert main_module._best_fitness(checkpoint) == 0.9


def test_best_fitness_handles_missing_and_malformed_input():
    assert main_module._best_fitness(None) is None
    assert main_module._best_fitness("not json") is None
    assert main_module._best_fitness('{"active_population": []}') is None
