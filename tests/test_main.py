import json

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.migrations.runner import run_migrations

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


def test_campaign_estimate_reports_the_work_it_is_based_on():
    # REQ-DASH-1.5 — the estimate is derived from the campaign actually
    # configured (POWER preset, cores, bars available), so it reports those
    # inputs rather than a fixed 15-60 band.
    response = client.get("/campaign/estimate")
    assert response.status_code == 200
    body = response.json()
    assert 1 <= body["min_minutes"] < body["max_minutes"]
    assert body["cpu_cores"] >= 1
    assert body["power"] in ("ECO", "MEDIUM", "HIGH", "EXTREME")
    assert body["symbol"] and body["timeframe"]


def test_data_instruments_lists_full_catalog():
    response = client.get("/data/instruments")
    assert response.status_code == 200
    body = response.json()
    assert "EURUSD" in body["symbols"]
    assert "XAGUSD" in body["symbols"]
    assert "M15" in body["timeframes"]


def test_vault_purge_reports_zero_on_empty_vault(tmp_path, monkeypatch):
    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    run_migrations(fake_config.vault_db_path)

    response = client.post("/vault/purge")

    assert response.status_code == 200
    body = response.json()
    assert body["purged"] == 0
    assert body["protected"] == 0
    assert body["skipped_reason"] is None


def test_recycle_bin_is_empty_by_default(tmp_path, monkeypatch):
    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    run_migrations(fake_config.vault_db_path)

    response = client.get("/vault/recycle-bin")

    assert response.status_code == 200
    assert response.json() == []


def test_system_info_reports_paths():
    response = client.get("/system/info")
    assert response.status_code == 200
    body = response.json()
    assert "data_dir" in body
    assert "vault_db_path" in body


class _FakeConfig:
    def __init__(self, tmp_path):
        self.data_dir = tmp_path
        self.bundled_data_dir = tmp_path / "bundled"
        self.cache_dir = tmp_path / "cache"
        self.built_in_data_only = False
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


def test_pausing_when_nothing_is_running_says_so(monkeypatch):
    # Both of these used to answer "not supported" while the campaign
    # carried on regardless. They do the job now; asking with no campaign
    # running is the one case where refusing is still right.
    monkeypatch.setattr(main_module, "_active_campaign_id", None)

    assert client.post("/campaign/pause").status_code == 409
    assert client.post("/campaign/skip").status_code == 409


def test_pause_and_skip_raise_the_flags_the_engine_reads(monkeypatch):
    monkeypatch.setattr(main_module, "_active_campaign_id", "campaign-1")
    main_module._campaign_control.clear()

    paused = client.post("/campaign/pause").json()
    assert paused["status"] == "pausing"
    assert main_module._campaign_control.pause_requested is True

    skipped = client.post("/campaign/skip").json()
    assert skipped["status"] == "skipping"
    assert main_module._campaign_control.take_skip() is True

    main_module._campaign_control.clear()


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


def test_parity_test_on_a_strategy_that_does_not_exist(tmp_path, monkeypatch):
    # The parity test is implemented (see test_parity.py): it compares how
    # often the EA actually trades against how often the backtest did,
    # which is what catches a robot on the wrong chart.
    from app.migrations.runner import run_migrations

    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    run_migrations(fake_config.vault_db_path)

    assert client.post("/vault/nope/parity-test").status_code == 404


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


def _seed_deployable_strategy(config, genome_dict: dict, strategy_id: str = "#DEP001") -> None:
    from sqlalchemy.orm import Session

    from app.db import make_engine
    from app.models import Strategy

    run_migrations(config.vault_db_path)
    engine = make_engine(config.vault_db_path)
    with Session(engine) as session:
        session.add(
            Strategy(
                id=strategy_id, data_source="dukascopy", asset="EURUSD", timeframe="M15",
                trades=120, return_pct=12.0, max_dd_pct=8.0, score=70, grade="GOOD",
                genome_json=json.dumps(genome_dict),
            )
        )
        session.commit()


def _simple_genome(**overrides) -> dict:
    genome = {
        "signal": {"name": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}},
        "bias": {"name": "SMA", "params": {"period": 50}},
        "filters": [],
        "execution": {"name": "MARKET2", "params": {"confirm_bars": 2}},
        "management": {"name": "EOD CLOSE", "params": {"close_hour": 21}},
        "stop_atr_mult": 1.5,
        "target_atr_mult": 3.0,
    }
    genome.update(overrides)
    return genome


def test_export_mql5_writes_the_ea_to_the_chosen_folder(tmp_path, monkeypatch):
    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    _seed_deployable_strategy(fake_config, _simple_genome())
    destination = tmp_path / "exports"

    response = client.post("/vault/%23DEP001/export-mql5", json={"destination_dir": str(destination)})

    assert response.status_code == 200
    exported = tmp_path / "exports" / "QE_DEP001.mq5"
    assert exported.exists()
    assert "OnTick" in exported.read_text(encoding="utf-8")


def test_export_refuses_a_genome_whose_gene_has_no_mql5_equivalent(tmp_path, monkeypatch):
    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    _seed_deployable_strategy(fake_config, _simple_genome(filters=[{"name": "EVENT", "params": {"mode": "FTMO"}}]))

    response = client.post("/vault/%23DEP001/export-mql5", json={"destination_dir": str(tmp_path / "out")})

    assert response.status_code == 422
    assert "EVENT" in response.json()["detail"]


def test_deploy_installs_the_ea_and_marks_the_strategy_deployed(tmp_path, monkeypatch):
    from sqlalchemy.orm import Session

    from app.broker.mt5_install import InstallResult
    from app.db import make_engine
    from app.models import Strategy

    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    _seed_deployable_strategy(fake_config, _simple_genome())

    captured = {}

    def fake_install(filename, source, compile_after_install=True):
        captured["filename"], captured["source"] = filename, source
        return InstallResult(installed_paths=["C:/fake/QE_DEP001.mq5"], detail="Installed into 1 terminal(s).")

    monkeypatch.setattr(main_module, "install_ea", fake_install)

    response = client.post("/deploy/%23DEP001")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "deployed"
    assert body["filename"] == "QE_DEP001.mq5"
    assert "OnInit" in captured["source"]

    engine = make_engine(fake_config.vault_db_path)
    with Session(engine) as session:
        strategy = session.get(Strategy, "#DEP001")
        assert strategy.deployed is True
        # Installing a file is not the same as the EA running on a chart.
        assert strategy.active_mt5 is False


def test_deploy_reports_a_conflict_when_no_terminal_is_installed(tmp_path, monkeypatch):
    from app.broker.mt5_install import InstallResult

    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    _seed_deployable_strategy(fake_config, _simple_genome())
    monkeypatch.setattr(
        main_module, "install_ea", lambda *a, **k: InstallResult(detail="No MetaTrader 5 terminal data directory was found.")
    )

    response = client.post("/deploy/%23DEP001")

    assert response.status_code == 409
    assert "MetaTrader 5" in response.json()["detail"]


def test_deploy_unknown_strategy_404(tmp_path, monkeypatch):
    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    run_migrations(fake_config.vault_db_path)

    assert client.post("/deploy/%23NOPE").status_code == 404


def _seed_portfolio_with_members(config, member_count: int = 2) -> str:
    """A portfolio whose members carry stored OOS equity curves, so the
    health endpoint has something real to correlate."""
    import uuid
    from datetime import datetime, timedelta, timezone

    from sqlalchemy.orm import Session

    from app.db import make_engine
    from app.models import Portfolio, PortfolioStrategy, Strategy

    run_migrations(config.vault_db_path)
    engine = make_engine(config.vault_db_path)
    portfolio_id = uuid.uuid4().hex

    def curve(offset: float) -> str:
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        equity, points = 10_000.0, []
        for day in range(30):
            equity *= 1 + (0.01 if (day + offset) % 3 else -0.008)
            points.append({"t": (start + timedelta(days=day)).isoformat(), "equity": round(equity, 2)})
        return json.dumps(points)

    with Session(engine) as session:
        session.add(Portfolio(id=portfolio_id, name="Core"))
        for index in range(member_count):
            strategy_id = f"#MEM{index}"
            session.add(
                Strategy(
                    id=strategy_id, data_source="dukascopy", asset="EURUSD" if index else "GBPUSD",
                    timeframe="M15", archetype="MULTI_SIGNAL", genome_json="{}", score=70, grade="GOOD",
                    trades=80, return_pct=5.0, max_dd_pct=6.0, equity_oos=curve(index),
                )
            )
            session.add(PortfolioStrategy(portfolio_id=portfolio_id, strategy_id=strategy_id))
        session.commit()
    return portfolio_id


def test_portfolio_detail_lists_its_members(tmp_path, monkeypatch):
    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    portfolio_id = _seed_portfolio_with_members(fake_config)

    response = client.get(f"/portfolios/{portfolio_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Core"
    assert {member["id"] for member in body["members"]} == {"#MEM0", "#MEM1"}


def test_portfolio_health_reports_correlation_and_a_grade(tmp_path, monkeypatch):
    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    portfolio_id = _seed_portfolio_with_members(fake_config)

    response = client.get(f"/portfolios/{portfolio_id}/health")

    assert response.status_code == 200
    body = response.json()
    assert body["strategy_count"] == 2
    assert body["diversity_grade"] in ("A", "B", "C", "D", "E", "F")
    assert body["recommendations"]  # two members: it should say to grow
    assert body["combined_equity"]


def test_portfolio_list_reports_member_counts(tmp_path, monkeypatch):
    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    _seed_portfolio_with_members(fake_config, member_count=2)

    body = client.get("/portfolios").json()

    assert body[0]["member_count"] == 2


def test_removing_a_member_leaves_the_strategy_in_the_vault(tmp_path, monkeypatch):
    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    portfolio_id = _seed_portfolio_with_members(fake_config)

    removed = client.delete(f"/portfolios/{portfolio_id}/members/%23MEM0")

    assert removed.status_code == 200
    members = client.get(f"/portfolios/{portfolio_id}").json()["members"]
    assert {member["id"] for member in members} == {"#MEM1"}
    assert any(row["id"] == "#MEM0" for row in client.get("/vault").json())


def test_deleting_a_portfolio_keeps_its_strategies(tmp_path, monkeypatch):
    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    portfolio_id = _seed_portfolio_with_members(fake_config)

    assert client.delete(f"/portfolios/{portfolio_id}").status_code == 200
    assert client.get(f"/portfolios/{portfolio_id}").status_code == 404
    assert len(client.get("/vault").json()) == 2


def test_unknown_portfolio_health_404(tmp_path, monkeypatch):
    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    run_migrations(fake_config.vault_db_path)

    assert client.get("/portfolios/nope/health").status_code == 404


def _seed_vault_for_analytics(config) -> None:
    from sqlalchemy.orm import Session

    from app.db import make_engine
    from app.models import Strategy

    run_migrations(config.vault_db_path)
    engine = make_engine(config.vault_db_path)
    rows = [
        ("#A1", "EURUSD", "M15", 20.0, 5.0, 80, "GOOD", "MULTI_SIGNAL"),
        ("#A2", "EURUSD", "M15", -4.0, 12.0, 40, "WEAK", "MULTI_SIGNAL"),
        ("#A3", "GBPUSD", "H1", 8.0, 7.0, 65, "GOOD", "MEAN_REVERTER"),
    ]
    with Session(engine) as session:
        for strategy_id, asset, timeframe, ret, dd, score, grade, archetype in rows:
            session.add(
                Strategy(
                    id=strategy_id, data_source="dukascopy", asset=asset, timeframe=timeframe,
                    trades=100, return_pct=ret, max_dd_pct=dd, score=score, grade=grade,
                    archetype=archetype, genome_json="{}",
                )
            )
        session.commit()


def test_analytics_reports_vault_wide_kpis(tmp_path, monkeypatch):
    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    _seed_vault_for_analytics(fake_config)

    body = client.get("/analytics").json()

    assert body["total_strategies"] == 3
    assert body["avg_return_pct"] == round((20.0 - 4.0 + 8.0) / 3, 2)
    assert body["avg_drawdown_pct"] == round((5.0 + 12.0 + 7.0) / 3, 2)
    assert body["profitable_pct"] == round(100 * 2 / 3, 1)
    assert body["good_grade_count"] == 2


def test_analytics_breakdowns_group_and_rank_by_size(tmp_path, monkeypatch):
    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    _seed_vault_for_analytics(fake_config)

    body = client.get("/analytics").json()

    assert body["by_symbol"][0]["name"] == "EURUSD"
    assert body["by_symbol"][0]["count"] == 2
    assert body["by_market"][0]["name"] == "EURUSD M15"
    assert {entry["name"] for entry in body["by_archetype"]} == {"MULTI_SIGNAL", "MEAN_REVERTER"}
    assert body["by_broker"][0]["name"] == "dukascopy"


def test_analytics_top_performers_are_ranked_by_score(tmp_path, monkeypatch):
    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    _seed_vault_for_analytics(fake_config)

    top = client.get("/analytics?top=2").json()["top_performers"]

    assert [row["id"] for row in top] == ["#A1", "#A3"]


def test_analytics_on_an_empty_vault_is_all_nulls_not_zeros(tmp_path, monkeypatch):
    # An empty Vault has no average return; reporting 0.0% would read as a
    # measured result.
    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    run_migrations(fake_config.vault_db_path)

    body = client.get("/analytics").json()

    assert body["total_strategies"] == 0
    assert body["avg_return_pct"] is None
    assert body["profitable_pct"] is None
    assert body["top_performers"] == []


def _seed_strategy_for_revalidation(config, monkeypatch, bars: int = 6_000) -> None:
    """A Vault row plus enough synthetic history for a re-check to run."""
    from sqlalchemy.orm import Session

    import app.main as main
    from app.db import make_engine
    from app.models import Strategy
    from tests.helpers import make_synthetic_ohlc

    run_migrations(config.vault_db_path)
    genome = {
        "signal": {"name": "BREAKOUT", "params": {"period": 20}},
        "bias": None,
        "filters": [],
        "execution": {"name": "MARKET2", "params": {"confirm_bars": 2}},
        "management": {"name": "EOD CLOSE", "params": {"close_hour": 21}},
        "stop_atr_mult": 1.5,
        "target_atr_mult": 3.0,
    }
    engine = make_engine(config.vault_db_path)
    with Session(engine) as session:
        session.add(
            Strategy(
                id="#REV001", data_source="dukascopy", asset="EURUSD", timeframe="H1",
                trades=10, return_pct=1.0, max_dd_pct=1.0, score=99, grade="GOOD",
                genome_json=json.dumps(genome),
            )
        )
        session.commit()

    df = make_synthetic_ohlc(bars, seed=12, drift=0.0004, volatility=0.002)
    monkeypatch.setattr(main, "_load_bars", lambda *a, **k: df)


def test_refresh_restates_the_strategy_against_current_data(tmp_path, monkeypatch):
    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    _seed_strategy_for_revalidation(fake_config, monkeypatch)

    response = client.post("/vault/%23REV001/refresh")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["previous_score"] == 99
    # The stored row now reflects the re-run, not the numbers it was seeded with.
    row = [r for r in client.get("/vault").json() if r["id"] == "#REV001"][0]
    assert row["trades"] == body["trades"]


def test_reassess_reapplies_the_phase_4_gates_and_rescores(tmp_path, monkeypatch):
    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    _seed_strategy_for_revalidation(fake_config, monkeypatch)

    body = client.post("/vault/%23REV001/reassess").json()

    assert body["status"] == "ok"
    assert body["still_passes"] in (True, False)
    if body["still_passes"] is False:
        assert body["failed_gate"]
    else:
        assert body["score"] is not None and body["grade"] in ("GOOD", "WEAK")


def test_revalidation_refuses_when_there_is_not_enough_history(tmp_path, monkeypatch):
    import app.main as main
    from tests.helpers import make_synthetic_ohlc

    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    _seed_strategy_for_revalidation(fake_config, monkeypatch)
    monkeypatch.setattr(main, "_load_bars", lambda *a, **k: make_synthetic_ohlc(500))

    response = client.post("/vault/%23REV001/reassess")

    assert response.status_code == 409
    assert "history" in response.json()["detail"]


def test_revalidating_an_unknown_strategy_404(tmp_path, monkeypatch):
    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    run_migrations(fake_config.vault_db_path)

    assert client.post("/vault/%23NOPE/refresh").status_code == 404


def test_true_sim_replays_the_members_on_one_account(tmp_path, monkeypatch):
    import app.main as main
    from tests.helpers import make_synthetic_ohlc

    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    portfolio_id = _seed_portfolio_with_members(fake_config)
    # Members seeded for health tests carry "{}" genomes; give them real ones.
    _give_members_real_genomes(fake_config)
    monkeypatch.setattr(main, "_load_bars", lambda *a, **k: make_synthetic_ohlc(3_000, seed=31, drift=0.0004))

    response = client.post(f"/portfolios/{portfolio_id}/true-sim", json={"sizing": "COMPOUNDING", "max_aggregate_risk_pct": 5.0})

    assert response.status_code == 200
    body = response.json()
    assert body["strategy_count"] == 2
    assert body["sizing"] == "COMPOUNDING"
    assert body["trades_blocked"] == body["blocked_by_symbol_lock"] + body["blocked_by_aggregate_risk"]
    assert body["computed_at"]


def _give_members_real_genomes(config) -> None:
    from sqlalchemy.orm import Session
    from sqlalchemy import select as sa_select

    from app.db import make_engine
    from app.genome.genes import ALL_GENES
    from app.models import Strategy

    genome = json.dumps(
        {
            "signal": {"name": "BREAKOUT", "params": dict(ALL_GENES[("signal", "BREAKOUT")].default_params)},
            "bias": None,
            "filters": [],
            "execution": {"name": "MARKET2", "params": {"confirm_bars": 2}},
            "management": None,
            "stop_atr_mult": 1.5,
            "target_atr_mult": 3.0,
        }
    )
    engine = make_engine(config.vault_db_path)
    with Session(engine) as session:
        for row in session.scalars(sa_select(Strategy)):
            row.genome_json = genome
        session.commit()


def test_true_sim_on_an_empty_portfolio_is_a_conflict(tmp_path, monkeypatch):
    from sqlalchemy.orm import Session

    from app.db import make_engine
    from app.models import Portfolio

    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    run_migrations(fake_config.vault_db_path)
    engine = make_engine(fake_config.vault_db_path)
    with Session(engine) as session:
        session.add(Portfolio(id="empty", name="Empty"))
        session.commit()

    response = client.post("/portfolios/empty/true-sim", json={})

    assert response.status_code == 409
    assert "members" in response.json()["detail"]


def test_deleting_a_strategy_by_hand_is_recoverable(tmp_path, monkeypatch):
    """DELETE used to remove the row outright.

    PURGE — the automated, bulk, criteria-driven one — sent its
    strategies to a Recycle Bin recoverable for thirty days. The safety
    net was on the operation nobody performs by accident and missing from
    the one that is a single click beside "Deploy".
    """
    from sqlalchemy.orm import Session as OrmSession

    from app.migrations.runner import run_migrations
    from app.models import Strategy

    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    run_migrations(fake_config.vault_db_path)

    engine = main_module.make_engine(fake_config.vault_db_path)
    with OrmSession(engine) as session:
        session.add(
            Strategy(
                id="#GONE", data_source="dukascopy", asset="EURUSD", timeframe="M15",
                genome_json="{}", trades=120, score=70, grade="GOOD",
            )
        )
        session.commit()

    response = client.delete("/vault/%23GONE")
    assert response.status_code == 200
    assert response.json()["recoverable"] is True

    with OrmSession(engine) as session:
        row = session.get(Strategy, "#GONE")
        assert row is not None              # still there to be restored
        assert row.deleted_at is not None   # and out of the Vault listing

    # It is gone from the Vault, and deleting it twice is not a thing.
    # (%23 is the # every strategy id starts with; the app encodes it the
    # same way.)
    assert "#GONE" not in [row["id"] for row in client.get("/vault").json()]
    assert client.delete("/vault/%23GONE").status_code == 404


def test_sync_now_accepts_the_body_the_app_actually_sends():
    """"Sync now" had never worked.

    `symbols: list[str] | None` binds as the *whole* request body, so the
    only shape FastAPI accepted was a bare JSON array. The renderer posts
    `{}`, as it does everywhere else, and got 422 back. Every cached bar
    came from the startup sync, which calls the runner directly.
    """
    from unittest.mock import patch

    import app.main as main_module

    captured: dict = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            pass

        async def sync_all(self, symbols):
            captured["symbols"] = symbols
            return {}

    with patch.object(main_module, "DeltaSyncRunner", FakeRunner):
        with TestClient(main_module.app) as client:
            main_module._sync_state.running = False
            response = client.post("/data/sync", json={})
            assert response.status_code == 200, response.text
            assert response.json()["started"] is True

            main_module._sync_state.running = False
            response = client.post("/data/sync", json={"symbols": ["XAUUSD"]})
            assert response.status_code == 200, response.text
            assert response.json()["symbols"] == ["XAUUSD"]


def test_a_running_sync_can_be_stopped():
    """A multi-year backfill is tens of thousands of hourly files and runs
    for hours. `create_task` used to be called and its result dropped, so
    there was no handle to reach it by — and the startup sync relaunches
    on every open, so closing the app restarted it rather than stopping
    it."""
    import asyncio
    import time

    from unittest.mock import patch

    import app.main as main_module

    class SlowRunner:
        def __init__(self, **kwargs):
            pass

        async def sync_all(self, symbols):
            main_module._sync_state.running = True
            try:
                await asyncio.sleep(3600)
            finally:
                main_module._sync_state.running = False

    with patch.object(main_module, "DeltaSyncRunner", SlowRunner):
        with TestClient(main_module.app) as client:
            main_module._sync_state.running = False
            assert client.post("/data/sync", json={"symbols": ["XAUUSD"]}).status_code == 200

            # The task starts on the app's own loop, so wait for it to be
            # running before stopping it - otherwise this passes by
            # stopping nothing.
            for _ in range(100):
                if client.get("/data/sync/status").json()["running"]:
                    break
                time.sleep(0.05)
            assert client.get("/data/sync/status").json()["running"] is True, "sync never started"

            response = client.post("/data/sync/stop")
            assert response.status_code == 200, response.text
            assert response.json()["stopped"] is True
            assert client.get("/data/sync/status").json()["running"] is False

            # Nothing to stop is a 409, not a silent success.
            assert client.post("/data/sync/stop").status_code == 409


def test_status_keeps_the_last_campaign_once_it_finishes(tmp_path, monkeypatch):
    """BREED lost every detail of a campaign the moment it ended.

    /status returned campaign: null once nothing was running, so the
    screen's asset went blank and the live feed — which filters the Vault
    by the campaign's started_at — had nothing to filter by and showed
    "no strategies yet" over a campaign that had just written three.
    """
    import uuid
    from datetime import datetime, timedelta, timezone

    from sqlalchemy.orm import Session

    from app.db import make_engine
    from app.models import Campaign

    fake_config = _FakeConfig(tmp_path)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    run_migrations(fake_config.vault_db_path)
    monkeypatch.setattr(main_module, "_active_campaign_id", None)

    started = datetime.now(timezone.utc) - timedelta(minutes=5)
    with Session(make_engine(fake_config.vault_db_path)) as session:
        session.add(
            Campaign(
                id=uuid.uuid4().hex,
                started_at=started,
                finished_at=datetime.now(timezone.utc),
                asset="XAUUSD",
                timeframe="M15",
                phase=4,
                status="completed",
            )
        )
        session.commit()

    body = client.get("/status").json()

    assert body["state"] == "idle", "a finished campaign is not a running one"
    assert body["campaign"] is not None, "the last campaign has to survive its own completion"
    assert body["campaign"]["asset"] == "XAUUSD"
    assert body["campaign"]["started_at"] is not None, "the live feed filters the Vault by this"
