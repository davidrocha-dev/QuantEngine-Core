import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.broker import mt5_install
from app.codegen.mql5 import _magic_number
from app.db import Base, make_engine
from app.engine import auto_deploy
from app.engine.auto_deploy import MIN_REPLACEMENT_SCORE, find_replacement, replace_degraded
from app.genome.genes import ALL_GENES
from app.models import LiveTrade, Strategy


@pytest.fixture
def session(tmp_path: Path) -> Session:
    engine = make_engine(tmp_path / "vault.db")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


@pytest.fixture(autouse=True)
def fake_terminal(tmp_path, monkeypatch):
    """Installs into a temp folder instead of a real MT5 terminal."""
    terminal = tmp_path / "terminal"
    (terminal / "MQL5").mkdir(parents=True)
    monkeypatch.setattr(mt5_install, "terminal_data_dirs", lambda: [terminal])
    monkeypatch.setattr(mt5_install, "find_metaeditor", lambda: None)
    return terminal


def genome_json() -> str:
    return json.dumps(
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


def make_strategy(session: Session, strategy_id: str, score: int = 70, deployed: bool = False, **overrides) -> Strategy:
    defaults = dict(
        id=strategy_id, data_source="dukascopy", asset="EURUSD", timeframe="M15",
        genome_json=genome_json(), trades=120, return_pct=18.0, score=score,
        grade="GOOD" if score >= 57 else "WEAK", deployed=deployed,
    )
    defaults.update(overrides)
    row = Strategy(**defaults)
    session.add(row)
    session.commit()
    return row


def make_degraded(session: Session, strategy: Strategy) -> None:
    """Live losses on a strategy whose backtest was profitable."""
    for index in range(12):
        session.add(
            LiveTrade(
                deal_ticket=f"{strategy.id}-{index}", strategy_id=strategy.id, magic=_magic_number(strategy.id),
                symbol=strategy.asset, closed_at=datetime.now(timezone.utc) - timedelta(hours=index),
                volume=0.1, profit=-5.0,
            )
        )
    session.commit()


def test_a_degraded_bot_is_replaced_by_the_best_candidate_for_the_same_market(session):
    degraded = make_strategy(session, "#OLD", score=60, deployed=True)
    make_strategy(session, "#BETTER", score=88)
    make_strategy(session, "#GOOD", score=70)
    make_degraded(session, degraded)

    actions = replace_degraded(session)

    assert len(actions) == 1
    assert actions[0].retired_id == "#OLD"
    assert actions[0].installed_id == "#BETTER"
    assert session.get(Strategy, "#OLD").deployed is False
    assert session.get(Strategy, "#BETTER").deployed is True


def test_the_replacement_starts_in_incubation_like_any_new_deploy(session):
    degraded = make_strategy(session, "#OLD", score=60, deployed=True, incubation_stage="FULL RISK")
    make_strategy(session, "#NEW", score=80)
    make_degraded(session, degraded)

    replace_degraded(session)

    assert session.get(Strategy, "#NEW").incubation_stage is None


def test_a_healthy_bot_is_never_replaced(session):
    strategy = make_strategy(session, "#FINE", score=70, deployed=True)
    make_strategy(session, "#OTHER", score=90)
    for index in range(12):
        session.add(
            LiveTrade(
                deal_ticket=f"ok-{index}", strategy_id=strategy.id, magic=_magic_number(strategy.id),
                symbol="EURUSD", closed_at=datetime.now(timezone.utc), volume=0.1, profit=4.0,
            )
        )
    session.commit()

    assert replace_degraded(session) == []
    assert session.get(Strategy, "#FINE").deployed is True


def test_a_bot_with_too_few_live_trades_is_not_replaced(session):
    # Absence of evidence is not evidence of decay.
    degraded = make_strategy(session, "#YOUNG", score=70, deployed=True)
    make_strategy(session, "#OTHER", score=90)
    session.add(
        LiveTrade(
            deal_ticket="only-one", strategy_id=degraded.id, magic=_magic_number(degraded.id),
            symbol="EURUSD", closed_at=datetime.now(timezone.utc), volume=0.1, profit=-50.0,
        )
    )
    session.commit()

    assert replace_degraded(session) == []
    assert session.get(Strategy, "#YOUNG").deployed is True


def test_a_degraded_bot_with_no_successor_is_left_running(session):
    # Retiring it would leave a gap the user never asked for.
    degraded = make_strategy(session, "#OLD", score=60, deployed=True)
    make_degraded(session, degraded)

    actions = replace_degraded(session)

    assert actions[0].installed_id is None
    assert "left running" in actions[0].detail
    assert session.get(Strategy, "#OLD").deployed is True


def test_a_mediocre_candidate_does_not_qualify_as_a_replacement(session):
    degraded = make_strategy(session, "#OLD", score=60, deployed=True)
    make_strategy(session, "#MEH", score=MIN_REPLACEMENT_SCORE - 1)
    make_degraded(session, degraded)

    actions = replace_degraded(session)

    assert actions[0].installed_id is None
    assert session.get(Strategy, "#MEH").deployed is False


def test_replacement_stays_on_the_same_symbol_and_timeframe(session):
    degraded = make_strategy(session, "#OLD", score=60, deployed=True)
    make_strategy(session, "#WRONG_MARKET", score=95, asset="GBPUSD")
    make_strategy(session, "#WRONG_TF", score=94, timeframe="H1")
    right = make_strategy(session, "#RIGHT", score=70)
    make_degraded(session, degraded)

    assert find_replacement(session, degraded).id == right.id


def test_the_old_ea_file_is_removed_when_it_is_retired(session, fake_terminal):
    degraded = make_strategy(session, "#OLD", score=60, deployed=True)
    make_strategy(session, "#NEW", score=85)
    make_degraded(session, degraded)

    experts = fake_terminal / mt5_install.EXPERTS_SUBDIR
    experts.mkdir(parents=True, exist_ok=True)
    (experts / "QE_OLD.mq5").write_text("// old", encoding="utf-8")

    replace_degraded(session)

    assert not (experts / "QE_OLD.mq5").exists()
    assert (experts / "QE_NEW.mq5").exists()


def test_nothing_is_retired_when_the_replacement_cannot_be_installed(session, monkeypatch):
    from app.broker.mt5_install import InstallResult

    degraded = make_strategy(session, "#OLD", score=60, deployed=True)
    make_strategy(session, "#NEW", score=85)
    make_degraded(session, degraded)
    monkeypatch.setattr(auto_deploy, "install_ea", lambda *a, **k: InstallResult(detail="No terminal found."))

    actions = replace_degraded(session)

    assert actions[0].installed_id is None
    assert session.get(Strategy, "#OLD").deployed is True
    assert session.get(Strategy, "#NEW").deployed is False


def test_a_replacement_says_the_old_bot_is_still_on_its_chart(session):
    """Deleting an EA's files does not stop a chart already running it —
    uninstall_ea documents exactly that, and nothing in this app can reach
    into a chart. The action used to report "Retired X and installed Y",
    so nobody had a reason to go and stop X: two bots on the same market,
    with the Vault counting one."""
    degraded = make_strategy(session, "#OLD", score=60, deployed=True)
    make_strategy(session, "#BETTER", score=88)
    make_degraded(session, degraded)

    actions = replace_degraded(session)

    assert len(actions) == 1
    action = actions[0]
    assert action.installed_id == "#BETTER"
    assert action.needs_manual_removal is True
    assert "remove it from its chart" in action.detail


def test_the_uninstall_filename_comes_from_the_generator():
    """A second spelling of the generator's naming rule deletes nothing
    and still reports success."""
    from app.codegen.mql5 import _safe_name
    from app.engine.auto_deploy import _ea_filename

    for strategy_id in ("#C8B8C5", "#abc123", "#0E6C97"):
        assert _ea_filename(strategy_id) == f"{_safe_name(strategy_id)}.mq5"
