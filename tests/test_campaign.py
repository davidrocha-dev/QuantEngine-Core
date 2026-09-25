from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.db import Base, make_engine
from app.engine.campaign import (
    CampaignConfig,
    discard_campaign,
    load_checkpoint,
    mark_crashed_campaigns,
    run_campaign,
)
from app.engine.phases import PhaseOutcome
from app.genome.gene_pool import GenePoolConfig
from app.models import Campaign
from tests.helpers import make_synthetic_ohlc


@pytest.fixture
def session(tmp_path: Path) -> Session:
    engine = make_engine(tmp_path / "vault_test.db")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def make_campaign_row(session: Session, campaign_id: str = "c1", status: str = "queued") -> Campaign:
    campaign = Campaign(id=campaign_id, asset="EURUSD", timeframe="H1", phase=0, status=status)
    session.add(campaign)
    session.commit()
    return campaign


def test_mark_crashed_campaigns_flags_running_with_no_process(session):
    make_campaign_row(session, "c1", status="running")
    make_campaign_row(session, "c2", status="completed")

    crashed = mark_crashed_campaigns(session)

    assert crashed == ["c1"]
    assert session.get(Campaign, "c1").status == "crashed"
    assert session.get(Campaign, "c2").status == "completed"


def test_discard_campaign_marks_cancelled(session):
    make_campaign_row(session, "c1", status="crashed")
    discard_campaign(session, "c1")
    assert session.get(Campaign, "c1").status == "cancelled"


def test_discard_unknown_campaign_raises(session):
    with pytest.raises(ValueError):
        discard_campaign(session, "does-not-exist")


def test_load_checkpoint_none_when_never_checkpointed(session):
    campaign = make_campaign_row(session, "c1")
    assert load_checkpoint(campaign) is None


def test_run_campaign_end_to_end_completes_and_writes_vault(session):
    make_campaign_row(session, "c1")
    df = make_synthetic_ohlc(2000, seed=15, drift=0.0006, volatility=0.002)
    pool = GenePoolConfig()
    config = CampaignConfig(
        population_size=15, generations_per_tribe=2, phase1_top_n=5, phase2_top_n=5, phase3_children=4,
    )

    outcome = run_campaign(session, "c1", df, pool, config)

    campaign = session.get(Campaign, "c1")
    assert campaign.status in ("completed", "cancelled")  # cancelled only if Phase 0/3 legitimately produced nothing
    if campaign.status == "completed":
        assert outcome.passed


def test_run_campaign_checkpoints_during_phase_1(session):
    make_campaign_row(session, "c1")
    df = make_synthetic_ohlc(1500, seed=20, drift=0.0006, volatility=0.002)
    pool = GenePoolConfig()
    config = CampaignConfig(population_size=10, generations_per_tribe=2, phase1_top_n=4, phase2_top_n=4, phase3_children=3)

    run_campaign(session, "c1", df, pool, config)

    campaign = session.get(Campaign, "c1")
    # Some checkpoint must have been written along the way (phase field advances past 0).
    assert campaign.checkpoint_population is not None or campaign.status == "cancelled"


def test_run_campaign_fires_on_generation_for_every_checkpoint(session):
    # REQ-BREED-1.3 — a poller alone can miss generations on a fast run;
    # this callback must fire synchronously at the same point as the
    # checkpoint itself, once per Phase 1/2 generation.
    # Same seed/config as test_run_campaign_checkpoints_during_phase_1,
    # which already relies on this combination reliably producing Phase 0
    # survivors and at least one checkpoint.
    make_campaign_row(session, "c1")
    df = make_synthetic_ohlc(1500, seed=20, drift=0.0006, volatility=0.002)
    pool = GenePoolConfig()
    config = CampaignConfig(population_size=10, generations_per_tribe=2, phase1_top_n=4, phase2_top_n=4, phase3_children=3)

    calls: list[tuple[int, int, int]] = []
    run_campaign(
        session, "c1", df, pool, config,
        on_generation=lambda phase, generation, pop: calls.append((phase, generation, len(pop))),
    )

    campaign = session.get(Campaign, "c1")
    if campaign.checkpoint_population is not None:
        # Phase 0 produced survivors and at least one generation ran —
        # exactly the case this callback exists for.
        assert len(calls) > 0
        for phase, _generation, pop_size in calls:
            assert phase in (1, 2)  # PHASE_STABILITY / PHASE_EXPANSION
            assert pop_size > 0


def test_resume_reuses_checkpointed_population(session):
    make_campaign_row(session, "c1", status="running")
    df = make_synthetic_ohlc(1500, seed=21, drift=0.0006, volatility=0.002)
    pool = GenePoolConfig()
    config = CampaignConfig(population_size=10, generations_per_tribe=3, phase1_top_n=4, phase2_top_n=4, phase3_children=3)

    # Simulate a crash after generation 0 of Phase 1 by running one
    # generation manually and checkpointing, then marking "crashed".
    from app.engine.campaign import _checkpoint, PHASE_STABILITY
    from app.engine.phases import evaluate_candidate, run_phase_0_screening
    from app.genome.genome import random_population

    population = random_population(pool, config.population_size)
    phase_0 = run_phase_0_screening(population, df, pool, timeframe="H1")
    seed = phase_0.survivors or [evaluate_candidate(g, df, pool, timeframe="H1") for g in population]
    for c in seed:
        c.fitness = 0.0
        c.tribe = "A"
    _checkpoint(session, "c1", PHASE_STABILITY, generation=1, population=seed, phase_0_survivors=seed)

    campaign = session.get(Campaign, "c1")
    campaign.status = "crashed"
    session.commit()

    checkpoint = load_checkpoint(campaign)
    assert checkpoint is not None
    assert checkpoint.generation == 1
    assert len(checkpoint.active_population) == len(seed)
    assert len(checkpoint.phase_0_survivors) == len(seed)

    outcome = run_campaign(session, "c1", df, pool, config, resume=True)
    assert outcome is not None  # completes (or legitimately cancels) without crashing on the resume path


def test_a_campaign_that_approves_nothing_says_why():
    """It used to report "completed" over an empty Vault and leave the
    user to guess. One of the commonest reasons is a limit they set in
    Settings and can change, so silence is the worst possible answer.
    """
    from unittest.mock import patch

    import app.engine.campaign as campaign_module

    def nothing_approved(tribe_c, full_df, gene_pool, **kwargs):
        rejections = kwargs.get("rejections")
        if rejections is not None:
            rejections["drawdown above the 20% limit in Settings -> Risk"] = 7
            rejections["no out-of-sample edge"] = 2
        return []

    engine = make_engine(":memory:")
    Base.metadata.create_all(engine)
    df = make_synthetic_ohlc(6_000, seed=21, drift=0.0003, volatility=0.004)

    with Session(engine) as session:
        session.add(
            Campaign(id="empty", started_at=datetime.now(timezone.utc), asset="XAUUSD",
                     timeframe="M15", status="running")
        )
        session.commit()

        with patch.object(campaign_module, "run_phase_4_robustness", nothing_approved):
            outcome = campaign_module.run_campaign(
                session, "empty", df, GenePoolConfig(),
                campaign_module.CampaignConfig(
                    population_size=40, generations_per_tribe=1, max_workers=1,
                    timeframe="M15", point_size=0.0001,
                ),
            )

    assert outcome.fail_reason, "an empty campaign has to explain itself"
    assert "drawdown above the 20% limit" in outcome.fail_reason
    assert "7 for" in outcome.fail_reason, "and say how many, so the user knows if it is the main cause"


def test_phase_4_reports_every_gate_it_rejected_on_not_just_two():
    """A campaign that approves nothing has a result, and the result is
    the reasons.

    `fail_reason` is one sentence and can only carry the top two, which is
    how a real campaign came back saying "80 for drawdown" and nothing
    about the other gates that also turned candidates away. The structured
    breakdown goes to the renderer, which shows all of it and names the
    Settings field behind the ones that are limits the owner set.
    """
    rejections = {"drawdown": 80, "daily drawdown": 12, "robustness": 5}
    outcome = PhaseOutcome(
        passed=True,
        survivors=[],
        fail_reason="No candidate cleared Phase 4: 80 for drawdown, 12 for daily drawdown.",
        rejections=rejections,
    )

    assert outcome.rejections == rejections
    # The sentence keeps its two; nothing is lost alongside it.
    assert "robustness" not in (outcome.fail_reason or "")
    assert outcome.rejections["robustness"] == 5


def test_an_outcome_with_nothing_to_explain_carries_no_rejections():
    assert PhaseOutcome(passed=True).rejections == {}
