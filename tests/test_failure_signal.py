"""Learning from what failed — the other half of Smart Mode.

Smart Mode weights a gene by the strategies it appears in, which is
learning from winners. It cannot tell a gene nobody has tried from a
gene drawn four thousand times that has never once survived screening:
both are simply absent from the Vault, and they are not the same fact.
"""

from pathlib import Path

import pytest

from app.engine.failure_signal import (
    MIN_DRAWS_TO_JUDGE,
    PENALTY_FLOOR,
    GeneOutcomes,
    gene_keys,
    load,
    penalties,
    record_screening,
    save,
)
from app.genome.genome import GeneChoice, Genome


def genome_with(signal: str, execution: str = "MARKET2", filters: list[str] | None = None) -> Genome:
    return Genome(
        signal=GeneChoice(signal, {}),
        execution=GeneChoice(execution, {}),
        filters=[GeneChoice(name, {}) for name in (filters or [])],
    )


def test_every_layer_of_a_genome_is_counted():
    keys = gene_keys(genome_with("BREAKOUT", "LIMIT", ["SESSION", "ATR"]))

    assert "signal::BREAKOUT" in keys
    assert "execution::LIMIT" in keys
    assert "filter::SESSION" in keys and "filter::ATR" in keys


def test_screening_counts_what_went_in_and_what_came_out():
    outcomes = GeneOutcomes()
    drawn = [genome_with("BREAKOUT"), genome_with("FADE"), genome_with("FADE")]
    survived = [genome_with("BREAKOUT")]

    record_screening(outcomes, drawn, survived)

    assert outcomes.drawn["signal::FADE"] == 2
    assert outcomes.survived.get("signal::FADE", 0) == 0
    assert outcomes.survived["signal::BREAKOUT"] == 1


def test_a_gene_is_not_judged_on_a_handful_of_draws():
    """A run of bad luck looks exactly like a bad gene until the sample
    is big enough to tell them apart."""
    outcomes = GeneOutcomes(drawn={"signal::FADE": 10}, survived={"signal::FADE": 0})

    assert outcomes.survival_rate("signal::FADE") is None
    assert penalties(outcomes) == {}


def test_a_gene_that_keeps_failing_is_drawn_less_often():
    outcomes = GeneOutcomes(
        drawn={"signal::GOOD": MIN_DRAWS_TO_JUDGE, "signal::BAD": MIN_DRAWS_TO_JUDGE},
        survived={"signal::GOOD": 40, "signal::BAD": 0},
    )

    result = penalties(outcomes)

    assert result["signal::GOOD"] == 1.0          # the best performer sets the bar
    assert result["signal::BAD"] == PENALTY_FLOOR  # never zero
    assert result["signal::BAD"] > 0


def test_a_penalty_never_silences_a_gene_completely():
    """"Has failed so far" is not "cannot work". A gene that is useless
    in a trending market may be the one that matters when the regime
    turns, and a search that deleted it cannot find out."""
    # A peer that does survive, so there is something to be judged
    # against: one gene alone proves nothing about itself.
    outcomes = GeneOutcomes(
        drawn={"signal::NEVER": MIN_DRAWS_TO_JUDGE * 100, "signal::WORKS": MIN_DRAWS_TO_JUDGE},
        survived={"signal::NEVER": 0, "signal::WORKS": 30},
    )

    penalty = penalties(outcomes)["signal::NEVER"]

    assert penalty == PENALTY_FLOOR
    assert penalty > 0  # discouraged, never silenced


def test_genes_are_judged_against_each_other_not_an_absolute_bar():
    # Screening is a hard gate: 2% survival may be the best anything
    # achieves on a given market, and that is not a failing gene.
    outcomes = GeneOutcomes(
        drawn={"signal::A": MIN_DRAWS_TO_JUDGE, "signal::B": MIN_DRAWS_TO_JUDGE},
        survived={"signal::A": 4, "signal::B": 2},
    )

    result = penalties(outcomes)

    assert result["signal::A"] == 1.0
    assert PENALTY_FLOOR < result["signal::B"] < 1.0


def test_nothing_surviving_anything_says_nothing_about_any_gene(tmp_path):
    # That is a statement about the market or the costs, not the genes.
    outcomes = GeneOutcomes(
        drawn={"signal::A": MIN_DRAWS_TO_JUDGE, "signal::B": MIN_DRAWS_TO_JUDGE},
        survived={},
    )

    assert penalties(outcomes) == {}


def test_counts_survive_a_restart(tmp_path: Path):
    outcomes = GeneOutcomes(drawn={"signal::A": 5}, survived={"signal::A": 1}, campaigns=2)

    save(tmp_path, outcomes)
    reloaded = load(tmp_path)

    assert reloaded.drawn == {"signal::A": 5}
    assert reloaded.campaigns == 2


def test_a_corrupt_counter_file_does_not_stop_a_campaign(tmp_path: Path):
    (tmp_path / "gene_outcomes.json").write_text("{ not json", encoding="utf-8")

    reloaded = load(tmp_path)

    assert reloaded.drawn == {}  # back to drawing uniformly, not a crash


def test_the_two_signals_multiply(tmp_path, monkeypatch):
    """Smart Mode says which genes are in winners; this says which keep
    failing. A gene can be absent from the Vault for either reason."""
    from app.config import EngineConfig
    import app.main as main_module

    monkeypatch.setattr(main_module, "get_config", lambda: EngineConfig(data_dir=tmp_path))
    monkeypatch.setattr(
        main_module.smart_mode, "load", lambda _dir: type("F", (), {"weights": {"signal::BAD": 2.0}})()
    )
    save(
        tmp_path,
        GeneOutcomes(
            drawn={"signal::BAD": MIN_DRAWS_TO_JUDGE, "signal::OK": MIN_DRAWS_TO_JUDGE},
            survived={"signal::BAD": 0, "signal::OK": 20},
        ),
    )

    weights = main_module._learned_gene_weights()

    # Smart Mode liked it at 2.0; screening says it never survives.
    assert weights["signal::BAD"] == pytest.approx(2.0 * PENALTY_FLOOR)
