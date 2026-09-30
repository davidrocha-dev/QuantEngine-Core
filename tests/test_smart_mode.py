import json
import random

from app.engine.smart_mode import (
    MAX_WEIGHT,
    MIN_STRATEGIES_TO_LEARN,
    MIN_WEIGHT,
    clear,
    compute_gene_fitness,
    load,
    save,
)
from app.genome.gene_pool import GenePoolConfig
from app.genome.genome import random_population
from app.models import Strategy


def make_strategy(strategy_id: str, signal: str, score: int) -> Strategy:
    genome = {
        "signal": {"name": signal, "params": {}},
        "bias": None,
        "filters": [],
        "execution": {"name": "MARKET2", "params": {}},
        "management": {"name": "EOD CLOSE", "params": {}},
    }
    return Strategy(
        id=strategy_id, data_source="dukascopy", asset="EURUSD", timeframe="M15",
        genome_json=json.dumps(genome), score=score,
    )


def vault(winner_count: int = 20, loser_count: int = 20) -> list[Strategy]:
    winners = [make_strategy(f"#W{i}", "BREAKOUT", 85) for i in range(winner_count)]
    losers = [make_strategy(f"#L{i}", "FADE", 45) for i in range(loser_count)]
    return winners + losers


def test_a_small_vault_teaches_nothing_and_says_so():
    fitness = compute_gene_fitness(vault(winner_count=2, loser_count=2))

    assert fitness.weights == {}
    assert str(MIN_STRATEGIES_TO_LEARN) in fitness.reason


def test_genes_from_high_scoring_strategies_are_weighted_up():
    fitness = compute_gene_fitness(vault())

    assert fitness.weight_for("signal", "BREAKOUT") > 1.0
    assert fitness.weight_for("signal", "FADE") < 1.0


def test_a_gene_that_never_produced_a_strategy_is_never_excluded():
    # "Has not worked yet" is not "cannot work" — a pool that collapses onto
    # last month's winners stops searching.
    fitness = compute_gene_fitness(vault())

    # Neutral, not the floor: no strategy yet is no evidence yet. The
    # floor made an untried gene rarer, so it stayed untried.
    assert fitness.weight_for("signal", "ORB") == 1.0
    assert all(MIN_WEIGHT <= weight <= MAX_WEIGHT for weight in fitness.weights.values())


def test_weights_cover_every_gene_in_the_catalog():
    fitness = compute_gene_fitness(vault())
    assert len(fitness.weights) == 46


def test_weights_survive_a_save_load_round_trip(tmp_path):
    original = compute_gene_fitness(vault())
    save(original, tmp_path)

    restored = load(tmp_path)

    assert restored.weights == original.weights
    assert restored.strategies_learned_from == original.strategies_learned_from


def test_loading_without_a_file_is_uniform_not_an_error(tmp_path):
    fitness = load(tmp_path)
    assert fitness.weights == {}
    assert fitness.weight_for("signal", "MACD") == 1.0


def test_reset_removes_what_was_learned(tmp_path):
    save(compute_gene_fitness(vault()), tmp_path)
    assert clear(tmp_path) is True
    assert clear(tmp_path) is False
    assert load(tmp_path).weights == {}


def test_weights_shift_which_genes_campaigns_draw():
    pool = GenePoolConfig()
    weights = {f"signal::{name}": 0.01 for name in pool.enabled_names("signal")}
    weights["signal::MACD"] = 50.0

    random.seed(3)
    population = random_population(pool, 60, weights)

    macd_share = sum(1 for genome in population if genome.signal.name == "MACD") / len(population)
    assert macd_share > 0.8


def test_uniform_sampling_when_no_weights_are_given():
    pool = GenePoolConfig()
    random.seed(4)
    population = random_population(pool, 120)

    assert len({genome.signal.name for genome in population}) > 5


def test_each_market_learns_from_its_own_strategies():
    """On gold H1 FADE is what works; on EURUSD M15 it is BREAKOUT. One
    pooled table let the market with the most rows set the weights for
    every other market."""
    gold = []
    for i in range(20):
        row = make_strategy(f"#GW{i}", "FADE", 90)
        row.asset, row.timeframe = "XAUUSD", "H1"
        gold.append(row)
    for i in range(10):
        row = make_strategy(f"#GL{i}", "BREAKOUT", 40)
        row.asset, row.timeframe = "XAUUSD", "H1"
        gold.append(row)

    fitness = compute_gene_fitness(vault() + gold)

    assert fitness.weights_for("XAUUSD", "H1")["signal::FADE"] > 1.0
    assert fitness.weights_for("XAUUSD", "H1")["signal::BREAKOUT"] < 1.0
    assert fitness.weights_for("EURUSD", "M15")["signal::BREAKOUT"] > 1.0
    # A market with no Vault of its own gets the pooled weights.
    assert fitness.weights_for("GBPUSD", "H4") == fitness.weights
