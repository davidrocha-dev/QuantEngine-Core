import random

from app.genome.gene_pool import GenePoolConfig
from app.genome.genome import Genome, crossover, mutate, random_genome, random_population


def test_random_genome_respects_disabled_genes():
    pool = GenePoolConfig()
    pool.set_enabled("signal", "MACD", False)
    random.seed(0)
    for _ in range(50):
        genome = random_genome(pool)
        assert genome.signal.name != "MACD"


def test_random_genome_raises_when_no_signal_genes_enabled():
    pool = GenePoolConfig()
    for name in list(pool.enabled_names("signal")):
        pool.set_enabled("signal", name, False)
    try:
        random_genome(pool)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_random_genome_allows_no_management_gene_when_pool_empties_it():
    pool = GenePoolConfig()
    for name in ["EOD CLOSE", "FRIDAY PROFIT", "PARTIAL TP"]:
        pool.set_enabled("management", name, False)
    genome = random_genome(pool)
    assert genome.management is None  # optional per-genome, see genome.py docstring


def test_random_population_size():
    pool = GenePoolConfig()
    population = random_population(pool, 25)
    assert len(population) == 25
    assert len({g.genome_id for g in population}) == 25  # unique ids


def test_genome_to_dict_from_dict_round_trip():
    pool = GenePoolConfig()
    genome = random_genome(pool)
    restored = Genome.from_dict(genome.to_dict())
    assert restored.to_dict() == genome.to_dict()


def test_crossover_inherits_layers_from_either_parent():
    pool = GenePoolConfig()
    parent_a = random_genome(pool)
    parent_b = random_genome(pool)
    child = crossover(parent_a, parent_b)
    assert child.signal.name in (parent_a.signal.name, parent_b.signal.name)
    assert child.execution.name in (parent_a.execution.name, parent_b.execution.name)


def test_mutate_can_change_genome_over_many_trials():
    pool = GenePoolConfig()
    genome = random_genome(pool)
    changed = False
    for _ in range(200):
        mutated = mutate(genome, pool, mutation_rate=0.5)
        if mutated.signal.name != genome.signal.name:
            changed = True
            break
    assert changed, "mutation never changed the signal gene across 200 trials"


def test_mutate_never_reintroduces_disabled_gene():
    pool = GenePoolConfig()
    pool.set_enabled("signal", "MACD", False)
    genome = random_genome(pool)
    for _ in range(100):
        genome = mutate(genome, pool, mutation_rate=0.9)
        assert genome.signal.name != "MACD"
