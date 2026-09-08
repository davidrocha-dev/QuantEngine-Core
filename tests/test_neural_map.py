import json

from app.engine.neural_map import MIN_STRATEGIES_FOR_MAP, compute_neural_map
from app.genome.gene_pool import GenePoolConfig
from app.genome.genome import GeneChoice, Genome, random_genome
from app.models import Strategy


def make_strategy(i: int, genome: Genome, *, score: int = 60, archetype: str = "MULTI_SIGNAL", asset: str = "EURUSD", timeframe: str = "H1") -> Strategy:
    return Strategy(
        id=f"strat{i}",
        data_source="dukascopy",
        asset=asset,
        timeframe=timeframe,
        score=score,
        grade="GOOD" if score >= 57 else "WEAK",
        archetype=archetype,
        return_pct=score * 0.5,
        max_dd_pct=10.0,
        genome_json=json.dumps(genome.to_dict()),
    )


def test_returns_none_below_minimum_population():
    pool = GenePoolConfig()
    strategies = [make_strategy(i, random_genome(pool)) for i in range(MIN_STRATEGIES_FOR_MAP - 1)]
    assert compute_neural_map(strategies) is None


def test_projects_and_clusters_a_sufficient_population():
    pool = GenePoolConfig()
    # Two visibly distinct genome "families" so clustering has something
    # real to separate, rather than one homogeneous blob.
    fade_genome = Genome(signal=GeneChoice("FADE", {}), execution=GeneChoice("MARKET", {}))
    breakout_genome = Genome(signal=GeneChoice("BREAKOUT", {}), execution=GeneChoice("MARKET", {}))

    strategies = [make_strategy(i, fade_genome, score=50 + i, archetype="MEAN_REVERTER") for i in range(5)]
    strategies += [make_strategy(i + 5, breakout_genome, score=60 + i, archetype="MULTI_SIGNAL") for i in range(5)]

    result = compute_neural_map(strategies)

    assert result is not None
    assert len(result.nodes) == 10
    assert {n.id for n in result.nodes} == {s.id for s in strategies}
    for node in result.nodes:
        assert 0.0 <= node.brightness <= 1.0
    assert len(result.clusters) >= 1
    total_members = sum(c.strategy_count for c in result.clusters)
    assert total_members == 10
    for cluster in result.clusters:
        assert cluster.archetype in ("MEAN_REVERTER", "MULTI_SIGNAL", "PATTERN_SPOTTER")
        assert cluster.profitable_pct >= 0.0


def test_skips_unparseable_genome_json_without_crashing():
    pool = GenePoolConfig()
    good = [make_strategy(i, random_genome(pool)) for i in range(MIN_STRATEGIES_FOR_MAP)]
    broken = Strategy(
        id="broken", data_source="dukascopy", asset="EURUSD", timeframe="H1",
        score=50, grade="WEAK", archetype="MULTI_SIGNAL", genome_json="not json",
    )
    result = compute_neural_map(good + [broken])
    assert result is not None
    assert "broken" not in {n.id for n in result.nodes}
