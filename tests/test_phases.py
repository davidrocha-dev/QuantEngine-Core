from app.engine.phases import (
    CandidateResult,
    evaluate_candidate,
    run_phase_0_screening,
    run_phase_1_stability,
    run_phase_2_expansion,
    run_phase_3_hybridization,
    run_phase_4_robustness,
    split_is_oos,
)
from app.genome.gene_pool import GenePoolConfig
from app.genome.genome import random_population
from tests.helpers import make_synthetic_ohlc


def test_split_is_oos_proportions_and_order():
    df = make_synthetic_ohlc(1000)
    is_df, oos_df = split_is_oos(df, 0.7)
    assert len(is_df) == 700
    assert len(oos_df) == 300
    assert is_df["timestamp"].iloc[-1] < oos_df["timestamp"].iloc[0]


def test_phase_0_screening_partitions_full_population():
    df = make_synthetic_ohlc(1500, seed=4, drift=0.0005, volatility=0.002)
    pool = GenePoolConfig()
    population = random_population(pool, 15)
    outcome = run_phase_0_screening(population, df, pool, timeframe="H1")
    assert len(outcome.survivors) + len(outcome.rejected) == 15
    assert outcome.passed


def test_phase_0_screening_parallel_path_matches_sequential():
    # REQ-DASH-1.5's duration estimate is scaled by CPU cores, so Phase 0 —
    # the one phase with no cross-candidate dependency — actually needs to
    # use them. Force the ProcessPoolExecutor branch (population size >=
    # the parallel threshold) and check it partitions the population
    # identically to the sequential path for the same inputs.
    df = make_synthetic_ohlc(1500, seed=4, drift=0.0005, volatility=0.002)
    pool = GenePoolConfig()
    population = random_population(pool, 200)

    sequential = run_phase_0_screening(population, df, pool, max_workers=1, timeframe="H1")
    parallel = run_phase_0_screening(population, df, pool, max_workers=2, timeframe="H1")

    assert len(parallel.survivors) + len(parallel.rejected) == 200
    assert [c.genome for c in parallel.survivors] == [c.genome for c in sequential.survivors]


def test_phase_1_and_2_generational_loop_preserves_population_size_and_tags_tribe():
    df = make_synthetic_ohlc(1500, seed=4, drift=0.0005, volatility=0.002)
    pool = GenePoolConfig()
    population = random_population(pool, 10)
    phase_0 = run_phase_0_screening(population, df, pool, timeframe="H1")
    seed = phase_0.survivors or [evaluate_candidate(g, df, pool, timeframe="H1") for g in population]

    checkpoints = []
    outcome_a = run_phase_1_stability(
        seed, pool, df, generations=2, top_n=5,
        on_generation_complete=lambda gen, pop: checkpoints.append((gen, len(pop))),
        timeframe="H1",
    )
    assert len(checkpoints) == 2
    assert all(c.tribe == "A" for c in outcome_a.survivors)
    assert len(outcome_a.survivors) <= 5

    outcome_b = run_phase_2_expansion(seed, pool, df, generations=2, top_n=5, timeframe="H1")
    assert all(c.tribe == "B" for c in outcome_b.survivors)


def test_phase_3_fails_when_all_management_genes_disabled():
    df = make_synthetic_ohlc(500)
    pool = GenePoolConfig()
    for name in ["EOD CLOSE", "FRIDAY PROFIT", "PARTIAL TP"]:
        pool.set_enabled("management", name, False)
    population = random_population(pool, 5)
    candidates = [evaluate_candidate(g, df, pool, timeframe="H1") for g in population]

    outcome = run_phase_3_hybridization(candidates, candidates, pool, df, num_children=5, timeframe="H1")

    assert not outcome.passed
    assert "Not Passed" in outcome.fail_reason


def test_phase_3_produces_tribe_c_children():
    df = make_synthetic_ohlc(800, seed=6, drift=0.0005, volatility=0.002)
    pool = GenePoolConfig()
    population = random_population(pool, 10)
    candidates = [evaluate_candidate(g, df, pool, timeframe="H1") for g in population]

    outcome = run_phase_3_hybridization(candidates[:5], candidates[5:], pool, df, num_children=6, timeframe="H1")

    assert outcome.passed
    assert len(outcome.survivors) == 6
    assert all(c.tribe == "C" for c in outcome.survivors)


def test_phase_4_returns_vault_candidates_with_hash_ids():
    df = make_synthetic_ohlc(3000, seed=8, drift=0.0008, volatility=0.0025)
    pool = GenePoolConfig()
    population = random_population(pool, 20)
    candidates = [evaluate_candidate(g, df, pool, timeframe="H1") for g in population]

    approved = run_phase_4_robustness(candidates, df, pool, point_size=0.0001, timeframe="H1")

    for vault_candidate in approved:
        assert vault_candidate.hash_id.startswith("#")
        assert vault_candidate.robustness.passed
