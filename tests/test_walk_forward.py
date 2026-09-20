from app.engine.walk_forward import MIN_FOLD_BARS, MIN_FOLDS, fold_ranges, run_walk_forward
from app.genome.gene_pool import GenePoolConfig
from app.genome.genome import random_population
from tests.helpers import make_synthetic_ohlc


def test_fold_ranges_cover_the_whole_history_without_gaps_or_overlap():
    ranges = fold_ranges(10_000, folds=5, min_fold_bars=500)

    assert len(ranges) == 5
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 10_000
    for (_, end), (next_start, _) in zip(ranges, ranges[1:]):
        assert end == next_start


def test_fold_count_shrinks_rather_than_producing_tiny_windows():
    # 1,800 bars cannot make 5 windows of 500, but it can make 3.
    ranges = fold_ranges(1_800, folds=5, min_fold_bars=500)
    assert len(ranges) == 3
    assert all(end - start >= 500 for start, end in ranges)


def test_no_folds_when_history_is_too_short_to_mean_anything():
    assert fold_ranges(MIN_FOLD_BARS * MIN_FOLDS - 1, min_fold_bars=MIN_FOLD_BARS) == []


def test_walk_forward_returns_none_on_short_history():
    df = make_synthetic_ohlc(400)
    pool = GenePoolConfig()
    genome = random_population(pool, 1)[0]

    assert run_walk_forward(genome, df, pool, timeframe="H1") is None


def test_walk_forward_reports_one_result_per_fold_and_a_consistent_persistence():
    df = make_synthetic_ohlc(4_000, seed=3, drift=0.0004, volatility=0.002)
    pool = GenePoolConfig()
    genome = random_population(pool, 1)[0]

    result = run_walk_forward(genome, df, pool, timeframe="H1")

    assert result is not None
    assert len(result.folds) >= MIN_FOLDS
    assert [fold.index for fold in result.folds] == list(range(len(result.folds)))
    profitable = sum(1 for fold in result.folds if fold.return_pct > 0)
    assert result.persistence_pct == round(100.0 * profitable / len(result.folds), 1)
    assert result.worst_fold_return_pct == min(fold.return_pct for fold in result.folds)


def test_walk_forward_folds_are_measured_out_of_sample_only():
    # Each fold's window is disjoint from the others, so a fold's numbers
    # cannot include bars another fold already reported on.
    df = make_synthetic_ohlc(4_000, seed=4)
    pool = GenePoolConfig()
    genome = random_population(pool, 1)[0]

    result = run_walk_forward(genome, df, pool, timeframe="H1")

    assert result is not None
    assert sum(fold.bars for fold in result.folds) == len(df)
    assert result.to_dict()["persistence_pct"] == result.persistence_pct


def test_folds_overlapping_the_training_period_are_labelled_and_excluded_from_oos_persistence():
    # Persistence measured on windows the GA searched on is not evidence of
    # an edge, so those folds are marked and reported separately.
    df = make_synthetic_ohlc(5_000, seed=7, drift=0.0004, volatility=0.002)
    pool = GenePoolConfig()
    genome = random_population(pool, 1)[0]
    train_bars = int(len(df) * 0.7)

    result = run_walk_forward(genome, df, pool, timeframe="H1", train_bars=train_bars)

    assert result is not None
    assert any(fold.in_sample for fold in result.folds)
    oos_folds = [fold for fold in result.folds if not fold.in_sample]
    if oos_folds:
        expected = round(100.0 * sum(1 for f in oos_folds if f.return_pct > 0) / len(oos_folds), 1)
        assert result.oos_persistence_pct == expected
    else:
        assert result.oos_persistence_pct is None


def test_without_train_bars_every_fold_is_treated_as_in_sample():
    df = make_synthetic_ohlc(4_000, seed=8)
    pool = GenePoolConfig()
    genome = random_population(pool, 1)[0]

    result = run_walk_forward(genome, df, pool, timeframe="H1")

    assert result is not None
    assert all(fold.in_sample for fold in result.folds)
    assert result.oos_persistence_pct is None
