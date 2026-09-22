from app.engine.walk_forward import (
    DEFAULT_FOLDS,
    MIN_FOLD_BARS,
    MIN_FOLDS,
    MIN_OOS_FOLDS,
    _windows,
    fold_ranges,
    run_walk_forward,
)
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


def test_the_post_training_side_gets_more_than_one_window():
    """The regression this module exists to prevent.

    Splitting the whole history into DEFAULT_FOLDS equal windows and then
    labelling them put exactly one window after a 70% training boundary —
    on every history, at every length. Persistence measured on that was a
    percentage of one sample, so it could only ever read 0.0 or 100.0.
    """
    for total in (5_000, 20_000, 100_000):
        train = int(total * 0.7)
        after = _windows(train, total, DEFAULT_FOLDS, MIN_FOLD_BARS, MIN_OOS_FOLDS)
        assert len(after) >= MIN_OOS_FOLDS, f"{total} bars gave {len(after)} post-training windows"
        assert all(start >= train for start, _ in after)


def test_persistence_counts_only_the_windows_after_the_training_period():
    df = make_synthetic_ohlc(20_000, seed=3, drift=0.0004, volatility=0.002)
    pool = GenePoolConfig()
    genome = random_population(pool, 1)[0]
    train_bars = int(len(df) * 0.7)

    result = run_walk_forward(genome, df, pool, timeframe="H1", train_bars=train_bars)

    assert result is not None
    oos_folds = [fold for fold in result.folds if not fold.in_sample]
    assert len(oos_folds) >= MIN_OOS_FOLDS
    assert result.oos_fold_count == len(oos_folds)

    expected = round(100.0 * sum(1 for f in oos_folds if f.return_pct > 0) / len(oos_folds), 1)
    assert result.persistence_pct == expected
    # And the training folds, which used to make up four fifths of it, do
    # not move the number.
    assert any(fold.in_sample for fold in result.folds)


def test_worst_and_average_describe_the_same_windows_persistence_does():
    df = make_synthetic_ohlc(20_000, seed=5, drift=0.0004, volatility=0.002)
    pool = GenePoolConfig()
    genome = random_population(pool, 1)[0]

    result = run_walk_forward(genome, df, pool, timeframe="H1", train_bars=int(len(df) * 0.7))

    assert result is not None
    oos_returns = [fold.return_pct for fold in result.folds if not fold.in_sample]
    assert result.worst_fold_return_pct == min(oos_returns)
    assert result.mean_fold_return_pct == round(sum(oos_returns) / len(oos_returns), 2)


def test_folds_are_disjoint_and_indexed_in_order():
    df = make_synthetic_ohlc(20_000, seed=4)
    pool = GenePoolConfig()
    genome = random_population(pool, 1)[0]

    result = run_walk_forward(genome, df, pool, timeframe="H1", train_bars=int(len(df) * 0.7))

    assert result is not None
    assert sum(fold.bars for fold in result.folds) == len(df)
    assert [fold.index for fold in result.folds] == list(range(len(result.folds)))
    assert result.to_dict()["persistence_pct"] == result.persistence_pct


def test_persistence_is_unknown_rather_than_a_number_from_too_few_windows():
    # 4,000 bars with a 70% boundary leaves 1,200 for the post-training
    # side: two windows of MIN_FOLD_BARS, one short of MIN_OOS_FOLDS.
    df = make_synthetic_ohlc(4_000, seed=9)
    pool = GenePoolConfig()
    genome = random_population(pool, 1)[0]

    result = run_walk_forward(genome, df, pool, timeframe="H1", train_bars=int(len(df) * 0.7))

    assert result is not None
    assert result.persistence_pct is None
    assert result.oos_fold_count == 0
    assert result.worst_fold_return_pct is None
    # The training windows still ran, and still say something.
    assert result.in_sample_persistence_pct is not None


def test_without_train_bars_nothing_is_out_of_sample():
    df = make_synthetic_ohlc(20_000, seed=8)
    pool = GenePoolConfig()
    genome = random_population(pool, 1)[0]

    result = run_walk_forward(genome, df, pool, timeframe="H1")

    assert result is not None
    assert all(fold.in_sample for fold in result.folds)
    assert result.persistence_pct is None
