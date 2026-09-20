"""Walk-forward validation.

The 70/30 IS/OOS split (REQ-ENGINE-3.6) answers one question: did the edge
survive on data the search never saw? It answers it once, on one period.
A strategy can pass it by being right about one regime and lucky about its
boundary.

Walk-forward asks the same question repeatedly, on consecutive slices of
history: train window, then the window immediately after it, rolled
forward. What matters is not the average — a single outsized fold carries
an average — but *how many* of the test windows the strategy was
profitable in. That fraction is the PERSISTENCE column the Vault has been
rendering as "—" since Phase 8, because nothing computed it.

Nothing here re-optimises: the genome is fixed, so each fold is a pure
out-of-sample measurement of an already-chosen strategy. The training
windows are therefore not simulated at all — only the test windows are.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.engine.backtest import simulate_trades
from app.engine.metrics import compute_trade_stats
from app.genome.gene_pool import GenePoolConfig
from app.genome.genome import Genome

# Fewer folds than this is not evidence of anything, so persistence is
# reported as unknown rather than as a number that invites over-reading.
MIN_FOLDS = 3
DEFAULT_FOLDS = 5
# A test window shorter than this cannot produce enough trades to judge.
MIN_FOLD_BARS = 500


@dataclass
class Fold:
    index: int
    start: str
    end: str
    bars: int
    trades: int
    return_pct: float
    profit_factor: float
    max_dd_pct: float
    # True when the window overlaps the period the GA searched on. Those
    # folds still say something — a strategy that loses money inside its
    # own training period is hopeless — but they are not evidence of an
    # edge. Only `out_of_sample` folds are.
    in_sample: bool = True


@dataclass
class WalkForwardResult:
    folds: list[Fold]
    persistence_pct: float  # share of folds that ended profitable
    worst_fold_return_pct: float
    mean_fold_return_pct: float
    # Same measure over the folds that fall after the training period only.
    # None when the history holds no complete post-training fold.
    oos_persistence_pct: float | None = None

    def to_dict(self) -> dict:
        return {
            "persistence_pct": self.persistence_pct,
            "oos_persistence_pct": self.oos_persistence_pct,
            "worst_fold_return_pct": self.worst_fold_return_pct,
            "mean_fold_return_pct": self.mean_fold_return_pct,
            "folds": [vars(f) for f in self.folds],
        }


def fold_ranges(total_bars: int, folds: int = DEFAULT_FOLDS, min_fold_bars: int = MIN_FOLD_BARS) -> list[tuple[int, int]]:
    """Consecutive, non-overlapping test windows covering the history.

    The fold count shrinks on short histories rather than producing windows
    too small to mean anything; it returns [] when even `MIN_FOLDS` windows
    of `min_fold_bars` do not fit.
    """
    usable_folds = min(folds, total_bars // min_fold_bars)
    if usable_folds < MIN_FOLDS:
        return []
    size = total_bars // usable_folds
    return [(i * size, (i + 1) * size if i < usable_folds - 1 else total_bars) for i in range(usable_folds)]


def run_walk_forward(
    genome: Genome,
    df: pd.DataFrame,
    gene_pool: GenePoolConfig,
    folds: int = DEFAULT_FOLDS,
    initial_balance: float = 10_000.0,
    timeframe: str = "H1",
    train_bars: int | None = None,
    **backtest_kwargs,
) -> WalkForwardResult | None:
    """Runs the genome over consecutive slices of `df`. Returns None when
    the history is too short for the result to carry information.

    `train_bars` is the size of the period the GA searched on, so folds
    inside it can be labelled: persistence across windows the strategy was
    fitted on is a much weaker claim than persistence after them.
    """
    ranges = fold_ranges(len(df), folds)
    if not ranges:
        return None

    results: list[Fold] = []
    for index, (start, end) in enumerate(ranges):
        window = df.iloc[start:end].reset_index(drop=True)
        outcome = simulate_trades(
            window, genome, gene_pool, initial_balance=initial_balance, **backtest_kwargs
        )
        stats = compute_trade_stats(outcome, window, initial_balance, timeframe)
        timestamps = pd.DatetimeIndex(window["timestamp"])
        results.append(
            Fold(
                index=index,
                start=timestamps.min().isoformat(),
                end=timestamps.max().isoformat(),
                bars=len(window),
                trades=stats.trade_count,
                return_pct=round(stats.net_profit_pct, 2),
                profit_factor=round(stats.profit_factor, 2) if stats.profit_factor != float("inf") else 0.0,
                max_dd_pct=round(stats.max_dd_pct, 2),
                # A window straddling the boundary counts as in-sample: it
                # contains bars the search saw.
                in_sample=train_bars is None or start < train_bars,
            )
        )

    # A fold with no trades is not a profitable fold: a strategy that goes
    # silent for a whole period has not shown persistence in it.
    profitable = sum(1 for fold in results if fold.return_pct > 0)
    returns = [fold.return_pct for fold in results]

    oos_folds = [fold for fold in results if not fold.in_sample]
    oos_persistence = (
        round(100.0 * sum(1 for fold in oos_folds if fold.return_pct > 0) / len(oos_folds), 1) if oos_folds else None
    )

    return WalkForwardResult(
        folds=results,
        persistence_pct=round(100.0 * profitable / len(results), 1),
        worst_fold_return_pct=min(returns),
        mean_fold_return_pct=round(sum(returns) / len(returns), 2),
        oos_persistence_pct=oos_persistence,
    )
