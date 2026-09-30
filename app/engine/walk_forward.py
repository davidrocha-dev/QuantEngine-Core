"""Walk-forward validation.

The 70/30 IS/OOS split (REQ-ENGINE-3.6) answers one question: did the edge
survive on data the search never saw? It answers it once, on one period.
A strategy can pass it by being right about one regime and lucky about its
boundary.

Walk-forward asks the same question repeatedly, on consecutive slices of
history. What matters is not the average — a single outsized fold carries
an average — but *how many* of the test windows the strategy was
profitable in. That fraction is the PERSISTENCE column in the Vault.

Only windows falling after the training period are evidence of an edge.
This module used to say so and then not do it: it laid DEFAULT_FOLDS equal
windows across the whole history and reported the share of *all* of them
that were profitable. With a 70% training split and five folds, four
windows always began inside the training period and exactly one did not —
on every history, at every length. The gated number was therefore
four-fifths made of windows the search had fitted on, and the honest one
printed beside it was a percentage of a single sample: it could only ever
read 0.0 or 100.0.

The windows are now laid out separately on each side of the boundary.
Persistence means the post-training windows and nothing else, and is None
rather than a number when too few of them fit to carry information. The
training windows are still simulated and still reported — a strategy that
loses money inside its own training period is hopeless — but they are
labelled, and they gate nothing.

Nothing here re-optimises: the genome is fixed, so every fold is a pure
measurement of an already-chosen strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from app.data.provider import TIMEFRAME_MINUTES
from app.engine.backtest import BacktestResult, simulate_trades
from app.engine.metrics import compute_trade_stats
from app.genome.gene_pool import GenePoolConfig
from app.genome.genome import Genome

# Fewer folds than this is not evidence of anything, so persistence is
# reported as unknown rather than as a number that invites over-reading.
MIN_FOLDS = 3
# The same floor applied where it decides something: post-training
# windows. Three is not many. One, which is what this used to be, is not
# a fraction at all.
MIN_OOS_FOLDS = 3
DEFAULT_FOLDS = 5
# A test window shorter than this cannot produce enough trades to judge.
MIN_FOLD_BARS = 500


# History given to the indicators before a window starts: the longest
# lookback a gene can draw is ADR EXHAUST's 28 broker days (SMA's 200
# bars is shorter on every timeframe below H4), with a margin for Wilder
# smoothing to settle. Never less than this many bars.
WARMUP_DAYS = 35
MIN_WARMUP_BARS = 300


def warmup_bars(timeframe: str) -> int:
    bars_per_day = 1440 // TIMEFRAME_MINUTES.get(timeframe, 60)
    return max(MIN_WARMUP_BARS, WARMUP_DAYS * bars_per_day)


def simulate_window(
    df: pd.DataFrame,
    start: int,
    end: int,
    genome: Genome,
    gene_pool: GenePoolConfig,
    timeframe: str = "H1",
    **backtest_kwargs,
) -> tuple[BacktestResult, pd.DataFrame]:
    """Trades `df[start:end]` with the bars before `start` as indicator
    history, and returns the result and the window it describes.

    Slicing the window out and simulating it alone restarted every
    indicator at its first bar. On an H1 walk-forward fold of ~900 bars a
    200-bar SMA was blind for a fifth of the fold and a 28-day ADR for
    most of it - trades the EA, which always has its chart behind it,
    would have taken.
    """
    lo = max(0, start - warmup_bars(timeframe))
    offset = start - lo
    frame = df.iloc[lo:end].reset_index(drop=True)
    result = simulate_trades(frame, genome, gene_pool, start_idx=offset, **backtest_kwargs)
    trimmed = BacktestResult(
        trades=[replace(t, entry_idx=t.entry_idx - offset, exit_idx=t.exit_idx - offset) for t in result.trades],
        equity_curve=result.equity_curve[offset:],
        final_balance=result.final_balance,
        margin_breach=result.margin_breach,
    )
    return trimmed, df.iloc[start:end].reset_index(drop=True)


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
    # True when the window falls inside the period the search ran on.
    in_sample: bool = True


@dataclass
class WalkForwardResult:
    folds: list[Fold]
    # Share of the *post-training* folds that ended profitable, or None
    # when fewer than MIN_OOS_FOLDS of them fit in the history. Everything
    # that gates on persistence reads this one.
    persistence_pct: float | None
    worst_fold_return_pct: float | None
    mean_fold_return_pct: float | None
    # Display only. A strategy unprofitable inside its own training period
    # is worth seeing; a strategy profitable there has shown nothing.
    in_sample_persistence_pct: float | None = None
    oos_fold_count: int = 0

    def to_dict(self) -> dict:
        return {
            "persistence_pct": self.persistence_pct,
            "in_sample_persistence_pct": self.in_sample_persistence_pct,
            "oos_fold_count": self.oos_fold_count,
            "worst_fold_return_pct": self.worst_fold_return_pct,
            "mean_fold_return_pct": self.mean_fold_return_pct,
            "folds": [vars(f) for f in self.folds],
        }


def _windows(lo: int, hi: int, folds: int, min_fold_bars: int, min_folds: int) -> list[tuple[int, int]]:
    """Up to `folds` consecutive, non-overlapping windows covering [lo, hi).

    Returns [] when even `min_folds` windows of `min_fold_bars` do not fit,
    rather than producing windows too small to mean anything.
    """
    span = hi - lo
    usable = min(folds, span // min_fold_bars) if min_fold_bars else folds
    if usable < min_folds:
        return []
    size = span // usable
    return [(lo + i * size, lo + (i + 1) * size if i < usable - 1 else hi) for i in range(usable)]


def fold_ranges(
    total_bars: int, folds: int = DEFAULT_FOLDS, min_fold_bars: int = MIN_FOLD_BARS
) -> list[tuple[int, int]]:
    """Consecutive test windows covering a whole history."""
    return _windows(0, total_bars, folds, min_fold_bars, MIN_FOLDS)


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

    `train_bars` is the size of the period the search ran on. Windows are
    laid out on each side of that boundary separately, so the post-training
    side gets several of them instead of the single one that splitting the
    whole history always produced.
    """
    train_end = len(df) if train_bars is None else max(0, min(train_bars, len(df)))

    is_ranges = _windows(0, train_end, folds, MIN_FOLD_BARS, MIN_FOLDS)
    oos_ranges = _windows(train_end, len(df), folds, MIN_FOLD_BARS, MIN_OOS_FOLDS)
    if not is_ranges and not oos_ranges:
        return None

    results: list[Fold] = []
    for start, end in [*is_ranges, *oos_ranges]:
        outcome, window = simulate_window(
            df, start, end, genome, gene_pool, timeframe=timeframe, initial_balance=initial_balance, **backtest_kwargs
        )
        stats = compute_trade_stats(outcome, window, initial_balance, timeframe)
        timestamps = pd.DatetimeIndex(window["timestamp"])
        results.append(
            Fold(
                index=len(results),
                start=timestamps.min().isoformat(),
                end=timestamps.max().isoformat(),
                bars=len(window),
                trades=stats.trade_count,
                return_pct=round(stats.net_profit_pct, 2),
                profit_factor=round(stats.profit_factor, 2) if stats.profit_factor != float("inf") else 0.0,
                max_dd_pct=round(stats.max_dd_pct, 2),
                in_sample=start < train_end,
            )
        )

    # A fold with no trades is not a profitable fold: a strategy that goes
    # silent for a whole period has not shown persistence in it.
    def share_profitable(rows: list[Fold]) -> float | None:
        return round(100.0 * sum(1 for f in rows if f.return_pct > 0) / len(rows), 1) if rows else None

    oos_folds = [fold for fold in results if not fold.in_sample]
    is_folds = [fold for fold in results if fold.in_sample]
    oos_returns = [fold.return_pct for fold in oos_folds]

    return WalkForwardResult(
        folds=results,
        persistence_pct=share_profitable(oos_folds),
        # Worst and average describe the same windows persistence does.
        # Averaging the training folds in flattered both.
        worst_fold_return_pct=min(oos_returns) if oos_returns else None,
        mean_fold_return_pct=round(sum(oos_returns) / len(oos_returns), 2) if oos_returns else None,
        in_sample_persistence_pct=share_profitable(is_folds),
        oos_fold_count=len(oos_folds),
    )
