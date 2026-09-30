"""Trade-by-trade parity between the backtest and a MetaTrader 5 report.

parity.py compares rates - how often live trading happens against how
often the backtest says it should. That catches an EA on the wrong chart;
it cannot say *which* trades differ, or from when. This lays the
backtest's trade list over the Strategy Tester's for the same period and
pairs them up, so the first trade that differs is a date someone can go
and look at.

Pairing: the same direction, and an MT5 entry inside the backtest trade's
entry bar (market entries fill at its open, resting orders somewhere in
it), allowing one bar either side for a fill that lands on a boundary.
Each trade pairs at most once, greedily in time order.

Before pairing, the MT5 times are tried at whole-hour shifts. If one of
them pairs far better than no shift at all, the broker's server clock is
not the one in Settings -> Broker caps -> Server time, and that is the
fix - the day and hour rules in the strategy fire in the wrong place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.data.broker_clock import bar_clock
from app.engine.backtest import Trade
from app.engine.mt5_report import Mt5Trade

# Shifts tried when looking for a clock mismatch, in hours.
CLOCK_SHIFTS_HOURS = range(-4, 5)
# A shift has to pair this many more trades than none before it is
# reported as the clock being off - one or two could be coincidence.
CLOCK_SHIFT_MIN_GAIN = 0.2
MAX_LISTED = 25


@dataclass
class TradeRow:
    entry_time: str
    direction: int
    entry_price: float
    outcome: float  # R on the backtest side, account-currency profit on MT5's

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class TradeParityReport:
    period_start: str
    period_end: str
    backtest_trades: int
    mt5_trades: int
    matched: int
    match_pct: float
    # Matched pairs whose outcome had the same sign (both won or both lost).
    same_outcome_pct: float | None
    # Median of (MT5 entry - backtest entry), in the instrument's points,
    # on matched pairs. Positive for longs means MT5 paid more.
    entry_slippage_points: float | None
    clock_shift_hours: int  # 0 when the configured clock pairs best
    first_divergence: str | None
    only_backtest: list[TradeRow] = field(default_factory=list)
    only_mt5: list[TradeRow] = field(default_factory=list)
    verdict: str = ""

    def to_dict(self) -> dict:
        out = {k: v for k, v in vars(self).items() if k not in ("only_backtest", "only_mt5")}
        out["only_backtest"] = [row.to_dict() for row in self.only_backtest]
        out["only_mt5"] = [row.to_dict() for row in self.only_mt5]
        return out


def _pair(
    bt_times: np.ndarray, bt_dirs: np.ndarray, mt_times: np.ndarray, mt_dirs: np.ndarray, bar: np.timedelta64
) -> list[tuple[int, int]]:
    """Greedy one-to-one pairing, both lists in time order."""
    pairs: list[tuple[int, int]] = []
    used = np.zeros(len(mt_times), dtype=bool)
    start = 0
    for i, (entry, direction) in enumerate(zip(bt_times, bt_dirs)):
        lo, hi = entry - bar, entry + 2 * bar
        while start < len(mt_times) and mt_times[start] < lo:
            start += 1
        best, best_gap = None, None
        j = start
        while j < len(mt_times) and mt_times[j] < hi:
            if not used[j] and mt_dirs[j] == direction:
                gap = abs(mt_times[j] - entry)
                if best is None or gap < best_gap:
                    best, best_gap = j, gap
            j += 1
        if best is not None:
            used[best] = True
            pairs.append((i, best))
    return pairs


def compare_trades(
    backtest_trades: list[Trade],
    df: pd.DataFrame,
    mt5_trades: list[Mt5Trade],
    bar_minutes: int,
    point_size: float,
) -> TradeParityReport:
    """`backtest_trades` were simulated on `df`; `mt5_trades` come from the
    report. Only backtest trades inside the report's period are compared."""
    clock = bar_clock(df)
    bar = np.timedelta64(bar_minutes, "m")

    mt5_sorted = sorted(mt5_trades, key=lambda t: t.entry_time)
    mt_times = np.array([t.entry_time.to_datetime64() for t in mt5_sorted], dtype="datetime64[ns]")
    mt_dirs = np.array([t.direction for t in mt5_sorted])
    last_mt5 = max((t.exit_time or t.entry_time) for t in mt5_sorted)
    period_start, period_end = mt5_sorted[0].entry_time, last_mt5

    def backtest_side(shift: np.timedelta64):
        """Backtest trades inside the report's period, with MT5's clock moved by `shift`."""
        lo, hi = period_start + pd.Timedelta(shift) - pd.Timedelta(bar), period_end + pd.Timedelta(shift)
        chosen = [t for t in backtest_trades if lo <= clock[t.entry_idx] <= hi]
        times = np.array([clock[t.entry_idx].to_datetime64() for t in chosen], dtype="datetime64[ns]")
        return chosen, times, np.array([t.direction for t in chosen])

    def exact_hits(pairs, times, shifted) -> int:
        """Pairs whose MT5 entry falls inside the backtest's own entry bar."""
        return sum(1 for i, j in pairs if times[i] <= shifted[j] < times[i] + bar)

    # Which whole-hour shift of the MT5 clock lines the trades up best. The
    # score is exact-bar hits, so a neighbouring shift that only pairs by
    # the one-bar tolerance does not win.
    in_period, bt_times, bt_dirs = backtest_side(np.timedelta64(0, "h"))
    pairs = _pair(bt_times, bt_dirs, mt_times, mt_dirs, bar)
    best_score, shift_hours = exact_hits(pairs, bt_times, mt_times), 0
    denominator = max(len(in_period), len(mt5_sorted), 1)
    for hours in CLOCK_SHIFTS_HOURS:
        if hours == 0:
            continue
        shift = np.timedelta64(hours, "h")
        chosen, times, dirs = backtest_side(shift)
        shifted = mt_times + shift
        score = exact_hits(_pair(times, dirs, shifted, mt_dirs, bar), times, shifted)
        if score > best_score and (score - best_score) / denominator >= CLOCK_SHIFT_MIN_GAIN:
            best_score, shift_hours = score, hours
    # A shift is only a diagnosis: the figures below are for the clock as configured.
    reported_pairs = pairs

    matched_bt = {i for i, _ in reported_pairs}
    matched_mt = {j for _, j in reported_pairs}
    only_bt = [i for i in range(len(in_period)) if i not in matched_bt]
    only_mt = [j for j in range(len(mt5_sorted)) if j not in matched_mt]

    same_outcome = [
        (in_period[i].r_multiple > 0) == (mt5_sorted[j].net_profit > 0) for i, j in reported_pairs
    ]
    slippage = [
        (mt5_sorted[j].entry_price - in_period[i].entry_price) * in_period[i].direction / point_size
        for i, j in reported_pairs
        if point_size > 0
    ]

    divergences = [bt_times[i] for i in only_bt] + [mt_times[j] for j in only_mt]
    first_divergence = pd.Timestamp(min(divergences)).isoformat() if divergences else None

    matched = len(reported_pairs)
    denominator = max(len(in_period), len(mt5_sorted), 1)
    report = TradeParityReport(
        period_start=period_start.isoformat(),
        period_end=period_end.isoformat(),
        backtest_trades=len(in_period),
        mt5_trades=len(mt5_sorted),
        matched=matched,
        match_pct=round(100.0 * matched / denominator, 1),
        same_outcome_pct=round(100.0 * sum(same_outcome) / len(same_outcome), 1) if same_outcome else None,
        entry_slippage_points=round(float(np.median(slippage)), 1) if slippage else None,
        clock_shift_hours=shift_hours,
        first_divergence=first_divergence,
        only_backtest=[
            TradeRow(pd.Timestamp(bt_times[i]).isoformat(), int(bt_dirs[i]), float(in_period[i].entry_price),
                     round(float(in_period[i].r_multiple), 2))
            for i in only_bt[:MAX_LISTED]
        ],
        only_mt5=[
            TradeRow(mt5_sorted[j].entry_time.isoformat(), int(mt_dirs[j]), float(mt5_sorted[j].entry_price),
                     round(float(mt5_sorted[j].net_profit), 2))
            for j in only_mt[:MAX_LISTED]
        ],
    )
    report.verdict = _verdict(report)
    return report


def _verdict(report: TradeParityReport) -> str:
    if report.clock_shift_hours:
        return (
            f"The trades line up far better with MetaTrader's times moved {report.clock_shift_hours:+d}h. "
            "The broker's server clock is not the one in Settings -> Broker caps -> Server time: fix it there, "
            "re-check the strategy and export it again."
        )
    if report.match_pct >= 90:
        return (
            f"{report.matched} of {max(report.backtest_trades, report.mt5_trades)} trades line up. "
            "The EA is executing the backtested strategy; remaining differences are data and fills."
        )
    if report.mt5_trades < report.backtest_trades * 0.6:
        return (
            "MetaTrader took far fewer trades. Check the tester used the right symbol and timeframe, that "
            "MaxSpreadPoints is 0, and look at the journal for refused orders (stops level, filling mode)."
        )
    if report.mt5_trades > report.backtest_trades * 1.4:
        return (
            "MetaTrader took far more trades. Check the EA is not attached twice and that the tester's "
            "period matches; then compare the first divergence below."
        )
    return (
        "The trade counts are close but the trades themselves differ. Start from the first divergence: the "
        "bars around it are where MetaTrader's data or indicators disagree with the backtest's."
    )
