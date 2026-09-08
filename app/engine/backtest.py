"""Genome backtester — evaluates one genome against one OHLC series.

Single-position-at-a-time, fixed-fractional risk sizing: each trade risks
`risk_pct` of current balance, sized off the distance to its stop-loss, so
a full stop-out always costs exactly `risk_pct` of balance at entry. The
baseline stop/target (DEFAULT_STOP_ATR_MULT / DEFAULT_TARGET_ATR_MULT) is
what makes a genome with no Management gene still backtestable — see
app/genome/genome.py for why Management is optional per-genome.

The innermost "scan forward until SL/TP/management exit" loop is Numba
JIT-compiled (`_scan_exit_numba`) since it is the dominant cost for a
4,000-candidate population — task 2.12 / design.md B.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numba
import numpy as np
import pandas as pd

from app.engine.news_filter import EconomicEvent, NewsMode, apply_news_filter
from app.genome.gene_pool import GenePoolConfig
from app.genome.genes import ALL_GENES
from app.genome.genome import Genome
from app.indicators import ta

DEFAULT_STOP_ATR_MULT = 1.5
DEFAULT_TARGET_ATR_MULT = 3.0
MAX_HOLD_BARS = 500  # safety cap so a trade can never hang open indefinitely
PENDING_ORDER_WINDOW_BARS = 5  # LIMIT/LIMIT2/STOP orders expire unfilled after this many bars

EXIT_STOP = 0
EXIT_TARGET = 1
EXIT_MAX_HOLD = 2
EXIT_MANAGEMENT = 3


@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    direction: int  # 1 long, -1 short
    entry_price: float
    exit_price: float
    stop_price: float
    r_multiple: float
    pnl: float
    exit_reason: str
    partial_closed: bool = False


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    final_balance: float = 0.0
    margin_breach: bool = False


@numba.njit(cache=True)
def _scan_exit_numba(
    highs: np.ndarray,
    lows: np.ndarray,
    entry_idx: int,
    direction: int,
    stop_price: float,
    target_price: float,
    max_hold: int,
) -> tuple[int, float, int]:
    """Scans forward from `entry_idx` (exclusive) until price touches the
    stop or target, or `max_hold` bars elapse. Conservative tie-break: if a
    single bar's range spans both levels, the stop is assumed hit first."""
    n = len(highs)
    last_idx = min(entry_idx + max_hold, n - 1)
    for i in range(entry_idx + 1, last_idx + 1):
        if direction == 1:
            if lows[i] <= stop_price:
                return i, stop_price, EXIT_STOP
            if highs[i] >= target_price:
                return i, target_price, EXIT_TARGET
        else:
            if highs[i] >= stop_price:
                return i, stop_price, EXIT_STOP
            if lows[i] <= target_price:
                return i, target_price, EXIT_TARGET
    return last_idx, np.nan, EXIT_MAX_HOLD


def _validate_genome_against_pool(genome: Genome, gene_pool: GenePoolConfig) -> None:
    """Defensive check: a stale genome (e.g. loaded from a checkpoint saved
    before a Gene Pool change) must not silently evaluate a now-disabled gene."""
    if not gene_pool.is_enabled("signal", genome.signal.name):
        raise ValueError(f"Genome uses disabled SIGNAL gene: {genome.signal.name}")
    if not gene_pool.is_enabled("execution", genome.execution.name):
        raise ValueError(f"Genome uses disabled EXECUTION gene: {genome.execution.name}")
    if genome.bias is not None and not gene_pool.is_enabled("bias", genome.bias.name):
        raise ValueError(f"Genome uses disabled BIAS gene: {genome.bias.name}")
    if genome.management is not None and not gene_pool.is_enabled("management", genome.management.name):
        raise ValueError(f"Genome uses disabled MANAGEMENT gene: {genome.management.name}")
    for f in genome.filters:
        if not gene_pool.is_enabled("filter", f.name):
            raise ValueError(f"Genome uses disabled FILTER gene: {f.name}")


def _combine_filter_veto(df: pd.DataFrame, genome: Genome, htf_df: pd.DataFrame | None) -> pd.Series:
    veto = pd.Series(False, index=df.index)
    for choice in genome.filters:
        spec = ALL_GENES.get(("filter", choice.name))
        if spec is None or spec.compute is None:
            continue
        veto = veto | spec.compute(df, htf_df, choice.params).fillna(False)
    return veto


def _apply_management_exit(
    df: pd.DataFrame, genome: Genome, entry_idx: int, exit_idx: int, direction: int
) -> tuple[int, str] | None:
    """Checks Management-gene early-exit conditions (EOD CLOSE, FRIDAY
    PROFIT) between entry and the baseline SL/TP exit. Returns the earlier
    (bar_index, reason) if one fires, else None. PARTIAL TP is handled
    separately in `simulate_trades` since it doesn't end the trade."""
    if genome.management is None:
        return None
    name, params = genome.management.name, genome.management.params
    if name not in ("EOD CLOSE", "FRIDAY PROFIT"):
        return None

    window = df.iloc[entry_idx + 1 : exit_idx + 1]
    if window.empty:
        return None
    timestamps = pd.DatetimeIndex(window["timestamp"])

    if name == "EOD CLOSE":
        hit = timestamps.hour >= params.get("close_hour", 21)
    else:  # FRIDAY PROFIT
        hit = (timestamps.dayofweek == 4) & (timestamps.hour >= params.get("lock_hour", 18))

    if hit.any():
        return int(window.index[np.argmax(hit)]), name
    return None


def simulate_trades(
    df: pd.DataFrame,
    genome: Genome,
    gene_pool: GenePoolConfig,
    htf_df: pd.DataFrame | None = None,
    news_events: list[EconomicEvent] | None = None,
    news_mode: NewsMode = "FTMO",
    initial_balance: float = 10_000.0,
    risk_pct: float = 0.01,
    spread_points: float = 0.0,
    max_leverage_breach_balance: float = 0.0,
) -> BacktestResult:
    """Runs one genome against one OHLC series end to end."""
    _validate_genome_against_pool(genome, gene_pool)

    signal_spec = ALL_GENES[("signal", genome.signal.name)]
    signal = signal_spec.compute(df, htf_df, genome.signal.params)

    if genome.bias is not None:
        bias_spec = ALL_GENES[("bias", genome.bias.name)]
        bias = bias_spec.compute(df, htf_df, genome.bias.params)
    else:
        bias = pd.Series(0, index=df.index)

    veto = _combine_filter_veto(df, genome, htf_df)
    if any(f.name == "EVENT" for f in genome.filters) and news_events:
        veto = veto | apply_news_filter(df, news_events, news_mode)

    atr_series = ta.atr(df).to_numpy()
    closes, highs, lows, opens = (df[c].to_numpy() for c in ("close", "high", "low", "open"))
    signal_arr, bias_arr, veto_arr = signal.to_numpy(), bias.to_numpy(), veto.to_numpy()

    n = len(df)
    balance = initial_balance
    equity_curve = np.empty(n)
    equity_curve[:] = balance
    trades: list[Trade] = []
    margin_breach = False

    i = 0
    while i < n - 1:
        direction = signal_arr[i]
        if direction == 0 or veto_arr[i] or np.isnan(atr_series[i]) or atr_series[i] <= 0:
            i += 1
            continue
        if bias_arr[i] != 0 and bias_arr[i] != direction:
            i += 1
            continue

        entry_idx, entry_price = _resolve_entry(df, genome, i, direction, closes, highs, lows, opens, atr_series)
        if entry_idx is None:
            i += 1
            continue

        atr_at_entry = atr_series[i]
        stop_price = entry_price - direction * DEFAULT_STOP_ATR_MULT * atr_at_entry
        target_price = entry_price + direction * DEFAULT_TARGET_ATR_MULT * atr_at_entry
        risk_amount = balance * risk_pct

        baseline_exit_idx, exit_price, exit_code = _scan_exit_numba(
            highs, lows, entry_idx, int(direction), stop_price, target_price, MAX_HOLD_BARS
        )
        if exit_code == EXIT_MAX_HOLD:
            exit_price = closes[baseline_exit_idx]

        exit_idx, exit_reason, partial_closed = baseline_exit_idx, _EXIT_NAMES[exit_code], False
        management_hit = _apply_management_exit(df, genome, entry_idx, baseline_exit_idx, int(direction))
        if management_hit is not None and management_hit[0] <= baseline_exit_idx:
            exit_idx, exit_reason = management_hit
            exit_price = closes[exit_idx]

        pnl_amount, r_multiple = _apply_partial_tp(
            genome, direction, entry_price, stop_price, exit_price, risk_amount
        )
        # `spread_points` is intentionally a no-op here — Phase 4's noise
        # injection (app/engine/monte_carlo.py) is where spread/slippage
        # perturbation is applied, per REQ-ENGINE-3.8. The baseline backtest
        # (Phases 0-3) runs on clean prices so fitness ranking isn't noisy.
        balance += pnl_amount
        equity_curve[exit_idx:] = balance
        if max_leverage_breach_balance > 0 and balance < max_leverage_breach_balance:
            margin_breach = True

        trades.append(
            Trade(entry_idx, exit_idx, int(direction), entry_price, exit_price, stop_price, r_multiple, pnl_amount, exit_reason, partial_closed)
        )
        i = exit_idx + 1

    return BacktestResult(trades=trades, equity_curve=equity_curve, final_balance=balance, margin_breach=margin_breach)


_EXIT_NAMES = {EXIT_STOP: "stop", EXIT_TARGET: "target", EXIT_MAX_HOLD: "max_hold"}


def _resolve_entry(df, genome, signal_idx, direction, closes, highs, lows, opens, atr_series):
    """Applies the genome's EXECUTION gene to resolve an actual fill index
    and price from the bar where the signal fired."""
    name, params = genome.execution.name, genome.execution.params
    signal_price = closes[signal_idx]
    atr_at_signal = atr_series[signal_idx]

    if name == "MARKET2":
        confirm_bars = params.get("confirm_bars", 2)
        target_i = signal_idx + confirm_bars
        if target_i >= len(closes):
            return None, None
        return target_i, opens[target_i]

    if name == "LIMIT":
        offset = params.get("offset_atr", 0.5) * atr_at_signal
        limit_price = signal_price - direction * offset
        return _wait_for_fill(highs, lows, signal_idx, direction, limit_price, is_limit=True)

    if name == "STOP":
        offset = params.get("offset_atr", 0.3) * atr_at_signal
        stop_price = signal_price + direction * offset
        return _wait_for_fill(highs, lows, signal_idx, direction, stop_price, is_limit=False)

    if name == "LIMIT2":
        offset_1 = params.get("offset_atr_1", 0.5) * atr_at_signal
        offset_2 = params.get("offset_atr_2", 1.0) * atr_at_signal
        price_1 = signal_price - direction * offset_1
        price_2 = signal_price - direction * offset_2
        avg_price = (price_1 + price_2) / 2
        return _wait_for_fill(highs, lows, signal_idx, direction, avg_price, is_limit=True)

    return signal_idx, signal_price  # unknown execution gene — immediate fill fallback


def _wait_for_fill(highs, lows, signal_idx, direction, order_price, is_limit: bool):
    last_i = min(signal_idx + PENDING_ORDER_WINDOW_BARS, len(highs) - 1)
    for i in range(signal_idx + 1, last_i + 1):
        touched = (lows[i] <= order_price <= highs[i])
        if touched:
            return i, order_price
    return None, None


def _apply_partial_tp(genome: Genome, direction: int, entry_price: float, stop_price: float, exit_price: float, risk_amount: float) -> tuple[float, float]:
    """PARTIAL TP splits the position: `close_fraction` exits at `r_multiple`
    R (if reached before the final exit), the remainder rides to the actual
    exit. Simplified to a weighted-average R multiple across both legs."""
    stop_distance = abs(entry_price - stop_price)
    full_r = (exit_price - entry_price) * direction / stop_distance if stop_distance > 0 else 0.0

    has_partial = genome.management is not None and genome.management.name == "PARTIAL TP"
    if not has_partial:
        return risk_amount * full_r, full_r

    r_target = genome.management.params.get("r_multiple", 1.0)
    fraction = genome.management.params.get("close_fraction", 0.5)
    partial_r = min(full_r, r_target) if full_r > 0 else full_r
    blended_r = fraction * partial_r + (1 - fraction) * full_r
    return risk_amount * blended_r, blended_r
