"""Genome backtester — evaluates one genome against one OHLC series.

Single-position-at-a-time, fixed-fractional risk sizing: each trade risks
`risk_pct` of current balance, sized off the distance to its stop-loss, so
a full stop-out costs `risk_pct` of balance at entry — exactly, unless a
broker volume ladder is given, and then only as exactly as that ladder
allows. The
baseline stop/target (DEFAULT_STOP_ATR_MULT / DEFAULT_TARGET_ATR_MULT) is
what makes a genome with no Management gene still backtestable — see
app/genome/genome.py for why Management is optional per-genome.

The innermost "scan forward until SL/TP/management exit" loop is Numba
JIT-compiled (`_scan_exit_numba`) since it is the dominant cost for a
4,000-candidate population — task 2.12 / design.md B.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import math

import numba
import numpy as np
import pandas as pd

from app.data.broker_clock import bar_clock
from app.engine.swap import PRICE as SWAP_PRICE
from app.engine.swap import SwapSpec, nights_held
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
    # Kept so a perturbation (Phase 4 noise injection) can replay the trade
    # under its own rules instead of reusing an exit price that no longer
    # corresponds to the perturbed entry.
    target_price: float
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
    opens: np.ndarray,
    entry_idx: int,
    direction: int,
    stop_price: float,
    target_price: float,
    max_hold: int,
) -> tuple[int, float, int]:
    """Scans forward from `entry_idx` (exclusive) until price touches the
    stop or target, or `max_hold` bars elapse. Conservative tie-break: if a
    single bar's range spans both levels, the stop is assumed hit first.

    A bar that *opens* beyond the stop fills the stop at that open, not at
    the stop price: a stop is a market order once triggered, and after a
    gap the first price available is the open. Filling it at the stop
    handed every weekend gap and every news jump to the strategy for
    free. A target gapped through stays at the target price, which is the
    conservative side of the same question."""
    n = len(highs)
    last_idx = min(entry_idx + max_hold, n - 1)
    for i in range(entry_idx + 1, last_idx + 1):
        if direction == 1:
            if lows[i] <= stop_price:
                return i, min(stop_price, opens[i]), EXIT_STOP
            if highs[i] >= target_price:
                return i, target_price, EXIT_TARGET
        else:
            if highs[i] >= stop_price:
                return i, max(stop_price, opens[i]), EXIT_STOP
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
    genome: Genome,
    hours: np.ndarray,
    weekdays: np.ndarray,
    closes: np.ndarray,
    entry_idx: int,
    exit_idx: int,
    direction: int,
    entry_price: float,
) -> tuple[int, str] | None:
    """Checks Management-gene early-exit conditions (EOD CLOSE, FRIDAY
    PROFIT) between entry and the baseline SL/TP exit. Returns the earlier
    (bar_index, reason) if one fires, else None. PARTIAL TP is handled
    separately in `simulate_trades` since it doesn't end the trade.

    Takes the hour and weekday of every bar as plain arrays rather than
    the DataFrame itself. This runs once per trade, and slicing a frame
    and rebuilding a DatetimeIndex each time was two thirds of the entire
    backtest - the same derivation, thousands of times, for data that
    never changes within a run.
    """
    if genome.management is None:
        return None
    name, params = genome.management.name, genome.management.params
    if name not in ("EOD CLOSE", "FRIDAY PROFIT"):
        return None

    start, stop = entry_idx + 1, exit_idx + 1
    if start >= stop:
        return None

    if name == "EOD CLOSE":
        hit = hours[start:stop] >= params.get("close_hour", 21)
    else:  # FRIDAY PROFIT — only locks in a position that is actually in profit
        in_profit = ((closes[start:stop] - entry_price) * direction) > 0
        hit = (weekdays[start:stop] == 4) & (hours[start:stop] >= params.get("lock_hour", 18)) & in_profit

    if hit.any():
        # Positional, so a caller that hands over a frame whose index does
        # not start at zero cannot silently get a bar index from another
        # part of history.
        return start + int(np.argmax(hit)), name
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
    point_size: float = 0.0001,
    cost_points: float = 0.0,
    commission_per_lot: float = 0.0,
    contract_size: float = 100_000.0,
    quote_to_account: pd.Series | None = None,
    swap: SwapSpec | None = None,
    entry_delay_bars: int = 0,
    max_leverage_breach_balance: float = 0.0,
    lot_step: float = 0.0,
    min_lot: float = 0.0,
    max_lot: float = 0.0,
) -> BacktestResult:
    """Runs one genome against one OHLC series end to end.

    Trading costs (Settings → Costs) are charged on every trade, in every
    phase — not only in Phase 4's noise injection. `cost_points` is the
    round-trip spread + slippage in instrument points; `commission_per_lot`
    is the round-turn commission, charged against the position size that
    fixed-fractional sizing implies (risk / (stop distance x contract size)).
    Without them the GA optimises towards high-frequency strategies whose
    edge is smaller than their cost.

    `lot_step`, `min_lot` and `max_lot` are the broker's volume ladder.
    Left at zero the sizing is continuous, which is what this did and
    which is a fiction on a small account: the requested size is rounded
    down to the step and then floored at the minimum, so a trade whose
    stop is wide enough risks *more* than the setting allows and there is
    nothing the strategy can do about it. Exported to MetaTrader and run
    on XAUUSD from a 10,000 account at 0.35%, every one of 344 trades was
    the 0.01 minimum: the median trade risked 27.20 of the 35.00 asked
    for, and the worst risked 68.85, twice the setting. On a prop-firm
    account that is the difference between a rule and a hope.

    `quote_to_account` is what one unit of the instrument's quote currency
    is worth in the account's currency, indexed by timestamp: 1/price for
    USDJPY on a dollar account, the GBPUSD rate for EURGBP, None when the
    quote currency is the account's own. A price move is measured in the
    quote currency, and sizing without converting it made a USDJPY lot
    look about 150 times bigger than it is - the broker ladder then
    rounded a real 0.5 lot to the 0.01 minimum and commission all but
    vanished.

    `swap` is the symbol's overnight financing (app/engine/swap.py),
    charged per rollover the position is held through, on the broker's
    clock, and folded into both the P&L and the trade's R so every
    downstream measure - Monte Carlo included - sees it."""
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
    # Only the Management genes need these, but deriving them here costs
    # one pass over the series instead of one per trade.
    if genome.management is not None and genome.management.name in ("EOD CLOSE", "FRIDAY PROFIT"):
        # The broker's clock: the hours the EA reads off its own bars.
        _timestamps = bar_clock(df)
        bar_hours = _timestamps.hour.to_numpy()
        bar_weekdays = _timestamps.dayofweek.to_numpy()
    else:
        bar_hours = bar_weekdays = np.empty(0, dtype=np.int64)
    signal_arr, bias_arr, veto_arr = signal.to_numpy(), bias.to_numpy(), veto.to_numpy()
    conversion = _conversion_per_bar(df, quote_to_account)
    bar_days = bar_clock(df).to_numpy().astype("datetime64[D]") if swap is not None else None
    pending_entry = genome.execution.name in PENDING_EXECUTION_GENES

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

        entry_idx, entry_price = _resolve_entry(
            df, genome, i, direction, closes, highs, lows, opens, atr_series, fill_margin=cost_points * point_size
        )
        if entry_idx is None:
            # A resting order that expired unfilled held the slot for its
            # whole window, and the EA does not look at another signal
            # while it waits: it evaluates the next one on the bar the
            # order expired after. Moving on one bar here let the
            # backtest place an order on every bar of the window that
            # the EA was blocked for.
            if pending_entry:
                i = max(i + 1, min(i + PENDING_ORDER_WINDOW_BARS, n - 1))
            else:
                i += 1
            continue

        # Phase 4's entry-displacement perturbation (REQ-ENGINE-3.8): the
        # fill lands `entry_delay_bars` later, at that bar's open, and the
        # trade then plays out under its own rules from there. Perturbing
        # the input and re-running is the only way to displace an entry
        # honestly — patching an already-simulated trade cannot know what
        # the stop and target would have done from the new entry.
        if entry_delay_bars:
            entry_idx = min(entry_idx + entry_delay_bars, n - 1)
            entry_price = opens[entry_idx]

        atr_at_entry = atr_series[i]

        # Costs move where the position sits, not only what it earns.
        #
        # Bars here are mid prices (data/dukascopy_client). A long is
        # filled at the ask and its stop and target are both triggered on
        # the bid, so in the bars' own space the stop ends up a full
        # spread nearer and the target a full spread further away. The
        # position is paid for once, at entry, and the barriers follow
        # from what was paid.
        #
        # This used to charge the cost as a flat deduction from R after
        # the fact, which leaves both barriers at their nominal distance
        # and therefore never changes which one is reached. It makes a
        # trade slightly less profitable; it cannot make it lose. For a
        # strategy anywhere near its break-even win rate that is the
        # entire result - at a 0.38 reward:risk the break-even is 72.4%,
        # and this reported 76.7% where the broker delivered 72.8%.
        effective_entry = entry_price + direction * cost_points * point_size

        # Risk geometry is part of the genome (app/genome/genome.py), not a
        # constant: the GA searches stop/target distance like any other gene
        # parameter. The module constants remain the documented default.
        stop_price = effective_entry - direction * genome.stop_atr_mult * atr_at_entry
        target_price = effective_entry + direction * genome.target_atr_mult * atr_at_entry
        risk_amount = balance * risk_pct

        # Every execution gene fills intrabar (LIMIT/LIMIT2/STOP) or at the
        # bar's open (MARKET2), so the rest of the fill bar is live exposure:
        # a stop inside the fill bar's range counts as hit (conservative).
        entry_bar_stopped = entry_idx != i and (
            (direction == 1 and lows[entry_idx] <= stop_price) or (direction == -1 and highs[entry_idx] >= stop_price)
        )
        if entry_bar_stopped:
            baseline_exit_idx, exit_price, exit_code = entry_idx, stop_price, EXIT_STOP
        else:
            baseline_exit_idx, exit_price, exit_code = _scan_exit_numba(
                highs, lows, opens, entry_idx, int(direction), stop_price, target_price, MAX_HOLD_BARS
            )
        if exit_code == EXIT_MAX_HOLD:
            exit_price = closes[baseline_exit_idx]

        partial_r = None
        if not entry_bar_stopped and genome.management is not None and genome.management.name == "PARTIAL TP":
            partial_r, baseline_exit_idx, exit_price, exit_code = _run_partial_tp(
                genome, highs, lows, opens, closes, entry_idx, int(direction), effective_entry, stop_price, target_price,
                baseline_exit_idx, exit_price, exit_code,
            )

        exit_idx, exit_reason, partial_closed = baseline_exit_idx, _EXIT_NAMES[exit_code], partial_r is not None
        management_hit = _apply_management_exit(
            genome, bar_hours, bar_weekdays, closes, entry_idx, baseline_exit_idx, int(direction), effective_entry
        )
        if management_hit is not None and management_hit[0] <= baseline_exit_idx:
            exit_idx, exit_reason = management_hit
            exit_price = closes[exit_idx]

        # Spread and slippage are already in `effective_entry`, and so
        # in every distance measured from it. Commission is not a price:
        # it is cash, charged against the position that fixed-fractional
        # sizing implies. Phase 4's noise injection perturbs the entry on
        # top of this baseline (REQ-ENGINE-3.8).
        stop_distance = abs(effective_entry - stop_price)
        # Account currency lost per lot on a full stop-out.
        loss_per_lot = stop_distance * contract_size * conversion[entry_idx]
        lots = risk_amount / loss_per_lot if loss_per_lot > 0 else 0.0
        traded_risk = risk_amount
        if lot_step > 0.0 and lots > 0.0:
            # Down to the step, never up: a broker fills 0.01 or 0.02, not
            # the 0.017 the risk model asked for.
            lots = math.floor(lots / lot_step + 1e-9) * lot_step
            lots = max(lots, min_lot)
            if max_lot > 0.0:
                lots = min(lots, max_lot)
            traded_risk = lots * loss_per_lot

        gross_pnl, r_multiple = _blend_partial_tp(
            genome, direction, effective_entry, stop_price, exit_price, traded_risk, partial_r
        )
        commission = commission_per_lot * lots
        pnl_amount = gross_pnl - commission
        if swap is not None and exit_idx > entry_idx:
            nights = nights_held(bar_days[entry_idx], bar_days[exit_idx], swap.triple_day)
            if nights:
                per_night = swap.per_lot_night(int(direction))
                if swap.kind == SWAP_PRICE:
                    per_night *= contract_size * conversion[entry_idx]
                financing = lots * nights * per_night
                pnl_amount += financing
                if traded_risk > 0:
                    r_multiple += financing / traded_risk

        # Mark-to-market between entry and exit, so Max DD reflects the
        # drawdown actually lived through, not only closed-trade equity.
        if exit_idx > entry_idx and stop_distance > 0:
            unrealized_r = (closes[entry_idx:exit_idx] - effective_entry) * direction / stop_distance
            equity_curve[entry_idx:exit_idx] = balance + traded_risk * unrealized_r
            if max_leverage_breach_balance > 0 and float(equity_curve[entry_idx:exit_idx].min()) < max_leverage_breach_balance:
                margin_breach = True

        balance += pnl_amount
        equity_curve[exit_idx:] = balance
        if max_leverage_breach_balance > 0 and balance < max_leverage_breach_balance:
            margin_breach = True

        trades.append(
            Trade(
                entry_idx, exit_idx, int(direction), effective_entry, exit_price, stop_price, target_price,
                r_multiple, pnl_amount, exit_reason, partial_closed,
            )
        )
        # The EA looks for its next entry on the bar after the one that
        # closed the position, and that bar's signal is the exit bar's -
        # a stop hit intrabar leaves the rest of the bar flat, and the
        # signal at its close is as good as any other. Resuming one bar
        # later skipped exactly those entries, which MetaTrader then took.
        i = max(exit_idx, i + 1)

    return BacktestResult(trades=trades, equity_curve=equity_curve, final_balance=balance, margin_breach=margin_breach)


_EXIT_NAMES = {EXIT_STOP: "stop", EXIT_TARGET: "target", EXIT_MAX_HOLD: "max_hold"}

# Execution genes that rest an order in the book and wait for a fill.
PENDING_EXECUTION_GENES = ("LIMIT", "LIMIT2", "STOP")


def _conversion_per_bar(df: pd.DataFrame, quote_to_account: pd.Series | None) -> np.ndarray:
    """Quote-to-account rate for every bar of `df`, last known value
    carried forward (and the first one carried back to the start)."""
    if quote_to_account is None or len(quote_to_account) == 0 or "timestamp" not in df:
        return np.ones(len(df))
    rates = quote_to_account.sort_index()
    rates = rates[~rates.index.duplicated(keep="last")]
    aligned = rates.reindex(pd.DatetimeIndex(df["timestamp"]), method="ffill")
    aligned = aligned.bfill().fillna(1.0)
    return aligned.to_numpy(dtype=float)


def _resolve_entry(df, genome, signal_idx, direction, closes, highs, lows, opens, atr_series, fill_margin=0.0):
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
        return _wait_for_fill(highs, lows, signal_idx, direction, limit_price, is_limit=True, margin=fill_margin, opens=opens)

    if name == "STOP":
        offset = params.get("offset_atr", 0.3) * atr_at_signal
        stop_price = signal_price + direction * offset
        return _wait_for_fill(highs, lows, signal_idx, direction, stop_price, is_limit=False, margin=fill_margin, opens=opens)

    if name == "LIMIT2":
        offset_1 = params.get("offset_atr_1", 0.5) * atr_at_signal
        offset_2 = params.get("offset_atr_2", 1.0) * atr_at_signal
        price_1 = signal_price - direction * offset_1
        price_2 = signal_price - direction * offset_2
        avg_price = (price_1 + price_2) / 2
        return _wait_for_fill(highs, lows, signal_idx, direction, avg_price, is_limit=True, margin=fill_margin, opens=opens)

    return signal_idx, signal_price  # unknown execution gene — immediate fill fallback


def _wait_for_fill(highs, lows, signal_idx, direction, order_price, is_limit: bool, margin: float = 0.0, opens=None):
    """When a resting order gets filled, if it does.

    The order fills once the market trades *through* its price, not when a
    bar's range happens to reach it. The difference is the whole edge of
    every limit-entry strategy this engine has ever bred.

    Filling a buy limit the moment `lows[i] <= order_price` means the
    backtester buys at the exact low of the bar that touched it — the one
    price in that bar nobody actually gets — and then measures the trade
    from there. A bar's close is above its low on average, so any exit
    that happens soon after is favourable for free. Measured on a real
    strategy whose management gene closes positions the same session:
    41.5% of its trades exited that way at a mean of +0.35R, and swapping
    only the execution gene, leaving signal, bias, filters and management
    untouched, moved the campaign return from +27,381% on LIMIT2 to -9.6%
    on MARKET2. Every one of the 26 strategies in the Vault at the time
    used LIMIT or LIMIT2: the search had found this and converged on it.

    `margin` is the distance the price has to go beyond the order before
    the fill counts — the spread, since a buy limit rests on the ask and
    the quote has to cross it, and since being merely touched does not
    put you at the front of a queue.

    A bar that opens already through the order fills at that open, which
    is what happens to a resting order after a gap - and what the EA does
    when the price has passed the order before it can be placed. For a
    stop entry that is worse than the order price; filling it at the
    order price gave breakouts every gap for free.
    """
    last_i = min(signal_idx + PENDING_ORDER_WINDOW_BARS, len(highs) - 1)
    for i in range(signal_idx + 1, last_i + 1):
        if opens is not None:
            if is_limit:
                gapped = opens[i] <= order_price - margin if direction == 1 else opens[i] >= order_price + margin
            else:
                gapped = opens[i] >= order_price + margin if direction == 1 else opens[i] <= order_price - margin
            if gapped:
                return i, opens[i]
        if is_limit:
            # A buy limit rests below: price has to come down through it.
            through = lows[i] <= order_price - margin if direction == 1 else highs[i] >= order_price + margin
        else:
            # A buy stop rests above: price has to rise up through it.
            through = highs[i] >= order_price + margin if direction == 1 else lows[i] <= order_price - margin
        if through:
            return i, order_price
    return None, None


def _run_partial_tp(
    genome, highs, lows, opens, closes, entry_idx, direction, entry_price, stop_price, target_price,
    exit_idx, exit_price, exit_code,
):
    """PARTIAL TP: if price reaches `r_multiple` R before the stop, that
    leg is banked at `r_multiple` R and the remainder's stop moves to
    breakeven, riding on to the original target. Returns the banked partial
    R (None if never reached) and the remainder's (exit_idx, price, code)."""
    r_target = genome.management.params.get("r_multiple", 1.0)
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return None, exit_idx, exit_price, exit_code

    # The take-profit is a real order sitting in the book: a position
    # cannot run through it to reach a partial level beyond it. This used
    # to scan from entry with `partial_price` standing in for the target,
    # which let a trade whose target sits at 0.39R keep going to 2.0R,
    # bank most of the position there, and only then close the remainder.
    # Every such winner was reported at several times the R the geometry
    # allows - and because the search reads those numbers, it bred towards
    # exactly this shape: a near target with a far partial level. The GA
    # was selecting for the bug.
    target_r = (target_price - entry_price) * direction / stop_distance
    if target_r <= r_target:
        return None, exit_idx, exit_price, exit_code

    partial_price = entry_price + direction * r_target * stop_distance
    partial_idx, _, partial_code = _scan_exit_numba(
        highs, lows, opens, entry_idx, direction, stop_price, partial_price, MAX_HOLD_BARS
    )
    if partial_code != EXIT_TARGET:
        return None, exit_idx, exit_price, exit_code

    remaining_hold = MAX_HOLD_BARS - (partial_idx - entry_idx)
    exit_idx, exit_price, exit_code = _scan_exit_numba(
        highs, lows, opens, partial_idx, direction, entry_price, target_price, remaining_hold
    )
    if exit_code == EXIT_MAX_HOLD:
        exit_price = closes[exit_idx]
    return r_target, exit_idx, exit_price, exit_code


def _blend_partial_tp(
    genome: Genome, direction: int, entry_price: float, stop_price: float, exit_price: float,
    risk_amount: float, partial_r: float | None,
) -> tuple[float, float]:
    """Weighted-average R across both PARTIAL TP legs: `close_fraction` at
    the banked `partial_r`, the remainder at the final exit."""
    stop_distance = abs(entry_price - stop_price)
    full_r = (exit_price - entry_price) * direction / stop_distance if stop_distance > 0 else 0.0
    if partial_r is None:
        return risk_amount * full_r, full_r

    fraction = genome.management.params.get("close_fraction", 0.5)
    blended_r = fraction * partial_r + (1 - fraction) * full_r
    return risk_amount * blended_r, blended_r
