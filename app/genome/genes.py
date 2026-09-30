"""The 46-gene catalog — REQ-ENGINE-1.2 (authoritative names, do not rename).

Layer roles:
  SIGNAL      -> per-bar entry trigger, returns -1 (short) / 0 (none) / 1 (long)
  BIAS        -> per-bar directional tilt, returns -1 / 0 / 1; a SIGNAL only
                 fires when BIAS agrees with it (0 = no opinion, always agrees)
  FILTER      -> per-bar veto, returns True when the bar should block entry
  EXECUTION   -> order-placement spec consumed by the backtester at entry time,
                 not a per-bar series (see app/engine/backtest.py)
  MANAGEMENT  -> position-management spec consumed by the backtester during
                 the life of a trade, not a per-bar series

Every function's docstring states the exact formula and default parameters,
per REQ-ENGINE-1.2.5, so it can be reviewed/adjusted rather than treated as
fixed. `HTF`-aware genes accept an optional `htf_df` (None when HIGHER TF is
disabled — REQ-BREED-2 — in which case they degenerate to neutral/no-op).
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import pandas as pd

from app.data.broker_clock import bar_clock, broker_dates
from app.indicators import ta

Layer = Literal["signal", "bias", "filter", "execution", "management"]


@dataclass(frozen=True)
class GeneSpec:
    name: str
    layer: Layer
    default_params: dict = field(default_factory=dict)
    compute: Callable | None = None  # None for execution/management genes
    # Search space for this gene's parameters — REQ-ENGINE-1.5 leaves the
    # exact values to the implementer, and a genetic engine that only ever
    # evaluates the defaults is not searching at all: it can only recombine
    # ~180k fixed gene tuples. Each entry is ("int"|"float", low, high) or
    # ("choice", [options]); `default_params` stays the documented centre of
    # each range, so a genome built from defaults still behaves as described.
    param_space: dict[str, tuple] = field(default_factory=dict)


def sample_params(spec: GeneSpec, rng: random.Random | None = None) -> dict:
    """Draws one random parameter set from the gene's search space."""
    rng = rng or random
    params = copy.deepcopy(spec.default_params)
    for key, space in spec.param_space.items():
        kind = space[0]
        if kind == "int":
            params[key] = rng.randint(space[1], space[2])
        elif kind == "float":
            params[key] = round(rng.uniform(space[1], space[2]), 4)
        elif kind == "choice":
            params[key] = rng.choice(space[1])
    return params


def jitter_params(spec: GeneSpec, params: dict, rate: float, rng: random.Random | None = None) -> dict:
    """Perturbs each parameter with probability `rate`, staying inside its
    range: local search around a working genome, as opposed to `sample_params`
    which restarts from scratch."""
    rng = rng or random
    mutated = copy.deepcopy(params)
    for key, space in spec.param_space.items():
        if rng.random() >= rate:
            continue
        kind = space[0]
        if kind == "choice":
            mutated[key] = rng.choice(space[1])
            continue
        low, high = space[1], space[2]
        current = mutated.get(key, low)
        step = (high - low) * 0.25
        value = min(max(current + rng.uniform(-step, step), low), high)
        mutated[key] = int(round(value)) if kind == "int" else round(value, 4)
    return mutated


# ---------------------------------------------------------------------------
# SIGNAL layer (16) — entry triggers
# ---------------------------------------------------------------------------

def _signal_atr_run(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """ATR RUN — enters with a strong directional run: fires long when price
    has moved more than `atr_mult` ATRs above its value `lookback` bars ago,
    short on the symmetric downward move. Default: lookback=10, atr_mult=1.5."""
    lookback, atr_mult = params.get("lookback", 10), params.get("atr_mult", 1.5)
    move = df["close"] - df["close"].shift(lookback)
    threshold = ta.atr(df) * atr_mult
    return pd.Series(np.select([move > threshold, move < -threshold], [1, -1], default=0), index=df.index)


def _signal_breakout(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """BREAKOUT — price clears the N-period prior high/low. Default N=20."""
    period = params.get("period", 20)
    prior_high = df["high"].rolling(period).max().shift(1)
    prior_low = df["low"].rolling(period).min().shift(1)
    return pd.Series(
        np.select([df["close"] > prior_high, df["close"] < prior_low], [1, -1], default=0), index=df.index
    )


def _signal_cci(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """CCI — classic ±100 threshold cross: long on cross above +100, short
    on cross below -100. Default period=20."""
    values = ta.cci(df, params.get("period", 20))
    crossed_up = (values > 100) & (values.shift(1) <= 100)
    crossed_down = (values < -100) & (values.shift(1) >= -100)
    return pd.Series(np.select([crossed_up, crossed_down], [1, -1], default=0), index=df.index)


def _signal_engulfing(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """ENGULFING — bullish engulfing candle fires long, bearish engulfing fires short."""
    bull = ta.is_bullish_engulfing(df)
    bear = ta.is_bearish_engulfing(df)
    return pd.Series(np.select([bull, bear], [1, -1], default=0), index=df.index)


def _signal_fade(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """FADE — contrarian fade of RSI extremes: long when RSI < `oversold`
    (fading a dip), short when RSI > `overbought`. Defaults: 30 / 70."""
    values = ta.rsi(df["close"], params.get("period", 14))
    oversold, overbought = params.get("oversold", 30), params.get("overbought", 70)
    return pd.Series(np.select([values < oversold, values > overbought], [1, -1], default=0), index=df.index)


def _signal_fib(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """FIB — price retraces into the 0.618 Fibonacci level of the last
    `lookback`-bar swing and closes back away from it, entering in the
    swing's direction. Default lookback=50, tolerance=0.1 (fraction of swing)."""
    lookback, tolerance = params.get("lookback", 50), params.get("tolerance", 0.1)
    swing_high = df["high"].rolling(lookback).max()
    swing_low = df["low"].rolling(lookback).min()
    swing_range = (swing_high - swing_low).replace(0, np.nan)
    level_618 = swing_high - 0.618 * swing_range
    near_level = (df["close"] - level_618).abs() <= tolerance * swing_range
    uptrend = df["close"] > df["close"].shift(lookback)
    bounced_up = near_level & uptrend & (df["close"] > df["close"].shift(1))
    bounced_down = near_level & ~uptrend & (df["close"] < df["close"].shift(1))
    return pd.Series(np.select([bounced_up, bounced_down], [1, -1], default=0), index=df.index)


def _signal_inside_breakout(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """INSIDE BREAKOUT — an inside bar (high/low fully within the prior
    bar's range) followed by a close beyond that inside bar's range."""
    inside = (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))
    inside_prev = inside.shift(1).fillna(False)
    breakout_up = inside_prev & (df["close"] > df["high"].shift(1))
    breakout_down = inside_prev & (df["close"] < df["low"].shift(1))
    return pd.Series(np.select([breakout_up, breakout_down], [1, -1], default=0), index=df.index)


def _signal_liqsweep(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """LIQSWEEP — liquidity sweep reversal: the bar wicks beyond the prior
    `period`-bar high/low but closes back inside it (a stop-hunt), entering
    opposite the wick direction. Default period=20."""
    period = params.get("period", 20)
    prior_high = df["high"].rolling(period).max().shift(1)
    prior_low = df["low"].rolling(period).min().shift(1)
    swept_high = (df["high"] > prior_high) & (df["close"] < prior_high)
    swept_low = (df["low"] < prior_low) & (df["close"] > prior_low)
    return pd.Series(np.select([swept_low, swept_high], [1, -1], default=0), index=df.index)


def _signal_macd(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """MACD — standard 12/26/9 line/signal crossover."""
    macd_line, signal_line, _ = ta.macd(
        df["close"], params.get("fast", 12), params.get("slow", 26), params.get("signal", 9)
    )
    crossed_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    crossed_down = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
    return pd.Series(np.select([crossed_up, crossed_down], [1, -1], default=0), index=df.index)


def _signal_mom_break(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """MOM BREAK — rate-of-change over `period` bars crosses `threshold`
    (fraction). Defaults: period=10, threshold=0.003 (0.3%)."""
    period, threshold = params.get("period", 10), params.get("threshold", 0.003)
    roc = df["close"] / df["close"].shift(period) - 1
    return pd.Series(np.select([roc > threshold, roc < -threshold], [1, -1], default=0), index=df.index)


def _signal_orb(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """ORB — Opening Range Breakout: the opening range is the first
    `range_bars` bars of each broker day; a close beyond that range's high/low
    later in the day fires the breakout direction. Default range_bars=4."""
    range_bars = params.get("range_bars", 4)
    day_series = broker_dates(df)
    bar_in_day = day_series.groupby(day_series).cumcount()

    or_high = df["high"].where(bar_in_day < range_bars).groupby(day_series).transform("max")
    or_low = df["low"].where(bar_in_day < range_bars).groupby(day_series).transform("min")

    after_range = bar_in_day >= range_bars
    breakout_up = after_range & (df["close"] > or_high)
    breakout_down = after_range & (df["close"] < or_low)
    return pd.Series(np.select([breakout_up, breakout_down], [1, -1], default=0), index=df.index)


def _signal_stoch(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """STOCH — %K crosses above %D while oversold (<20) fires long; %K
    crosses below %D while overbought (>80) fires short. Default 14/3."""
    percent_k, percent_d = ta.stochastic(df, params.get("k_period", 14), params.get("d_period", 3))
    crossed_up = (percent_k > percent_d) & (percent_k.shift(1) <= percent_d.shift(1)) & (percent_k < 20)
    crossed_down = (percent_k < percent_d) & (percent_k.shift(1) >= percent_d.shift(1)) & (percent_k > 80)
    return pd.Series(np.select([crossed_up, crossed_down], [1, -1], default=0), index=df.index)


def _signal_three_soldiers(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """THREE SOLDIERS — three white soldiers fires long; the bearish
    counterpart (three black crows) fires short."""
    white = ta.three_white_soldiers(df)
    bearish = df["close"] < df["open"]
    falling = df["close"] < df["close"].shift(1)
    black = bearish & bearish.shift(1).fillna(False) & bearish.shift(2).fillna(False)
    black &= falling & falling.shift(1).fillna(False)
    return pd.Series(np.select([white, black.fillna(False)], [1, -1], default=0), index=df.index)


def _signal_vwap(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """VWAP — close crosses above the session VWAP fires long; crosses below fires short."""
    values = ta.vwap(df)
    crossed_up = (df["close"] > values) & (df["close"].shift(1) <= values.shift(1))
    crossed_down = (df["close"] < values) & (df["close"].shift(1) >= values.shift(1))
    return pd.Series(np.select([crossed_up, crossed_down], [1, -1], default=0), index=df.index)


def _signal_wick_rejection(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """WICK REJECTION — a wick at least `wick_ratio`x the candle body fires
    in the rejection direction. Default wick_ratio=2.0."""
    return ta.wick_rejection(df, params.get("wick_ratio", 2.0))


def _signal_williams(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """WILLIAMS — Williams %R crossing back above -80 (out of oversold)
    fires long; crossing back below -20 (out of overbought) fires short."""
    values = ta.williams_r(df, params.get("period", 14))
    crossed_up = (values > -80) & (values.shift(1) <= -80)
    crossed_down = (values < -20) & (values.shift(1) >= -20)
    return pd.Series(np.select([crossed_up, crossed_down], [1, -1], default=0), index=df.index)


SIGNAL_GENES: dict[str, GeneSpec] = {
    "ATR RUN": GeneSpec("ATR RUN", "signal", {"lookback": 10, "atr_mult": 1.5}, _signal_atr_run, {"lookback": ("int", 5, 30), "atr_mult": ("float", 0.5, 3.0)}),
    "BREAKOUT": GeneSpec("BREAKOUT", "signal", {"period": 20}, _signal_breakout, {"period": ("int", 10, 60)}),
    "CCI": GeneSpec("CCI", "signal", {"period": 20}, _signal_cci, {"period": ("int", 10, 40)}),
    "ENGULFING": GeneSpec("ENGULFING", "signal", {}, _signal_engulfing),
    "FADE": GeneSpec("FADE", "signal", {"period": 14, "oversold": 30, "overbought": 70}, _signal_fade, {"period": ("int", 7, 28), "oversold": ("int", 15, 35), "overbought": ("int", 65, 85)}),
    "FIB": GeneSpec("FIB", "signal", {"lookback": 50, "tolerance": 0.1}, _signal_fib, {"lookback": ("int", 20, 100), "tolerance": ("float", 0.05, 0.25)}),
    "INSIDE BREAKOUT": GeneSpec("INSIDE BREAKOUT", "signal", {}, _signal_inside_breakout),
    "LIQSWEEP": GeneSpec("LIQSWEEP", "signal", {"period": 20}, _signal_liqsweep, {"period": ("int", 10, 40)}),
    "MACD": GeneSpec("MACD", "signal", {"fast": 12, "slow": 26, "signal": 9}, _signal_macd, {"fast": ("int", 6, 18), "slow": ("int", 20, 40), "signal": ("int", 5, 14)}),
    "MOM BREAK": GeneSpec("MOM BREAK", "signal", {"period": 10, "threshold": 0.003}, _signal_mom_break, {"period": ("int", 5, 30), "threshold": ("float", 0.0005, 0.01)}),
    "ORB": GeneSpec("ORB", "signal", {"range_bars": 4}, _signal_orb, {"range_bars": ("int", 2, 12)}),
    "STOCH": GeneSpec("STOCH", "signal", {"k_period": 14, "d_period": 3}, _signal_stoch, {"k_period": ("int", 7, 28), "d_period": ("int", 2, 7)}),
    "THREE SOLDIERS": GeneSpec("THREE SOLDIERS", "signal", {}, _signal_three_soldiers),
    "VWAP": GeneSpec("VWAP", "signal", {}, _signal_vwap),
    "WICK REJECTION": GeneSpec("WICK REJECTION", "signal", {"wick_ratio": 2.0}, _signal_wick_rejection, {"wick_ratio": ("float", 1.2, 4.0)}),
    "WILLIAMS": GeneSpec("WILLIAMS", "signal", {"period": 14}, _signal_williams, {"period": ("int", 7, 28)}),
}

# ---------------------------------------------------------------------------
# BIAS layer (11) — directional tilt; a SIGNAL only fires when BIAS agrees
# ---------------------------------------------------------------------------

def _bias_adx(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """ADX — trend direction gated by trend strength: bias follows +DI vs
    -DI only while ADX exceeds `threshold` (default 20); otherwise neutral."""
    plus_di, minus_di = ta.directional_indicators(df, params.get("period", 14))
    strong = ta.adx(df, params.get("period", 14)) > params.get("threshold", 20)
    direction = np.select([plus_di > minus_di, plus_di < minus_di], [1, -1], default=0)
    return pd.Series(np.where(strong, direction, 0), index=df.index)


def _bias_chandelier(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """CHANDELIER — bullish while price holds above the long Chandelier
    Exit level, bearish below the short level. Default period=22, mult=3.0."""
    long_stop, short_stop = ta.chandelier_exit(df, params.get("period", 22), params.get("atr_mult", 3.0))
    return pd.Series(np.select([df["close"] > long_stop, df["close"] < short_stop], [1, -1], default=0), index=df.index)


def _bias_daily_mid(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """DAILY MID — bullish above the previous broker day's (high+low)/2
    midpoint, bearish below: the EA's D1 bar 1."""
    # Grouped by the days that have bars, so Monday's "yesterday" is Friday.
    dates = broker_dates(df)
    daily = df.groupby(dates).agg(high=("high", "max"), low=("low", "min")).sort_index()
    daily_mid = ((daily["high"] + daily["low"]) / 2).shift(1)
    mid_series = dates.map(daily_mid).astype(float)
    return pd.Series(np.select([df["close"] > mid_series, df["close"] < mid_series], [1, -1], default=0), index=df.index)


def _bias_donchian_mid(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """DONCHIAN MID — bullish above the N-period Donchian midline, bearish below. Default N=20."""
    _, _, mid = ta.donchian(df, params.get("period", 20))
    return pd.Series(np.select([df["close"] > mid, df["close"] < mid], [1, -1], default=0), index=df.index)


def _bar_duration(frame: pd.DataFrame) -> pd.Timedelta:
    """A bar's length, read off the data: the commonest gap between opens
    (weekend and holiday gaps are rarer than ordinary ones)."""
    gaps = pd.DatetimeIndex(frame["timestamp"]).to_series().diff().dropna()
    return gaps.mode().iloc[0] if not gaps.empty else pd.Timedelta(0)


def _bias_htf(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """HTF — bullish/bearish per the higher-timeframe close vs. its SMA
    (default period=20), forward-filled onto the trading timeframe. Neutral
    (all zeros) when HIGHER TF is disabled — REQ-BREED-2.2."""
    if htf_df is None or htf_df.empty:
        return pd.Series(0, index=df.index)
    htf_sma = ta.sma(htf_df["close"], params.get("period", 20))
    htf_bias = pd.Series(np.select([htf_df["close"] > htf_sma, htf_df["close"] < htf_sma], [1, -1], default=0), index=htf_df.index)
    # Each trading bar is decided at its close, and at that moment the HTF
    # bar that has just closed is known - which is what the EA reads
    # (HtfClose(1) at the open of the next bar). Aligning on open times
    # with a one-bar lag instead left the last trading bar of every HTF
    # period reading the HTF bar before: 7.6% of M15 bars on EURUSD with
    # an H1 context disagreed with the EA.
    htf_closes = pd.DatetimeIndex(htf_df["timestamp"]) + _bar_duration(htf_df)
    trading_closes = pd.DatetimeIndex(df["timestamp"]) + _bar_duration(df)
    aligned = pd.Series(htf_bias.to_numpy(), index=htf_closes).reindex(trading_closes, method="ffill")
    return aligned.set_axis(df.index).fillna(0)


def _bias_market_structure(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """MARKET STRUCTURE — bullish on higher-highs & higher-lows vs.
    `lookback` bars ago, bearish on lower-highs & lower-lows. Default lookback=10."""
    lookback = params.get("lookback", 10)
    hh = df["high"] > df["high"].shift(lookback)
    hl = df["low"] > df["low"].shift(lookback)
    lh = df["high"] < df["high"].shift(lookback)
    ll = df["low"] < df["low"].shift(lookback)
    return pd.Series(np.select([hh & hl, lh & ll], [1, -1], default=0), index=df.index)


def _bias_momentum(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """MOMENTUM — sign of the `period`-bar rate of change. Default period=20."""
    roc = df["close"] / df["close"].shift(params.get("period", 20)) - 1
    return pd.Series(np.select([roc > 0, roc < 0], [1, -1], default=0), index=df.index)


def _bias_psar(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """PSAR — bullish while price is above the Parabolic SAR dot, bearish below."""
    sar = ta.psar(df, params.get("step", 0.02), params.get("max_step", 0.2))
    return pd.Series(np.select([df["close"] > sar, df["close"] < sar], [1, -1], default=0), index=df.index)


def _bias_rsi(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """RSI — bullish above the 50 midline, bearish below. Default period=14."""
    values = ta.rsi(df["close"], params.get("period", 14))
    return pd.Series(np.select([values > 50, values < 50], [1, -1], default=0), index=df.index)


def _bias_sma(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """SMA — classic trend filter: bullish above the SMA, bearish below. Default period=50."""
    values = ta.sma(df["close"], params.get("period", 50))
    return pd.Series(np.select([df["close"] > values, df["close"] < values], [1, -1], default=0), index=df.index)


def _bias_trailing(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """TRAILING — bullish while price holds above a trailing `period`-bar
    low, bearish while below a trailing `period`-bar high. Default period=14."""
    trailing_low = df["low"].rolling(params.get("period", 14)).min().shift(1)
    trailing_high = df["high"].rolling(params.get("period", 14)).max().shift(1)
    return pd.Series(np.select([df["close"] > trailing_high, df["close"] < trailing_low], [1, -1], default=0), index=df.index)


BIAS_GENES: dict[str, GeneSpec] = {
    "ADX": GeneSpec("ADX", "bias", {"period": 14, "threshold": 20}, _bias_adx, {"period": ("int", 7, 28), "threshold": ("int", 12, 35)}),
    "CHANDELIER": GeneSpec("CHANDELIER", "bias", {"period": 22, "atr_mult": 3.0}, _bias_chandelier, {"period": ("int", 10, 40), "atr_mult": ("float", 1.5, 4.5)}),
    "DAILY MID": GeneSpec("DAILY MID", "bias", {}, _bias_daily_mid),
    "DONCHIAN MID": GeneSpec("DONCHIAN MID", "bias", {"period": 20}, _bias_donchian_mid, {"period": ("int", 10, 50)}),
    "HTF": GeneSpec("HTF", "bias", {"period": 20}, _bias_htf, {"period": ("int", 10, 50)}),
    "MARKET STRUCTURE": GeneSpec("MARKET STRUCTURE", "bias", {"lookback": 10}, _bias_market_structure, {"lookback": ("int", 4, 25)}),
    "MOMENTUM": GeneSpec("MOMENTUM", "bias", {"period": 20}, _bias_momentum, {"period": ("int", 8, 60)}),
    "PSAR": GeneSpec("PSAR", "bias", {"step": 0.02, "max_step": 0.2}, _bias_psar, {"step": ("float", 0.008, 0.05), "max_step": ("float", 0.1, 0.4)}),
    "RSI": GeneSpec("RSI", "bias", {"period": 14}, _bias_rsi, {"period": ("int", 7, 28)}),
    "SMA": GeneSpec("SMA", "bias", {"period": 50}, _bias_sma, {"period": ("int", 15, 200)}),
    "TRAILING": GeneSpec("TRAILING", "bias", {"period": 14}, _bias_trailing, {"period": ("int", 6, 40)}),
}

# ---------------------------------------------------------------------------
# FILTER layer (12) — veto conditions (True blocks entry)
# ---------------------------------------------------------------------------

def _filter_adr_exhaust(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """ADR EXHAUST — blocks entry once today's range already exceeds
    `exhaust_pct` of the average daily range (avoid chasing an exhausted
    move). Default exhaust_pct=0.8 (80%)."""
    exhaust_pct = params.get("exhaust_pct", 0.8)
    adr = ta.average_daily_range(df, params.get("period", 14))
    day_series = broker_dates(df)
    today_high = df["high"].groupby(day_series).cummax()
    today_low = df["low"].groupby(day_series).cummin()
    today_range = today_high - today_low
    return (today_range >= exhaust_pct * adr).fillna(False)


def _filter_cci(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """CCI — blocks entry when |CCI| exceeds an extreme threshold (avoid
    whipsaw at extremes). Default period=20, extreme=200."""
    values = ta.cci(df, params.get("period", 20))
    return (values.abs() > params.get("extreme", 200)).fillna(False)


def _filter_consec(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """CONSEC — cool-down filter: blocks entry after `max_losses`
    consecutive losing trades. Evaluated by the backtester against live
    trade history (not a pure price-series filter) — see
    app/engine/backtest.py `consecutive_losses_veto`. Default max_losses=3."""
    return pd.Series(False, index=df.index)  # placeholder series; real veto applied in backtester


def _filter_doji(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """DOJI — blocks entry on an indecision (doji) bar. Default body_ratio=0.1."""
    return ta.is_doji(df, params.get("body_ratio", 0.1)).fillna(False)


def _filter_event(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """EVENT — the News Filter gene (REQ-ENGINE-2): blocks entry within the
    configured window of a High/Medium-impact economic event. Delegated to
    app/engine/news_filter.py — see `apply_news_filter`."""
    return pd.Series(False, index=df.index)  # replaced by news_filter.apply_news_filter when EVENT is enabled


def _filter_keltner(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """KELTNER — blocks entry when price is outside the Keltner Channel
    (extreme volatility spike). Default period=20, atr_mult=2.0."""
    upper, lower, _ = ta.keltner(df, params.get("period", 20), params.get("atr_mult", 2.0))
    return ((df["close"] > upper) | (df["close"] < lower)).fillna(False)


def _filter_no_fri_open(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """NO FRI OPEN — blocks new entries on Fridays after `cutoff_hour`
    broker time (avoid weekend gap risk). Default cutoff_hour=18."""
    timestamps = bar_clock(df)
    is_friday = timestamps.dayofweek == 4
    after_cutoff = timestamps.hour >= params.get("cutoff_hour", 18)
    return pd.Series(is_friday & after_cutoff, index=df.index)


def _filter_receding(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """RECEDING — blocks entry while volatility (ATR) has been declining
    for `lookback` consecutive bars (avoid low-momentum chop). Default lookback=5."""
    lookback = params.get("lookback", 5)
    atr_series = ta.atr(df)
    declining = atr_series.diff() < 0
    return declining.rolling(lookback).apply(lambda x: bool(x.all()), raw=True).fillna(0).astype(bool)


def _filter_regime(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """REGIME — blocks entry when the market is not in the required regime:
    `require` = "trend" blocks when ADX < threshold, "range" blocks when
    ADX >= threshold. Default require="trend", threshold=20."""
    strength = ta.adx(df, params.get("period", 14))
    threshold = params.get("threshold", 20)
    if params.get("require", "trend") == "trend":
        return (strength < threshold).fillna(True)
    return (strength >= threshold).fillna(False)


def _filter_rsi(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """RSI — blocks entry when RSI is at an extreme (>80 or <20), avoiding
    entries into an already-exhausted move. Default period=14."""
    values = ta.rsi(df["close"], params.get("period", 14))
    return ((values > 80) | (values < 20)).fillna(False)


def _filter_sma(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """SMA — blocks entry when price is more than `atr_mult` ATRs away from
    its SMA (avoid chasing an overextended move). Default period=50, atr_mult=3.0."""
    values = ta.sma(df["close"], params.get("period", 50))
    distance = (df["close"] - values).abs()
    return (distance > ta.atr(df) * params.get("atr_mult", 3.0)).fillna(False)


def _filter_volatility(df: pd.DataFrame, htf_df: pd.DataFrame | None, params: dict) -> pd.Series:
    """VOLATILITY — blocks entry when ATR falls outside an acceptable
    [min_atr, max_atr] band (too quiet or too wild), expressed as a
    fraction of the `period`-bar average ATR. Defaults: min=0.5x, max=2.0x."""
    values = ta.atr(df, params.get("period", 14))
    baseline = values.rolling(params.get("period", 14) * 5).mean()
    too_low = values < baseline * params.get("min_mult", 0.5)
    too_high = values > baseline * params.get("max_mult", 2.0)
    return (too_low | too_high).fillna(False)


FILTER_GENES: dict[str, GeneSpec] = {
    "ADR EXHAUST": GeneSpec("ADR EXHAUST", "filter", {"period": 14, "exhaust_pct": 0.8}, _filter_adr_exhaust, {"period": ("int", 7, 28), "exhaust_pct": ("float", 0.5, 1.3)}),
    "CCI": GeneSpec("CCI", "filter", {"period": 20, "extreme": 200}, _filter_cci, {"period": ("int", 10, 40), "extreme": ("float", 120, 320)}),
    "CONSEC": GeneSpec("CONSEC", "filter", {"max_losses": 3}, _filter_consec, {"max_losses": ("int", 2, 6)}),
    "DOJI": GeneSpec("DOJI", "filter", {"body_ratio": 0.1}, _filter_doji, {"body_ratio": ("float", 0.04, 0.25)}),
    "EVENT": GeneSpec("EVENT", "filter", {"mode": "FTMO"}, _filter_event),
    "KELTNER": GeneSpec("KELTNER", "filter", {"period": 20, "atr_mult": 2.0}, _filter_keltner, {"period": ("int", 10, 40), "atr_mult": ("float", 1.0, 3.5)}),
    "NO FRI OPEN": GeneSpec("NO FRI OPEN", "filter", {"cutoff_hour": 18}, _filter_no_fri_open, {"cutoff_hour": ("int", 12, 22)}),
    "RECEDING": GeneSpec("RECEDING", "filter", {"lookback": 5}, _filter_receding, {"lookback": ("int", 3, 12)}),
    "REGIME": GeneSpec("REGIME", "filter", {"period": 14, "threshold": 20, "require": "trend"}, _filter_regime, {"period": ("int", 7, 28), "threshold": ("int", 12, 35), "require": ("choice", ["trend", "range"])}),
    "RSI": GeneSpec("RSI", "filter", {"period": 14}, _filter_rsi, {"period": ("int", 7, 28)}),
    "SMA": GeneSpec("SMA", "filter", {"period": 50, "atr_mult": 3.0}, _filter_sma, {"period": ("int", 15, 200), "atr_mult": ("float", 1.5, 5.0)}),
    "VOLATILITY": GeneSpec("VOLATILITY", "filter", {"period": 14, "min_mult": 0.5, "max_mult": 2.0}, _filter_volatility, {"period": ("int", 7, 28), "min_mult": ("float", 0.2, 0.8), "max_mult": ("float", 1.4, 3.0)}),
}

# ---------------------------------------------------------------------------
# EXECUTION layer (4) — order-placement specs, applied by the backtester
# ---------------------------------------------------------------------------

EXECUTION_GENES: dict[str, GeneSpec] = {
    # LIMIT — enter on a pullback: place a limit order `offset_atr` ATRs
    # behind the signal price, filled only if price retraces to it.
    "LIMIT": GeneSpec("LIMIT", "execution", {"offset_atr": 0.5}, None, {"offset_atr": ("float", 0.1, 1.2)}),
    # LIMIT2 — one limit order at the average of two ATR pullbacks
    # (offset_atr_1, offset_atr_2). Named for a two-tier scaled entry, but
    # both the backtester and the EA place a single order at the midpoint;
    # `split` is carried for stored genomes and not read.
    "LIMIT2": GeneSpec("LIMIT2", "execution", {"offset_atr_1": 0.5, "offset_atr_2": 1.0, "split": 0.5}, None, {"offset_atr_1": ("float", 0.1, 1.0), "offset_atr_2": ("float", 0.6, 2.0)}),
    # MARKET2 — market entry at the open `confirm_bars` bars after the
    # signal bar, whatever the signal does meanwhile (the backtester and the
    # EA both latch it; nothing is re-tested while it waits).
    "MARKET2": GeneSpec("MARKET2", "execution", {"confirm_bars": 2}, None, {"confirm_bars": ("int", 1, 5)}),
    # STOP — breakout-style stop entry `offset_atr` ATRs beyond the signal price.
    "STOP": GeneSpec("STOP", "execution", {"offset_atr": 0.3}, None, {"offset_atr": ("float", 0.05, 1.0)}),
}

# ---------------------------------------------------------------------------
# MANAGEMENT layer (3) — position-management specs, applied by the backtester
# ---------------------------------------------------------------------------

MANAGEMENT_GENES: dict[str, GeneSpec] = {
    # EOD CLOSE — force-close any open position at `close_hour`, broker time.
    "EOD CLOSE": GeneSpec("EOD CLOSE", "management", {"close_hour": 21}, None, {"close_hour": ("int", 15, 23)}),
    # FRIDAY PROFIT — if a position is in profit after `lock_hour` (broker time) on
    # Friday, close it to avoid weekend gap risk.
    "FRIDAY PROFIT": GeneSpec("FRIDAY PROFIT", "management", {"lock_hour": 18}, None, {"lock_hour": ("int", 12, 21)}),
    # PARTIAL TP — close `close_fraction` of the position at `r_multiple` R
    # and move the stop to breakeven for the remainder.
    "PARTIAL TP": GeneSpec("PARTIAL TP", "management", {"r_multiple": 1.0, "close_fraction": 0.5}, None, {"r_multiple": ("float", 0.5, 2.5), "close_fraction": ("float", 0.2, 0.8)}),
}

# Keyed by (layer, name): CCI/RSI/SMA each appear in more than one layer
# with layer-specific logic (e.g. SIGNAL CCI triggers, FILTER CCI vetoes) —
# a bare-name key would silently collide and drop genes across layers.
ALL_GENES: dict[tuple[Layer, str], GeneSpec] = {
    (spec.layer, spec.name): spec
    for layer_dict in (SIGNAL_GENES, BIAS_GENES, FILTER_GENES, EXECUTION_GENES, MANAGEMENT_GENES)
    for spec in layer_dict.values()
}

LAYER_GENES: dict[Layer, dict[str, GeneSpec]] = {
    "signal": SIGNAL_GENES,
    "bias": BIAS_GENES,
    "filter": FILTER_GENES,
    "execution": EXECUTION_GENES,
    "management": MANAGEMENT_GENES,
}

assert len(SIGNAL_GENES) == 16, "REQ-ENGINE-1.2 requires exactly 16 SIGNAL genes"
assert len(BIAS_GENES) == 11, "REQ-ENGINE-1.2 requires exactly 11 BIAS genes"
assert len(FILTER_GENES) == 12, "REQ-ENGINE-1.2 requires exactly 12 FILTER genes"
assert len(EXECUTION_GENES) == 4, "REQ-ENGINE-1.2 requires exactly 4 EXECUTION genes"
assert len(MANAGEMENT_GENES) == 3, "REQ-ENGINE-1.2 requires exactly 3 MANAGEMENT genes"
assert len(ALL_GENES) == 46, "REQ-ENGINE-1.2 requires exactly 46 genes total"
