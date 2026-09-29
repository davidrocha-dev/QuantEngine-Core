"""Vectorized technical-analysis primitives shared by the 46 genes.

Every function takes an OHLC(V) DataFrame (columns: open, high, low, close,
volume) and returns a pandas Series aligned to the same index. These are
standard, textbook definitions — see each docstring for the exact formula
and default parameters (REQ-ENGINE-1.2.5: "reasonable, standard,
well-documented implementation consistent with the gene's common
technical-analysis definition").
"""

from __future__ import annotations

import numba
import numpy as np
import pandas as pd

from app.data.broker_clock import broker_dates


@numba.njit(cache=True)
def _rolling_mean_abs_deviation(values: np.ndarray, period: int) -> np.ndarray:
    """Rolling mean absolute deviation, compiled.

    Pandas has no vectorised form of this - the window's mean has to be
    known before its deviations can be measured - so the natural spelling
    is `rolling(period).apply(lambda ...)`, which runs a Python function
    once per window. On a year of M15 bars that single lambda was the
    most expensive thing in the entire indicator layer.
    """
    n = values.size
    out = np.full(n, np.nan)
    for end in range(period - 1, n):
        start = end - period + 1
        total = 0.0
        for index in range(start, end + 1):
            total += values[index]
        mean = total / period
        deviation = 0.0
        for index in range(start, end + 1):
            deviation += abs(values[index] - mean)
        out[end] = deviation / period
    return out


def sma(series: pd.Series, period: int = 20) -> pd.Series:
    """Simple moving average over `period` bars."""
    return series.rolling(window=period, min_periods=period).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range as MetaTrader computes it: a simple moving
    average of the true range over `period` bars.

    Not Wilder's smoothing. Every exported EA sizes its stop, its target
    and its lot from MT5's built-in iATR, which is this SMA, and every
    ATR-based gene reads the same handle there. With Wilder here the two
    sides disagreed by -16% to +28% on the same bar (EURUSD H1, 5th-95th
    percentile), so the EA traded different stops, different sizes and -
    for RECEDING, which asks whether ATR fell five bars running - a
    different veto on more than half the bars that either side vetoed.
    """
    return true_range(df).rolling(window=period, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Standard 14-period relative strength index (Wilder's smoothing)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    # No losses in the window is the strongest reading RSI has, not a
    # neutral one: MT5 reports 100 there, and this used to report 50.
    result = result.mask(avg_loss.eq(0) & avg_gain.gt(0), 100.0)
    return result.fillna(50.0)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram, as MetaTrader's iMACD
    computes them: EMA(fast) - EMA(slow), and a signal line that is a
    *simple* moving average of the MACD line.

    Textbook MACD smooths the signal with an EMA; MT5's built-in does
    not, and the exported EA reads the built-in. With an EMA here only
    half of the crossovers the backtest traded happened in MetaTrader.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.rolling(window=signal, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index: (typical price - SMA) / (0.015 * mean deviation)."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_sma = sma(typical_price, period)
    mean_deviation = pd.Series(
        _rolling_mean_abs_deviation(typical_price.to_numpy(dtype=np.float64), period),
        index=typical_price.index,
    )
    return (typical_price - tp_sma) / (0.015 * mean_deviation.replace(0, np.nan))


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    """Stochastic oscillator %K / %D, standard 14/3 setup, with no
    slowing: iStochastic(k, d, slowing=1) in MetaTrader."""
    lowest_low = df["low"].rolling(k_period).min()
    highest_high = df["high"].rolling(k_period).max()
    percent_k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    percent_d = percent_k.rolling(d_period).mean()
    return percent_k, percent_d


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R, standard 14-period setup."""
    highest_high = df["high"].rolling(period).max()
    lowest_low = df["low"].rolling(period).min()
    return -100 * (highest_high - df["close"]) / (highest_high - lowest_low).replace(0, np.nan)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index, standard Wilder 14-period setup.

    The EA must read this through iADXWilder, not iADX: MetaTrader's
    plain iADX smooths with an EMA of 2/(n+1), which is not Wilder."""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df)
    tr_smooth = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / tr_smooth.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / tr_smooth.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def directional_indicators(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series]:
    """+DI / -DI (Wilder), the directional components behind ADX."""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_smooth = true_range(df).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / tr_smooth.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / tr_smooth.replace(0, np.nan)
    return plus_di, minus_di


def donchian(df: pd.DataFrame, period: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Donchian channel: rolling high/low/mid over `period` bars."""
    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()
    mid = (upper + lower) / 2
    return upper, lower, mid


def keltner(df: pd.DataFrame, period: int = 20, atr_mult: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channel: EMA basis +/- `atr_mult` * ATR."""
    basis = df["close"].ewm(span=period, adjust=False).mean()
    band = atr(df, period) * atr_mult
    return basis + band, basis - band, basis


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP, reset at the broker's midnight - the day
    boundary MetaTrader's own bars use."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical_price * df["volume"]
    day = broker_dates(df)
    cum_pv = pv.groupby(day).cumsum()
    cum_vol = df["volume"].groupby(day).cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


def psar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    """Parabolic SAR, standard acceleration-factor implementation."""
    high, low = df["high"].to_numpy(), df["low"].to_numpy()
    n = len(df)
    result = np.zeros(n)
    if n == 0:
        return pd.Series(result, index=df.index)

    bullish = True
    af = step
    ep = high[0]
    sar = low[0]

    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if bullish:
            sar = min(sar, low[i - 1], low[max(i - 2, 0)])
            if low[i] < sar:
                bullish = False
                sar = ep
                ep = low[i]
                af = step
            elif high[i] > ep:
                ep = high[i]
                af = min(af + step, max_step)
        else:
            sar = max(sar, high[i - 1], high[max(i - 2, 0)])
            if high[i] > sar:
                bullish = True
                sar = ep
                ep = high[i]
                af = step
            elif low[i] < ep:
                ep = low[i]
                af = min(af + step, max_step)
        result[i] = sar

    result[0] = sar
    return pd.Series(result, index=df.index)


def chandelier_exit(df: pd.DataFrame, period: int = 22, atr_mult: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """Chandelier Exit: highest-high/lowest-low over `period` bars +/- `atr_mult` * ATR."""
    highest_high = df["high"].rolling(period).max()
    lowest_low = df["low"].rolling(period).min()
    band = atr(df, period) * atr_mult
    return highest_high - band, lowest_low + band


def average_daily_range(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average range of the last `period` completed broker days - the D1
    bars MetaTrader would show - mapped onto every bar of the next day.

    Days are the broker's, not UTC's: a UTC day on a New York close feed
    includes a two-hour Sunday stub whose tiny range dragged the average
    down, and the EA reads real D1 bars that have no such day."""
    dates = broker_dates(df)
    daily = df.groupby(dates).agg(high=("high", "max"), low=("low", "min")).sort_index()
    # shift(1): a bar only knows *completed* days — today's final range is
    # not known until the day closes (using it would be look-ahead bias).
    daily_range = (daily["high"] - daily["low"]).rolling(period).mean().shift(1)
    return dates.map(daily_range).astype(float)


def is_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Bullish engulfing: current bullish candle's body fully engulfs the prior bearish candle's body."""
    prev_open, prev_close = df["open"].shift(1), df["close"].shift(1)
    bullish = df["close"] > df["open"]
    prev_bearish = prev_close < prev_open
    engulfs = (df["open"] <= prev_close) & (df["close"] >= prev_open)
    return bullish & prev_bearish & engulfs


def is_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Bearish engulfing: current bearish candle's body fully engulfs the prior bullish candle's body."""
    prev_open, prev_close = df["open"].shift(1), df["close"].shift(1)
    bearish = df["close"] < df["open"]
    prev_bullish = prev_close > prev_open
    engulfs = (df["open"] >= prev_close) & (df["close"] <= prev_open)
    return bearish & prev_bullish & engulfs


def is_doji(df: pd.DataFrame, body_ratio: float = 0.1) -> pd.Series:
    """Doji: candle body is <= `body_ratio` of the full high-low range."""
    body = (df["close"] - df["open"]).abs()
    full_range = (df["high"] - df["low"]).replace(0, np.nan)
    return (body / full_range) <= body_ratio


def three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """Three White Soldiers: 3 consecutive bullish candles, each closing higher than the last."""
    bullish = df["close"] > df["open"]
    rising = df["close"] > df["close"].shift(1)
    pattern = bullish & bullish.shift(1).fillna(False) & bullish.shift(2).fillna(False)
    pattern &= rising & rising.shift(1).fillna(False)
    return pattern.fillna(False)


def wick_rejection(df: pd.DataFrame, wick_ratio: float = 2.0) -> pd.Series:
    """Wick rejection: the longer wick is at least `wick_ratio`x the candle body.
    Returns +1 (bullish, long lower wick), -1 (bearish, long upper wick), 0 otherwise."""
    body = (df["close"] - df["open"]).abs().replace(0, np.nan)
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    bullish_reject = (lower_wick / body) >= wick_ratio
    bearish_reject = (upper_wick / body) >= wick_ratio
    return pd.Series(np.select([bullish_reject, bearish_reject], [1, -1], default=0), index=df.index)


def fibonacci_retracement_levels(swing_high: float, swing_low: float) -> dict[str, float]:
    """Standard Fibonacci retracement levels between a swing high and low."""
    diff = swing_high - swing_low
    return {
        "0.0": swing_high,
        "0.236": swing_high - 0.236 * diff,
        "0.382": swing_high - 0.382 * diff,
        "0.5": swing_high - 0.5 * diff,
        "0.618": swing_high - 0.618 * diff,
        "0.786": swing_high - 0.786 * diff,
        "1.0": swing_low,
    }
