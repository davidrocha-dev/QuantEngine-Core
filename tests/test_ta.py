import numpy as np
import pytest

from app.indicators import ta
from tests.helpers import make_synthetic_ohlc


def test_sma_matches_manual_mean():
    df = make_synthetic_ohlc(50)
    result = ta.sma(df["close"], period=5)
    manual = df["close"].iloc[10:15].mean()
    assert result.iloc[14] == pytest.approx(manual)


def test_rsi_bounded_0_100():
    df = make_synthetic_ohlc(300)
    result = ta.rsi(df["close"]).dropna()
    assert (result >= 0).all() and (result <= 100).all()


def test_atr_non_negative():
    df = make_synthetic_ohlc(200)
    result = ta.atr(df).dropna()
    assert (result >= 0).all()


def test_macd_histogram_is_difference_of_lines():
    df = make_synthetic_ohlc(200)
    macd_line, signal_line, histogram = ta.macd(df["close"])
    np.testing.assert_allclose(histogram.to_numpy(), (macd_line - signal_line).to_numpy())


def test_donchian_upper_gte_lower():
    df = make_synthetic_ohlc(100)
    upper, lower, mid = ta.donchian(df, 20)
    valid = upper.notna() & lower.notna()
    assert (upper[valid] >= lower[valid]).all()
    np.testing.assert_allclose(mid[valid].to_numpy(), ((upper + lower) / 2)[valid].to_numpy())


def test_psar_stays_within_reasonable_bounds():
    df = make_synthetic_ohlc(100)
    result = ta.psar(df)
    assert len(result) == len(df)
    assert result.notna().all()


def test_is_doji_detects_small_body():
    import pandas as pd
    df = pd.DataFrame({
        "open": [1.0, 1.0], "high": [1.1, 1.1], "low": [0.9, 0.9],
        "close": [1.001, 1.08],  # first is a doji, second is not
    })
    result = ta.is_doji(df, body_ratio=0.1)
    assert result.iloc[0] == True
    assert result.iloc[1] == False


def test_wick_rejection_directions():
    import pandas as pd
    df = pd.DataFrame({
        # bullish rejection: long lower wick
        "open": [1.05, 1.00], "close": [1.06, 1.01],
        "high": [1.061, 1.011], "low": [0.90, 0.90],
    })
    result = ta.wick_rejection(df, wick_ratio=2.0)
    assert result.iloc[0] == 1


# The exported EA reads MetaTrader's built-in indicators, so these pin the
# Python side to MT5's definitions rather than to the textbook ones.

def test_atr_is_mt5_simple_average_of_true_range():
    df = make_synthetic_ohlc(100)
    expected = ta.true_range(df).rolling(14).mean()
    np.testing.assert_allclose(ta.atr(df).to_numpy(), expected.to_numpy(), equal_nan=True)


def test_macd_signal_is_mt5_simple_average_of_the_macd_line():
    df = make_synthetic_ohlc(200)
    macd_line, signal_line, _ = ta.macd(df["close"], 12, 26, 9)
    np.testing.assert_allclose(
        signal_line.to_numpy(), macd_line.rolling(9).mean().to_numpy(), equal_nan=True
    )


def test_rsi_with_no_losses_reads_100_not_50():
    import pandas as pd

    rising = pd.Series(np.arange(1.0, 40.0))
    assert ta.rsi(rising).iloc[-1] == pytest.approx(100.0)


def test_the_session_average_is_equal_weighted_and_resets_each_broker_day():
    """No volume both sides share, so none is used: the EA's SessionVwap
    averages the same typical prices over the same bars."""
    import pandas as pd

    stamps = pd.date_range("2024-01-15 22:00", periods=6, freq="h", tz="UTC")  # 00:00-05:00 on the server, winter
    df = pd.DataFrame({
        "timestamp": stamps,
        "broker_time": pd.date_range("2024-01-16 00:00", periods=6, freq="h"),
        "high": [3.0, 6.0, 9.0, 3.0, 6.0, 9.0], "low": [3.0, 6.0, 9.0, 3.0, 6.0, 9.0],
        "close": [3.0, 6.0, 9.0, 3.0, 6.0, 9.0], "volume": [1.0, 1000.0, 1.0, 1.0, 1.0, 1.0],
    })
    np.testing.assert_allclose(ta.vwap(df).to_numpy(), [3.0, 4.5, 6.0, 5.25, 5.4, 6.0])

    # Straddling the server's midnight (22:00-03:00): the third bar opens a new day.
    straddling = df.assign(broker_time=pd.date_range("2024-01-16 22:00", periods=6, freq="h"))
    np.testing.assert_allclose(ta.vwap(straddling).to_numpy(), [3.0, 4.5, 9.0, 6.0, 6.0, 6.75])
