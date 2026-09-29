import pandas as pd

from app.data.conversion import quote_currency, quote_to_account_series
from app.data.symbols import INSTRUMENTS


def test_quote_currency_of_fx_metals_and_indices():
    assert quote_currency(INSTRUMENTS["EURUSD"]) == "USD"
    assert quote_currency(INSTRUMENTS["USDJPY"]) == "JPY"
    assert quote_currency(INSTRUMENTS["EURGBP"]) == "GBP"
    assert quote_currency(INSTRUMENTS["XAUUSD"]) == "USD"
    assert quote_currency(INSTRUMENTS["GER40"]) == "EUR"
    assert quote_currency(INSTRUMENTS["JP225"]) == "JPY"


def _bars(price):
    return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"), "close": [price] * 3})


def test_a_dollar_quoted_instrument_needs_no_conversion():
    assert quote_to_account_series(INSTRUMENTS["EURUSD"], lambda s, tf: _bars(1.1)) is None


def test_yen_is_converted_through_usdjpy_inverted():
    loaded = []
    rates = quote_to_account_series(INSTRUMENTS["EURJPY"], lambda s, tf: loaded.append(s) or _bars(150.0))
    assert loaded == ["USDJPY"]
    assert rates.iloc[0] == 1 / 150.0


def test_pounds_are_converted_through_gbpusd_directly():
    rates = quote_to_account_series(INSTRUMENTS["EURGBP"], lambda s, tf: _bars(1.25))
    assert rates.iloc[0] == 1.25


def test_no_cached_rate_falls_back_to_no_conversion():
    assert quote_to_account_series(INSTRUMENTS["EURGBP"], lambda s, tf: pd.DataFrame()) is None
