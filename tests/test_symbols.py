import pytest

from app.data.symbols import BUNDLED_SYMBOLS, get_instrument


def test_get_instrument_known_symbol():
    instrument = get_instrument("eurusd")
    assert instrument.symbol == "EURUSD"
    assert instrument.point_divisor == 100000


def test_get_instrument_unknown_symbol_raises():
    with pytest.raises(ValueError):
        get_instrument("NOTASYMBOL")


def test_bundled_symbols_include_xauusd_per_req_data_1_4():
    assert "XAUUSD" in BUNDLED_SYMBOLS
