"""What a price move is worth in the account's currency.

A move on USDJPY is measured in yen, on EURGBP in pounds, on GER40 in
euros. The backtester sizes a lot from the stop distance, so it has to
know what one unit of the quote currency is worth in the account's own
currency - otherwise a yen lot looks about 150 times bigger than it is,
and the broker's volume ladder then rounds the real position to the
minimum lot. MetaTrader does this conversion itself (SYMBOL_TRADE_TICK_VALUE),
which is why the exported EA traded a different size from the backtest.

The account currency is taken to be USD, which is what the risk settings
and the built-in instrument table assume.
"""

from __future__ import annotations

import logging
from typing import Callable

import pandas as pd

from app.data.symbols import INSTRUMENTS, Instrument

ACCOUNT_CURRENCY = "USD"
CONVERSION_TIMEFRAME = "H1"  # an hourly rate is plenty for sizing, and cheap

_CURRENCIES = {
    "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD", "SGD", "HKD",
    "NOK", "SEK", "PLN", "MXN", "ZAR", "TRY",
}

log = logging.getLogger("quantengine")


def quote_currency(instrument: Instrument) -> str:
    """The currency a price of this instrument is quoted in."""
    if instrument.category == "fx" and len(instrument.symbol) >= 6:
        return instrument.symbol[3:6]
    code = instrument.dukascopy_code.upper()
    # Dukascopy index and commodity codes end in their currency: DEUIDXEUR,
    # JPNIDXJPY, LIGHTCMDUSD.
    if code[-3:] in _CURRENCIES:
        return code[-3:]
    return ACCOUNT_CURRENCY


def quote_to_account_series(
    instrument: Instrument, load_bars: Callable[[str, str], pd.DataFrame]
) -> pd.Series | None:
    """Account currency per unit of quote currency, indexed by timestamp,
    or None when the quote currency is the account's own.

    `load_bars(symbol, timeframe)` reads cached history. When the pair
    that carries the rate has none, sizing falls back to no conversion and
    says so in the log - the old behaviour, and wrong by the exchange rate.
    """
    quote = quote_currency(instrument)
    if quote == ACCOUNT_CURRENCY:
        return None

    direct, inverse = f"{quote}{ACCOUNT_CURRENCY}", f"{ACCOUNT_CURRENCY}{quote}"
    for pair, invert in ((direct, False), (inverse, True)):
        if pair not in INSTRUMENTS:
            continue
        bars = load_bars(pair, CONVERSION_TIMEFRAME)
        if bars is None or bars.empty:
            continue
        closes = bars["close"].astype(float)
        rates = (1.0 / closes) if invert else closes
        return pd.Series(rates.to_numpy(), index=pd.DatetimeIndex(bars["timestamp"]))

    log.warning(
        "No cached %s or %s history to convert %s prices into %s: position sizing and commission for %s "
        "will be off by the exchange rate. Sync one of those pairs to fix it.",
        direct, inverse, quote, ACCOUNT_CURRENCY, instrument.symbol,
    )
    return None
