"""Instrument metadata for the Dukascopy Historical Data Feed — design.md B.4.

Dukascopy publishes tick prices as integers; ``point_divisor`` converts the
raw integer to a real price (price = raw / point_divisor). Values below are
the standard Dukascopy point divisors for each instrument class.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    symbol: str  # QuantEngine-facing symbol, e.g. "EURUSD"
    dukascopy_code: str  # path segment used by the Dukascopy feed
    point_divisor: int
    category: str  # "fx" | "metal" | "index"


# The pre-bundled subset (REQ-DATA-1.4) and default Gene Pool markets
# both draw from this table — "common pairs/XAUUSD" per REQ-DATA-1.4.
INSTRUMENTS: dict[str, Instrument] = {
    "EURUSD": Instrument("EURUSD", "EURUSD", 100000, "fx"),
    "GBPUSD": Instrument("GBPUSD", "GBPUSD", 100000, "fx"),
    "USDJPY": Instrument("USDJPY", "USDJPY", 1000, "fx"),
    "USDCHF": Instrument("USDCHF", "USDCHF", 100000, "fx"),
    "AUDUSD": Instrument("AUDUSD", "AUDUSD", 100000, "fx"),
    "USDCAD": Instrument("USDCAD", "USDCAD", 100000, "fx"),
    "NZDUSD": Instrument("NZDUSD", "NZDUSD", 100000, "fx"),
    "EURGBP": Instrument("EURGBP", "EURGBP", 100000, "fx"),
    "EURJPY": Instrument("EURJPY", "EURJPY", 1000, "fx"),
    "GBPJPY": Instrument("GBPJPY", "GBPJPY", 1000, "fx"),
    "XAUUSD": Instrument("XAUUSD", "XAUUSD", 1000, "metal"),
    "XAGUSD": Instrument("XAGUSD", "XAGUSD", 1000, "metal"),
}

# Bundled with the installer for immediate first-run availability — REQ-DATA-1.4.
BUNDLED_SYMBOLS: tuple[str, ...] = ("EURUSD", "GBPUSD", "USDJPY", "XAUUSD")

# Earliest date the Dukascopy feed reliably covers — design.md B.4.
DUKASCOPY_EPOCH_YEAR = 2003
DUKASCOPY_EPOCH_MONTH = 5

TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H2", "H4", "D1")


def get_instrument(symbol: str) -> Instrument:
    try:
        return INSTRUMENTS[symbol.upper()]
    except KeyError as exc:
        raise ValueError(f"Unknown instrument: {symbol}") from exc
