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
# both draw from this table.
#
# Every code and divisor below was verified against the live feed rather
# than taken from documentation: one hour of real ticks was fetched for
# each and the resulting price checked against what that instrument
# should cost. A wrong divisor does not fail loudly - it produces a
# backtest at the wrong order of magnitude, and every number downstream
# of it is quietly nonsense.
#
# That check earned its keep. Platinum, palladium and natural gas answer
# 404 under the obvious codes and are not here. Copper answers, but at a
# price that reconciles with neither dollars per pound nor per tonne, so
# it is not here either - an instrument nobody can price is worse than a
# missing one.
INSTRUMENTS: dict[str, Instrument] = {
    # --- FX majors -------------------------------------------------------
    "EURUSD": Instrument("EURUSD", "EURUSD", 100000, "fx"),
    "GBPUSD": Instrument("GBPUSD", "GBPUSD", 100000, "fx"),
    "USDJPY": Instrument("USDJPY", "USDJPY", 1000, "fx"),
    "USDCHF": Instrument("USDCHF", "USDCHF", 100000, "fx"),
    "AUDUSD": Instrument("AUDUSD", "AUDUSD", 100000, "fx"),
    "USDCAD": Instrument("USDCAD", "USDCAD", 100000, "fx"),
    "NZDUSD": Instrument("NZDUSD", "NZDUSD", 100000, "fx"),
    # --- FX crosses ------------------------------------------------------
    "EURGBP": Instrument("EURGBP", "EURGBP", 100000, "fx"),
    "EURJPY": Instrument("EURJPY", "EURJPY", 1000, "fx"),
    "GBPJPY": Instrument("GBPJPY", "GBPJPY", 1000, "fx"),
    "EURCHF": Instrument("EURCHF", "EURCHF", 100000, "fx"),
    "EURAUD": Instrument("EURAUD", "EURAUD", 100000, "fx"),
    "EURCAD": Instrument("EURCAD", "EURCAD", 100000, "fx"),
    "EURNZD": Instrument("EURNZD", "EURNZD", 100000, "fx"),
    "GBPAUD": Instrument("GBPAUD", "GBPAUD", 100000, "fx"),
    "GBPCAD": Instrument("GBPCAD", "GBPCAD", 100000, "fx"),
    "GBPCHF": Instrument("GBPCHF", "GBPCHF", 100000, "fx"),
    "GBPNZD": Instrument("GBPNZD", "GBPNZD", 100000, "fx"),
    "AUDCAD": Instrument("AUDCAD", "AUDCAD", 100000, "fx"),
    "AUDCHF": Instrument("AUDCHF", "AUDCHF", 100000, "fx"),
    "AUDJPY": Instrument("AUDJPY", "AUDJPY", 1000, "fx"),
    "AUDNZD": Instrument("AUDNZD", "AUDNZD", 100000, "fx"),
    "CADCHF": Instrument("CADCHF", "CADCHF", 100000, "fx"),
    "CADJPY": Instrument("CADJPY", "CADJPY", 1000, "fx"),
    "CHFJPY": Instrument("CHFJPY", "CHFJPY", 1000, "fx"),
    "NZDCAD": Instrument("NZDCAD", "NZDCAD", 100000, "fx"),
    "NZDCHF": Instrument("NZDCHF", "NZDCHF", 100000, "fx"),
    "NZDJPY": Instrument("NZDJPY", "NZDJPY", 1000, "fx"),
    # --- FX, thinner books. Real data, fewer ticks per hour than a major,
    # --- so treat a campaign on one with more suspicion, not less.
    "USDSGD": Instrument("USDSGD", "USDSGD", 100000, "fx"),
    "USDHKD": Instrument("USDHKD", "USDHKD", 100000, "fx"),
    "USDNOK": Instrument("USDNOK", "USDNOK", 100000, "fx"),
    "USDSEK": Instrument("USDSEK", "USDSEK", 100000, "fx"),
    "USDPLN": Instrument("USDPLN", "USDPLN", 100000, "fx"),
    "USDMXN": Instrument("USDMXN", "USDMXN", 100000, "fx"),
    "USDZAR": Instrument("USDZAR", "USDZAR", 100000, "fx"),
    "USDTRY": Instrument("USDTRY", "USDTRY", 100000, "fx"),
    "EURNOK": Instrument("EURNOK", "EURNOK", 100000, "fx"),
    "EURSEK": Instrument("EURSEK", "EURSEK", 100000, "fx"),
    "EURTRY": Instrument("EURTRY", "EURTRY", 100000, "fx"),
    # --- metals ----------------------------------------------------------
    "XAUUSD": Instrument("XAUUSD", "XAUUSD", 1000, "metal"),
    "XAGUSD": Instrument("XAGUSD", "XAGUSD", 1000, "metal"),
    # --- indices ---------------------------------------------------------
    "GER40": Instrument("GER40", "DEUIDXEUR", 1000, "index"),
    "US500": Instrument("US500", "USA500IDXUSD", 1000, "index"),
    "NAS100": Instrument("NAS100", "USATECHIDXUSD", 1000, "index"),
    "US30": Instrument("US30", "USA30IDXUSD", 1000, "index"),
    "UK100": Instrument("UK100", "GBRIDXGBP", 1000, "index"),
    "FRA40": Instrument("FRA40", "FRAIDXEUR", 1000, "index"),
    "JP225": Instrument("JP225", "JPNIDXJPY", 1000, "index"),
    "EU50": Instrument("EU50", "EUSIDXEUR", 1000, "index"),
    "AUS200": Instrument("AUS200", "AUSIDXAUD", 1000, "index"),
    "ESP35": Instrument("ESP35", "ESPIDXEUR", 1000, "index"),
    "SUI20": Instrument("SUI20", "CHEIDXCHF", 1000, "index"),
    "HK50": Instrument("HK50", "HKGIDXHKD", 1000, "index"),
    # --- energy ----------------------------------------------------------
    "WTI": Instrument("WTI", "LIGHTCMDUSD", 1000, "commodity"),
    "BRENT": Instrument("BRENT", "BRENTCMDUSD", 1000, "commodity"),
    # --- crypto. Weekend trading and gap risk that FX does not have; the
    # --- tick verification on a strategy here is worth more than usual.
    "BTCUSD": Instrument("BTCUSD", "BTCUSD", 10, "crypto"),
    "ETHUSD": Instrument("ETHUSD", "ETHUSD", 10, "crypto"),
}

# Bundled with the installer for immediate first-run availability — REQ-DATA-1.4.
BUNDLED_SYMBOLS: tuple[str, ...] = ("EURUSD", "GBPUSD", "USDJPY", "XAUUSD")

# Earliest date the Dukascopy feed reliably covers — design.md B.4.
DUKASCOPY_EPOCH_YEAR = 2003
DUKASCOPY_EPOCH_MONTH = 5

TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H2", "H4", "D1")


def get_instrument(symbol: str) -> Instrument:
    """Built-in first, then anything the user added themselves.

    The import is local because custom_instruments imports the feed
    client, which imports this module - and a circular import at startup
    would be a silly way to lose the whole engine.
    """
    symbol = symbol.upper()
    if symbol in INSTRUMENTS:
        return INSTRUMENTS[symbol]

    from app.config import get_config
    from app.data.custom_instruments import load as load_custom

    custom = load_custom(get_config().data_dir)
    if symbol in custom:
        return custom[symbol]
    raise ValueError(f"Unknown instrument: {symbol}")
