"""What a backtest is charged: spread, swap and contract size, from Settings -> Costs or MetaTrader."""

import logging

from app.api.runtime import _mt5_bridge
from app.engine.swap import MONEY as SWAP_MONEY
from app.engine.swap import PRICE as SWAP_PRICE
from app.engine.swap import SwapSpec

# Contract size per lot, used to price Settings → Costs commission against
# the position that fixed-fractional sizing implies.
# Contract size only scales the commission charge (lots = risk / (stop x
# contract size)); the P&L itself comes from R multiples, so an imperfect
# value here cannot distort a return. Smaller is the conservative
# direction, because it implies more lots and therefore more commission.
CONTRACT_SIZE_BY_CATEGORY = {
    "fx": 100_000.0,
    "metal": 100.0,
    "index": 1.0,
    "commodity": 100.0,  # barrels per lot, on the cautious side of broker practice
    "crypto": 1.0,       # one coin per lot
}


def _manual_spread_points(costs: dict, symbol: str) -> float:
    """The configured manual spread for one symbol, in that symbol's points.

    A single global figure cannot serve the catalogue. A point is
    1/point_divisor, and the divisors span two orders of magnitude: ten
    points is one pip on EURUSD (divisor 100,000) and one cent on XAUUSD
    (divisor 1,000), about a seventeenth of what gold actually costs to
    cross. Campaigns bred with no terminal open charged that cent, and a
    cost that small is most of the distance between a backtest and a
    broker statement.
    """
    per_symbol = costs.get("manual_spread_by_symbol") or {}
    configured = per_symbol.get(symbol)
    return float(configured if configured is not None else costs["manual_spread_points"])


class SpreadUnavailable(ValueError):
    """AUTO SPREAD was asked for and MetaTrader could not provide it.

    A ValueError so every endpoint that already turns a bad configuration
    into a message for the user does the same with this one.
    """


def _spread_points(costs: dict, symbol: str, instrument) -> float:
    """The spread every backtest is charged, in the instrument's points.

    AUTO reads it from the broker, and only from the broker: MetaTrader
    has to be open. It used to fall back to the manual figure when no
    terminal answered, which meant a campaign labelled AUTO could be
    charged a number typed in months ago for a different instrument -
    and nobody could tell afterwards which of the two it had been.
    MANUAL is the user's number, per symbol when set, the default
    otherwise.
    """
    if costs["spread_mode"] != "AUTO":
        return _manual_spread_points(costs, symbol)
    # Attach to a running terminal; never launch one - a campaign
    # starting is not a reason to open a broker terminal.
    if not _mt5_bridge.is_connected():
        _mt5_bridge.connect()
    if not _mt5_bridge.is_connected():
        raise SpreadUnavailable(
            "Spread mode is AUTO, which reads the spread from MetaTrader 5, and no terminal is open. "
            "Open MetaTrader 5 and log in to your broker, or switch Settings -> Costs -> Spread mode to MANUAL."
        )
    quote = _mt5_bridge.get_live_quote(symbol)
    if quote is None or quote.spread_points <= 0:
        raise SpreadUnavailable(
            f"MetaTrader 5 is open but returned no quote for {symbol}. Check the symbol exists on this broker "
            "and the market is open, or switch Settings -> Costs -> Spread mode to MANUAL."
        )
    live = float(quote.spread_points * instrument.point_divisor)
    logging.getLogger("quantengine").info(
        "AUTO SPREAD: charging %s %.1f points, read from MetaTrader (%s).", symbol, live, quote.symbol
    )
    return live


# MetaTrader's SYMBOL_SWAP_MODE values this engine can price.
_MT5_SWAP_POINTS = 1
_MT5_SWAP_CURRENCY_DEPOSIT = 4


def _swap_spec(costs: dict, symbol: str, instrument) -> SwapSpec | None:
    """Overnight financing for `symbol`, by Settings -> Costs -> Swap mode.

    AUTO reads it from MetaTrader, which must be open - as with spread,
    there is no quiet fallback. MANUAL takes per-symbol long/short figures
    in the instrument's points (negative is a charge). OFF charges none.
    """
    mode = costs.get("swap_mode", "AUTO")
    if mode == "OFF":
        return None
    if mode != "AUTO":
        manual = (costs.get("manual_swap_by_symbol") or {}).get(symbol)
        if not manual:
            return None
        point = 1 / instrument.point_divisor
        return SwapSpec(float(manual.get("long", 0.0)) * point, float(manual.get("short", 0.0)) * point, SWAP_PRICE)

    if not _mt5_bridge.is_connected():
        _mt5_bridge.connect()
    quote = _mt5_bridge.get_swap(symbol) if _mt5_bridge.is_connected() else None
    if quote is None:
        raise SpreadUnavailable(
            f"Swap mode is AUTO, which reads {symbol}'s swap from MetaTrader 5, and no terminal answered. "
            "Open MetaTrader 5, or switch Settings -> Costs -> Swap mode to MANUAL or OFF."
        )
    triple_day = (quote["rollover3days"] - 1) % 7  # MT5 counts from Sunday, SwapSpec from Monday
    if quote["mode"] == _MT5_SWAP_POINTS:
        return SwapSpec(quote["long"] * quote["point"], quote["short"] * quote["point"], SWAP_PRICE, triple_day)
    if quote["mode"] == _MT5_SWAP_CURRENCY_DEPOSIT:
        return SwapSpec(quote["long"], quote["short"], SWAP_MONEY, triple_day)
    if quote["mode"] == 0:  # swaps disabled on this symbol
        return None
    logging.getLogger("quantengine").warning(
        "%s swap is quoted in MetaTrader mode %d, which this engine does not price yet; no swap is charged.",
        symbol, quote["mode"],
    )
    return None
