"""MetaTrader 5 bridge — REQ-DATA-1.2, REQ-DATA-1.3, design.md B.4.

The MT5/broker connection is used exclusively for live price reference
(spread for AUTO SPREAD) and order execution. It is NEVER a source of
breeding/backtest history — that is always the Dukascopy feed
(app/data/provider.py). Do not add a bars/rates/history fetch function to
this module; if you need historical data, use app.data.provider instead.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import MetaTrader5 as mt5  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - only available on Windows w/ terminal
    mt5 = None


@dataclass(frozen=True)
class LiveQuote:
    symbol: str
    bid: float
    ask: float
    spread_points: float


class MT5Bridge:
    """Live price reference and order execution only. See module docstring."""

    def __init__(self) -> None:
        self._connected = False

    def connect(self) -> bool:
        if mt5 is None:
            return False
        self._connected = bool(mt5.initialize())
        return self._connected

    def disconnect(self) -> None:
        if mt5 is not None and self._connected:
            mt5.shutdown()
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_live_quote(self, symbol: str) -> LiveQuote | None:
        if mt5 is None or not self._connected:
            return None
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return LiveQuote(
            symbol=symbol,
            bid=tick.bid,
            ask=tick.ask,
            spread_points=(tick.ask - tick.bid),
        )

    def get_server_info(self) -> dict | None:
        if mt5 is None or not self._connected:
            return None
        account = mt5.account_info()
        terminal = mt5.terminal_info()
        if account is None or terminal is None:
            return None
        return {
            "server": account.server,
            "is_demo": "demo" in account.server.lower(),
            "is_metaquotes_demo": account.server == "MetaQuotes-Demo",
        }

    def place_order(self, request: dict) -> object | None:
        if mt5 is None or not self._connected:
            return None
        return mt5.order_send(request)
