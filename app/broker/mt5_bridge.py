"""MetaTrader 5 bridge — REQ-DATA-1.2, REQ-DATA-1.3, design.md B.4.

The MT5/broker connection is used exclusively for live price reference
(spread for AUTO SPREAD) and order execution. It is NEVER a source of
breeding/backtest history — that is always the Dukascopy feed
(app/data/provider.py). Do not add a bars/rates/history fetch function to
this module; if you need historical data, use app.data.provider instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.broker.terminal_watch import forget as forget_terminal_state
from app.broker.terminal_watch import terminal_is_running

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

    def connect(self, launch: bool = False) -> bool:
        """Attach to a running terminal. Launch one only when asked.

        `mt5.initialize()` starts MetaTrader when none is running, which
        made simply opening QuantEngine open MetaTrader - the Dashboard
        polls this, so it happened on every launch whether or not anybody
        wanted it. Nothing here needs the terminal running: EAs are
        installed by writing files into its data directory and the
        Pilot's reports are read out of one, both of which work on a
        closed terminal. The connection is for live spread, and only
        while a campaign is being configured.
        """
        if mt5 is None:
            return False
        if not launch and not terminal_is_running():
            self._connected = False
            return False
        self._connected = bool(mt5.initialize())
        if self._connected:
            # A launch changes the answer the watcher just cached.
            forget_terminal_state()
        return self._connected

    def disconnect(self) -> None:
        if mt5 is not None and self._connected:
            mt5.shutdown()
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def resolve_symbol(self, symbol: str) -> str | None:
        """The broker's name for `symbol`.

        Brokers decorate names - EURUSD.r, EURUSDm, XAUUSD.pro - and an
        exact lookup against those returns nothing. Exact first, then the
        shortest broker symbol that starts with the plain name, which is
        the undecorated variant when a broker lists several.
        """
        if mt5 is None or not self._connected:
            return None
        if mt5.symbol_info(symbol) is not None:
            return symbol
        candidates = mt5.symbols_get(f"{symbol}*") or ()
        names = sorted((c.name for c in candidates), key=len)
        return names[0] if names else None

    def get_live_quote(self, symbol: str) -> LiveQuote | None:
        if mt5 is None or not self._connected:
            return None
        name = self.resolve_symbol(symbol)
        if name is None:
            return None
        # A symbol outside Market Watch has no ticks streaming to it.
        mt5.symbol_select(name, True)
        tick = mt5.symbol_info_tick(name)
        if tick is None or tick.bid <= 0 or tick.ask <= 0:
            return None
        return LiveQuote(
            symbol=name,
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
