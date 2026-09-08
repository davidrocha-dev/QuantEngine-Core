"""Guards REQ-DATA-1.2/1.3: MT5 must never become a historical data source."""

import inspect

from app.broker import mt5_bridge

FORBIDDEN_NAME_FRAGMENTS = ("history", "rates", "copy_rates", "backtest")


def test_mt5_bridge_exposes_no_historical_data_methods():
    members = inspect.getmembers(mt5_bridge.MT5Bridge, predicate=inspect.isfunction)
    for name, _ in members:
        lowered = name.lower()
        assert not any(fragment in lowered for fragment in FORBIDDEN_NAME_FRAGMENTS), (
            f"MT5Bridge.{name} looks like a historical-data method — forbidden by REQ-DATA-1.2/1.3"
        )
