"""QuantEngine must not open MetaTrader by itself.

`MetaTrader5.initialize()` starts the terminal when none is running.
The Dashboard polls the connection status, so opening this app opened
MetaTrader — every launch, whether or not anybody wanted it.

Nothing here needs the terminal running. An EA is installed by writing
files into its data directory and the Pilot's reports are read out of
one; both work on a closed terminal. The connection is for live spread,
and only while a campaign is being configured.
"""

import pytest

from app.broker import mt5_bridge as bridge_module
from app.broker.mt5_bridge import MT5Bridge


class FakeMt5:
    """Stands in for the MetaTrader5 package, and records being started."""

    def __init__(self) -> None:
        self.initialize_calls = 0

    def initialize(self) -> bool:
        self.initialize_calls += 1
        return True

    def shutdown(self) -> None:
        pass


@pytest.fixture
def fake_mt5(monkeypatch):
    fake = FakeMt5()
    monkeypatch.setattr(bridge_module, "mt5", fake)
    return fake


def test_polling_the_status_does_not_start_the_terminal(fake_mt5, monkeypatch):
    # The whole point. This is what the Dashboard calls, on a timer.
    monkeypatch.setattr(bridge_module, "terminal_is_running", lambda: False)
    bridge = MT5Bridge()

    connected = bridge.connect()

    assert connected is False
    assert fake_mt5.initialize_calls == 0  # nothing was launched
    assert bridge.is_connected() is False


def test_a_terminal_that_is_already_open_is_attached_to(fake_mt5, monkeypatch):
    # Somebody with MetaTrader already running expects the card to say so.
    monkeypatch.setattr(bridge_module, "terminal_is_running", lambda: True)
    bridge = MT5Bridge()

    assert bridge.connect() is True
    assert fake_mt5.initialize_calls == 1
    assert bridge.is_connected() is True


def test_launching_is_possible_when_it_is_asked_for(fake_mt5, monkeypatch):
    # The Connect button. Starting a broker terminal is a decision, and
    # this is the path where somebody made it.
    monkeypatch.setattr(bridge_module, "terminal_is_running", lambda: False)
    bridge = MT5Bridge()

    assert bridge.connect(launch=True) is True
    assert fake_mt5.initialize_calls == 1


def test_without_the_metatrader_package_nothing_happens(monkeypatch):
    # macOS, Linux, or a Windows box without the terminal installed.
    monkeypatch.setattr(bridge_module, "mt5", None)
    bridge = MT5Bridge()

    assert bridge.connect() is False
    assert bridge.connect(launch=True) is False


def test_the_process_check_never_reports_a_terminal_it_cannot_see(monkeypatch):
    """Any failure answers False.

    Being wrong that way costs a card that says "not connected". Being
    wrong the other way launches a trading terminal nobody asked for.
    """
    from app.broker import terminal_watch

    monkeypatch.setattr(terminal_watch, "_running_process_names", lambda: (_ for _ in ()).throw(OSError("nope")))
    terminal_watch.forget()

    assert terminal_watch.terminal_is_running(use_cache=False) is False


def test_the_answer_is_cached_so_a_poll_is_not_a_process_scan(monkeypatch):
    from app.broker import terminal_watch

    calls = []

    def counted():
        calls.append(1)
        return {"terminal64.exe"}

    monkeypatch.setattr(terminal_watch, "_running_process_names", counted)
    terminal_watch.forget()

    assert terminal_watch.terminal_is_running() is True
    assert terminal_watch.terminal_is_running() is True
    assert len(calls) == 1

    terminal_watch.forget()
    assert terminal_watch.terminal_is_running() is True
    assert len(calls) == 2
