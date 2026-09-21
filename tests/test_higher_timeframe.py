"""The HIGHER TF context timeframe.

The failure this guards against is silent: loading a timeframe *below*
the traded one as "higher context" raises nothing and returns a real
series, so every gene that consults it reads the wrong thing and the
campaign merely produces worse strategies.
"""

from app.main import _higher_timeframe


def settings(mode: str = "AUTO", manual: str | None = None) -> dict:
    return {"higher_tf_mode": mode, "higher_tf_manual_timeframe": manual}


def test_auto_uses_the_mapped_timeframe():
    assert _higher_timeframe("M15", settings()) == "H1"
    assert _higher_timeframe("H1", settings()) == "H4"


def test_auto_covers_the_two_fastest_timeframes():
    # These were absent, which left the toggle disabled on the timeframes
    # people reach for first.
    assert _higher_timeframe("M1", settings()) == "M15"
    assert _higher_timeframe("M5", settings()) == "H1"


def test_manual_is_honoured_when_it_is_actually_higher():
    assert _higher_timeframe("M15", settings("MANUAL", "H4")) == "H4"
    assert _higher_timeframe("M15", settings("MANUAL", "M30")) == "M30"


def test_manual_below_the_trading_timeframe_falls_back_to_the_mapping():
    # Breed on M15 with a manual H1 context, then move up to H4: the
    # stored H1 is no longer above what is being traded.
    assert _higher_timeframe("H4", settings("MANUAL", "H1")) == "D1"


def test_manual_equal_to_the_trading_timeframe_is_not_higher():
    assert _higher_timeframe("H1", settings("MANUAL", "H1")) == "H4"


def test_unset_manual_falls_back_even_in_manual_mode():
    assert _higher_timeframe("M30", settings("MANUAL", None)) == "H1"


def test_unknown_timeframes_do_not_raise():
    assert _higher_timeframe("W1", settings("MANUAL", "MN")) is None
