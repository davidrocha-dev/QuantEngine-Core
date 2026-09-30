"""Overnight financing, charged per rollover on the broker's clock."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.engine.backtest import simulate_trades
from app.engine.swap import MONEY, SwapSpec, nights_held
from app.genome.gene_pool import GenePoolConfig
from app.genome.genome import GeneChoice, Genome


def _day(text):
    return np.datetime64(text, "D")


def test_an_intraday_trade_pays_no_swap():
    assert nights_held(_day("2024-01-10"), _day("2024-01-10"), triple_day=2) == 0


def test_one_night_is_one_rollover():
    assert nights_held(_day("2024-01-08"), _day("2024-01-09"), triple_day=2) == 1  # Mon -> Tue


def test_wednesday_night_counts_three():
    assert nights_held(_day("2024-01-10"), _day("2024-01-11"), triple_day=2) == 3  # Wed -> Thu


def test_the_weekend_is_not_charged_twice():
    # Fri -> Mon: Friday's rollover only; the weekend is in Wednesday's triple.
    assert nights_held(_day("2024-01-12"), _day("2024-01-15"), triple_day=2) == 1
    # A full week Mon -> next Mon: 4 single nights + one triple.
    assert nights_held(_day("2024-01-08"), _day("2024-01-15"), triple_day=2) == 7


def _held_for_days():
    """A long that enters on day one and is stopped out three days later."""
    start = datetime(2024, 1, 8)  # Monday
    n = 24 * 4
    stamps = [start + timedelta(hours=i) for i in range(n)]
    close = np.full(n, 100.0)
    frame = pd.DataFrame({
        "timestamp": pd.DatetimeIndex(stamps).tz_localize("UTC"), "broker_time": stamps,
        "open": close, "high": close + 0.05, "low": close - 0.05, "close": close, "volume": 1.0,
    })
    frame.loc[80, "low"] = 90.0  # the stop, on Thursday
    return frame


def test_swap_reaches_the_pnl_and_the_r_of_a_trade_held_overnight(monkeypatch):
    import app.engine.backtest as backtest

    df = _held_for_days()
    spec = backtest.ALL_GENES[("signal", "BREAKOUT")]
    once = pd.Series(0, index=df.index)
    once.iloc[20] = 1
    monkeypatch.setitem(
        backtest.ALL_GENES, ("signal", "BREAKOUT"),
        type(spec)(spec.name, spec.layer, spec.default_params, lambda d, h, p: once),
    )
    genome = Genome(signal=GeneChoice("BREAKOUT", {}), execution=GeneChoice("MARKET2", {"confirm_bars": 1}),
                    stop_atr_mult=10.0, target_atr_mult=50.0)

    plain = simulate_trades(df, genome, GenePoolConfig(), contract_size=1.0).trades[0]
    charged = simulate_trades(df, genome, GenePoolConfig(), contract_size=1.0,
                              swap=SwapSpec(-2.0, 1.0, MONEY, triple_day=2)).trades[0]

    # Mon -> Thu: Mon, Tue and Wednesday's triple = 5 nights at -2.00 per lot.
    lots = abs(plain.pnl) / abs(plain.r_multiple) / (abs(plain.entry_price - plain.stop_price) * 1.0)
    assert charged.pnl - plain.pnl == pytest.approx(-2.0 * 5 * lots, rel=1e-6)
    assert charged.r_multiple < plain.r_multiple
