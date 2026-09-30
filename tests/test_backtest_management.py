"""Management genes against the stop and target on the same bar."""

from dataclasses import replace

import pandas as pd

from app.engine.backtest import simulate_trades
from app.genome.gene_pool import GenePoolConfig
from app.genome.genes import ALL_GENES
from app.genome.genome import Genome


def _bars(stop_bar_close: float) -> pd.DataFrame:
    rows = []
    for i in range(40):
        rows.append({"open": 1.1000, "high": 1.1005, "low": 1.0995, "close": 1.1000, "volume": 1.0})
    # Bar 22 (22:00 UTC): its low runs through the stop, then it closes
    # back above the entry.
    rows[22] = {"open": 1.1000, "high": 1.1015, "low": 1.0950, "close": stop_bar_close, "volume": 1.0}
    df = pd.DataFrame(rows)
    df.insert(0, "timestamp", pd.date_range("2024-01-01", periods=len(df), freq="1h", tz="UTC"))
    return df


def _one_signal_at(bar: int):
    def compute(df, htf_df, params):
        signal = pd.Series(0, index=df.index)
        signal.iloc[bar] = 1
        return signal

    return compute


def _genome(management: str, params: dict) -> Genome:
    return Genome.from_dict({
        "signal": {"name": "BREAKOUT", "params": dict(ALL_GENES[("signal", "BREAKOUT")].default_params)},
        "bias": None, "filters": [],
        "execution": {"name": "MARKET2", "params": {"confirm_bars": 1}},
        "management": {"name": management, "params": params},
        "stop_atr_mult": 1.5, "target_atr_mult": 10.0,
    })


def test_a_stop_hit_inside_the_bar_beats_a_close_at_that_bars_close(monkeypatch):
    """EOD CLOSE acts on the bar's close; the stop is an order that fills
    inside it. The close used to win the tie, so a stopped-out trade was
    booked at the close - here, above the entry, as a winner."""
    spec = ALL_GENES[("signal", "BREAKOUT")]
    monkeypatch.setitem(ALL_GENES, ("signal", "BREAKOUT"), replace(spec, compute=_one_signal_at(20)))

    result = simulate_trades(_bars(stop_bar_close=1.1010), _genome("EOD CLOSE", {"close_hour": 21}), GenePoolConfig())

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_idx == 22
    assert trade.exit_reason == "stop"
    assert trade.pnl < 0


def test_friday_profit_cannot_turn_a_stopped_trade_into_a_winner(monkeypatch):
    spec = ALL_GENES[("signal", "BREAKOUT")]
    monkeypatch.setitem(ALL_GENES, ("signal", "BREAKOUT"), replace(spec, compute=_one_signal_at(20)))
    df = _bars(stop_bar_close=1.1010)
    # 2024-01-05 is a Friday.
    df["timestamp"] = pd.date_range("2024-01-05", periods=len(df), freq="1h", tz="UTC")

    result = simulate_trades(df, _genome("FRIDAY PROFIT", {"lock_hour": 18}), GenePoolConfig())

    assert result.trades[0].exit_reason == "stop"
    assert result.trades[0].pnl < 0


def test_management_still_closes_before_a_later_stop(monkeypatch):
    spec = ALL_GENES[("signal", "BREAKOUT")]
    monkeypatch.setitem(ALL_GENES, ("signal", "BREAKOUT"), replace(spec, compute=_one_signal_at(20)))
    df = _bars(stop_bar_close=1.1010)
    # Move the stop-out to bar 30; EOD fires at bar 22's close first.
    df.loc[22, ["high", "low", "close"]] = [1.1005, 1.0995, 1.1000]
    df.loc[30, ["high", "low", "close"]] = [1.1005, 1.0950, 1.1000]

    result = simulate_trades(df, _genome("EOD CLOSE", {"close_hour": 21}), GenePoolConfig())

    assert result.trades[0].exit_idx == 22
    assert result.trades[0].exit_reason == "EOD CLOSE"


def _partial_bars() -> pd.DataFrame:
    rows = [{"open": 1.1000, "high": 1.1005, "low": 1.0995, "close": 1.1000, "volume": 1.0} for _ in range(40)]
    # Bar 23 reaches the partial level, bar 25 comes back through the entry.
    rows[23] = {"open": 1.1000, "high": 1.1030, "low": 1.0999, "close": 1.1020, "volume": 1.0}
    rows[25] = {"open": 1.1010, "high": 1.1010, "low": 1.0980, "close": 1.0990, "volume": 1.0}
    df = pd.DataFrame(rows)
    df.insert(0, "timestamp", pd.date_range("2024-01-01", periods=len(df), freq="1h", tz="UTC"))
    return df


def test_a_partial_below_the_minimum_lot_banks_nothing(monkeypatch):
    """At 0.01 lots half a position is 0.005, which no broker fills. The
    EA closes nothing and moves the stop to breakeven anyway; the backtest
    banked half the position at the partial level regardless."""
    spec = ALL_GENES[("signal", "BREAKOUT")]
    monkeypatch.setitem(ALL_GENES, ("signal", "BREAKOUT"), replace(spec, compute=_one_signal_at(20)))
    genome = _genome("PARTIAL TP", {"r_multiple": 1.0, "close_fraction": 0.5})
    ladder = {"lot_step": 0.01, "min_lot": 0.01, "max_lot": 100.0}

    # A 10k account at 1% on this stop sizes well above the minimum...
    roomy = simulate_trades(_partial_bars(), genome, GenePoolConfig(), **ladder).trades[0]
    # ...a 100 account at 1% sizes to the minimum lot.
    tight = simulate_trades(_partial_bars(), genome, GenePoolConfig(), initial_balance=100.0, **ladder).trades[0]

    assert roomy.partial_closed and roomy.r_multiple > 0
    assert not tight.partial_closed
    assert tight.r_multiple == 0.0  # the whole position out at breakeven
