"""Every strategy parameter the search varies is one both sides trade on.

A parameter that the backtest reads and the EA ignores (or the other way
round) is a strategy that changes when it is exported. These tests vary
each searchable parameter of every exportable gene and check that both
the Python gene and the MQL5 it exports respond.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from app.codegen.genes_mql5 import emit
from app.data.provider import resample_bars
from app.data.broker_clock import DEFAULT_CLOCK
from app.genome.genes import BIAS_GENES, FILTER_GENES, SIGNAL_GENES

# Genes the export deliberately does not reproduce, each for a documented reason.
NOT_TRADED = {("filter", "EVENT"), ("filter", "CONSEC")}

GENES = [
    (layer, name, spec)
    for layer, catalog in (("signal", SIGNAL_GENES), ("bias", BIAS_GENES), ("filter", FILTER_GENES))
    for name, spec in catalog.items()
    if (layer, name) not in NOT_TRADED
]


@pytest.fixture(scope="module")
def bars():
    m1 = pd.read_parquet("app/data/bundled/XAUUSD_M1.parquet")
    m15 = resample_bars(m1, "M15", DEFAULT_CLOCK).reset_index(drop=True)
    h1 = resample_bars(m1, "H1", DEFAULT_CLOCK).reset_index(drop=True)
    return m15, h1


def _other_value(space, current):
    if space[0] == "choice":
        return next(option for option in space[1] if option != current)
    low, high = space[1], space[2]
    return high if current != high else low


@pytest.mark.parametrize("layer,name,spec", GENES, ids=[f"{l}:{n}" for l, n, _ in GENES])
def test_defaults_sit_inside_the_search_space(layer, name, spec):
    for key, space in spec.param_space.items():
        value = spec.default_params[key]
        if space[0] == "choice":
            assert value in space[1], key
        else:
            assert space[1] <= value <= space[2], key


@pytest.mark.parametrize("layer,name,spec", GENES, ids=[f"{l}:{n}" for l, n, _ in GENES])
def test_every_searched_parameter_reaches_the_ea(layer, name, spec):
    base = copy.deepcopy(spec.default_params)
    exported = vars(emit(layer, name, base, 0))
    for key, space in spec.param_space.items():
        changed = {**base, key: _other_value(space, base[key])}
        assert vars(emit(layer, name, changed, 0)) != exported, f"the EA ignores {key}"


@pytest.fixture(scope="module")
def long_bars():
    """A year of synthetic hourly bars, for the genes whose parameters only
    show on a longer history than the bundled two weeks (ADR's day count,
    PSAR's acceleration cap)."""
    from tests.helpers import make_synthetic_ohlc

    df = make_synthetic_ohlc(8000, seed=3, volatility=0.004)
    return df, df.iloc[::4].reset_index(drop=True)


@pytest.mark.parametrize("layer,name,spec", GENES, ids=[f"{l}:{n}" for l, n, _ in GENES])
def test_every_searched_parameter_changes_what_the_backtest_does(layer, name, spec, bars, long_bars):
    base = copy.deepcopy(spec.default_params)
    for key, space in spec.param_space.items():
        changed = {**base, key: _other_value(space, base[key])}
        responds = False
        for df, htf in (bars, long_bars):
            reference = spec.compute(df, htf, base).fillna(0).astype(float).to_numpy()
            result = spec.compute(df, htf, changed).fillna(0).astype(float).to_numpy()
            if not np.array_equal(result, reference):
                responds = True
                break
        assert responds, f"the backtest ignores {key}"


def test_the_htf_bias_reads_the_htf_bar_that_has_just_closed(bars):
    """The EA reads HtfClose(1) at the open of the next bar: at the close of
    the last M15 bar of an hour, that is the H1 bar closing with it. The
    backtest used to read the one before, on 7.6% of bars."""
    df, htf = bars
    bias = BIAS_GENES["HTF"].compute(df, htf, {"period": 5})
    sma = htf["close"].rolling(5).mean()
    raw = np.select([htf["close"] > sma, htf["close"] < sma], [1, -1], 0)
    htf_closes = pd.DatetimeIndex(htf["timestamp"]) + pd.Timedelta(hours=1)
    bar_closes = pd.DatetimeIndex(df["timestamp"]) + pd.Timedelta(minutes=15)
    expected = pd.Series(raw, index=htf_closes).reindex(bar_closes, method="ffill").fillna(0).to_numpy()
    assert np.array_equal(bias.to_numpy(), expected)
