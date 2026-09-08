import pandas as pd

from app.genome.genes import (
    ALL_GENES,
    BIAS_GENES,
    EXECUTION_GENES,
    FILTER_GENES,
    MANAGEMENT_GENES,
    SIGNAL_GENES,
)
from tests.helpers import make_synthetic_ohlc


def test_exact_gene_counts_per_req_engine_1_2():
    assert len(SIGNAL_GENES) == 16
    assert len(BIAS_GENES) == 11
    assert len(FILTER_GENES) == 12
    assert len(EXECUTION_GENES) == 4
    assert len(MANAGEMENT_GENES) == 3
    assert len(ALL_GENES) == 46


def test_authoritative_gene_names_present():
    # REQ-ENGINE-1.2 authoritative table — spot-check names that also
    # appear in more than one layer (CCI, RSI, SMA) to guard the
    # (layer, name)-keyed ALL_GENES merge against silent collisions.
    assert ("signal", "CCI") in ALL_GENES
    assert ("filter", "CCI") in ALL_GENES
    assert ("bias", "RSI") in ALL_GENES
    assert ("filter", "RSI") in ALL_GENES
    assert ("bias", "SMA") in ALL_GENES
    assert ("filter", "SMA") in ALL_GENES


def test_every_signal_gene_computes_without_error_and_returns_valid_range():
    df = make_synthetic_ohlc(300)
    for name, spec in SIGNAL_GENES.items():
        result = spec.compute(df, None, spec.default_params)
        assert len(result) == len(df), f"{name} returned wrong length"
        assert result.isin([-1, 0, 1]).all(), f"{name} returned values outside {{-1,0,1}}"


def test_every_bias_gene_computes_without_error_and_returns_valid_range():
    df = make_synthetic_ohlc(300)
    for name, spec in BIAS_GENES.items():
        result = spec.compute(df, None, spec.default_params)
        assert len(result) == len(df), f"{name} returned wrong length"
        assert result.isin([-1, 0, 1]).all(), f"{name} returned values outside {{-1,0,1}}"


def test_every_filter_gene_computes_without_error_and_returns_boolean():
    df = make_synthetic_ohlc(300)
    for name, spec in FILTER_GENES.items():
        result = spec.compute(df, None, spec.default_params)
        assert len(result) == len(df), f"{name} returned wrong length"
        assert result.dtype == bool, f"{name} did not return a boolean series"


def test_htf_bias_neutral_when_htf_disabled():
    df = make_synthetic_ohlc(100)
    spec = BIAS_GENES["HTF"]
    result = spec.compute(df, None, spec.default_params)
    assert (result == 0).all()


def test_breakout_fires_long_on_new_high():
    df = make_synthetic_ohlc(60)
    # Force a clean breakout at the last bar.
    df.loc[df.index[-1], "close"] = df["high"].iloc[:-1].max() + 10
    spec = SIGNAL_GENES["BREAKOUT"]
    result = spec.compute(df, None, {"period": 20})
    assert result.iloc[-1] == 1
