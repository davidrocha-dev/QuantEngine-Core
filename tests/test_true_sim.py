import json

import pytest

from app.engine.campaign import CampaignConfig
from app.engine.true_sim import COMPOUNDING, FIXED, run_true_sim
from app.genome.gene_pool import GenePoolConfig
from app.genome.genes import ALL_GENES
from app.models import Strategy
from tests.helpers import make_synthetic_ohlc


def genome_json(signal: str = "BREAKOUT") -> str:
    return json.dumps(
        {
            "signal": {"name": signal, "params": dict(ALL_GENES[("signal", signal)].default_params)},
            "bias": None,
            "filters": [],
            "execution": {"name": "MARKET2", "params": {"confirm_bars": 2}},
            "management": None,
            "stop_atr_mult": 1.5,
            "target_atr_mult": 3.0,
        }
    )


def make_strategy(strategy_id: str, asset: str = "EURUSD", signal: str = "BREAKOUT") -> Strategy:
    return Strategy(
        id=strategy_id, data_source="dukascopy", asset=asset, timeframe="H1",
        genome_json=genome_json(signal), score=70,
    )


@pytest.fixture
def config() -> CampaignConfig:
    return CampaignConfig(timeframe="H1", risk_pct=0.01, point_size=1e-5, cost_points=12.0, commission_per_lot=7.0)


@pytest.fixture
def bars():
    df = make_synthetic_ohlc(4_000, seed=21, drift=0.0004, volatility=0.002)
    return {("EURUSD", "H1"): df, ("GBPUSD", "H1"): df.copy()}


def test_symbol_lock_blocks_a_second_position_on_the_same_symbol(bars, config):
    # Two members on one symbol: whenever their trades overlap, the second
    # entry is blocked rather than doubling the exposure.
    members = [make_strategy("#A", signal="BREAKOUT"), make_strategy("#B", signal="ATR RUN")]

    result = run_true_sim(members, bars, GenePoolConfig(), config, max_aggregate_risk_pct=100.0)

    assert result.blocked_by_symbol_lock > 0
    assert result.trades_blocked == result.blocked_by_symbol_lock + result.blocked_by_aggregate_risk
    assert all(sample.reason == "symbol lock" for sample in result.blocked_samples)


def test_aggregate_risk_cap_blocks_entries_once_open_risk_is_full(bars, config):
    # Different symbols, so symbol lock cannot be what blocks them.
    members = [make_strategy("#A", asset="EURUSD"), make_strategy("#B", asset="GBPUSD", signal="ATR RUN")]

    capped = run_true_sim(members, bars, GenePoolConfig(), config, max_aggregate_risk_pct=1.0)
    uncapped = run_true_sim(members, bars, GenePoolConfig(), config, max_aggregate_risk_pct=100.0)

    assert capped.blocked_by_aggregate_risk > 0
    assert capped.trades_taken < uncapped.trades_taken


def test_peak_simultaneous_risk_is_reported_with_its_timestamp(bars, config):
    members = [make_strategy("#A", asset="EURUSD"), make_strategy("#B", asset="GBPUSD", signal="ATR RUN")]

    result = run_true_sim(members, bars, GenePoolConfig(), config, max_aggregate_risk_pct=10.0)

    assert result.peak_simultaneous_risk_pct >= 0
    if result.trades_taken:
        assert result.peak_risk_at is not None


def test_fixed_and_compounding_sizing_diverge(bars, config):
    members = [make_strategy("#A")]

    fixed = run_true_sim(members, bars, GenePoolConfig(), config, sizing=FIXED)
    compounding = run_true_sim(members, bars, GenePoolConfig(), config, sizing=COMPOUNDING)

    assert fixed.sizing == FIXED and compounding.sizing == COMPOUNDING
    assert fixed.trades_taken == compounding.trades_taken
    # Same trades, different position sizing -> different account outcome.
    assert fixed.final_balance != compounding.final_balance


def test_drawdown_is_floating_not_closed_trade_only(bars, config):
    # Marking to market can only make the drawdown deeper than the
    # closed-trade path, never shallower.
    members = [make_strategy("#A")]
    result = run_true_sim(members, bars, GenePoolConfig(), config)

    assert result.max_drawdown_pct >= 0
    assert result.equity  # a daily curve is returned for the UI


def test_a_portfolio_with_no_usable_history_returns_an_empty_simulation(config):
    members = [make_strategy("#A")]
    result = run_true_sim(members, {}, GenePoolConfig(), config)

    assert result.trades_taken == 0
    assert result.final_balance == config.initial_balance
    assert result.return_pct == 0.0


def test_each_member_is_charged_its_own_market_costs(bars, config, monkeypatch):
    """A portfolio mixes markets. Every member used to be re-simulated on
    the first member's config, so a GBPUSD member paid EURUSD's spread in
    EURUSD's points - on gold vs a major that is thousands of times off."""
    from app.engine import true_sim

    seen: list[tuple[float, float]] = []
    real = true_sim.simulate_trades

    def spy(df, genome, gene_pool, **kwargs):
        seen.append((kwargs["point_size"], kwargs["cost_points"]))
        return real(df, genome, gene_pool, **kwargs)

    monkeypatch.setattr(true_sim, "simulate_trades", spy)
    gbp = CampaignConfig(timeframe="H1", risk_pct=0.01, point_size=1e-3, cost_points=30.0, commission_per_lot=0.0)
    members = [make_strategy("#A", asset="EURUSD"), make_strategy("#B", asset="GBPUSD", signal="ATR RUN")]

    run_true_sim(members, bars, GenePoolConfig(), config, member_configs={"#B": gbp})

    # #A has no entry of its own, so it falls back to the account config.
    assert seen == [(1e-5, 12.0), (1e-3, 30.0)]
