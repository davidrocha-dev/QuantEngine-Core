import random

import pytest

from app.codegen.mql5 import UnsupportedGeneError, generate_ea
from app.genome.gene_pool import GenePoolConfig
from app.genome.genes import ALL_GENES
from app.genome.genome import GeneChoice, Genome, random_genome


def make_genome(**overrides) -> Genome:
    defaults = dict(
        signal=GeneChoice("MACD", dict(ALL_GENES[("signal", "MACD")].default_params)),
        execution=GeneChoice("MARKET2", dict(ALL_GENES[("execution", "MARKET2")].default_params)),
        management=GeneChoice("EOD CLOSE", dict(ALL_GENES[("management", "EOD CLOSE")].default_params)),
        bias=GeneChoice("SMA", dict(ALL_GENES[("bias", "SMA")].default_params)),
        filters=[],
        stop_atr_mult=1.5,
        target_atr_mult=3.0,
    )
    defaults.update(overrides)
    return Genome(**defaults)


def test_generated_ea_is_named_after_the_hash_id():
    ea = generate_ea(make_genome(), "#D3332A", "EURUSD", "M15")
    assert ea.filename == "QE_D3332A.mq5"


def test_generated_source_is_pure_ascii():
    # MetaEditor reads plain source; a stray em dash lands as mojibake.
    ea = generate_ea(make_genome(), "#D3332A", "EURUSD", "M15")
    ea.source.encode("ascii")


def test_genome_risk_geometry_reaches_the_ea():
    ea = generate_ea(make_genome(stop_atr_mult=2.25, target_atr_mult=4.5), "#AAA111", "EURUSD", "H1")
    assert "2.25 * atr" in ea.source
    assert "4.5 * atr" in ea.source


def test_gene_parameters_reach_the_ea():
    genome = make_genome(
        signal=GeneChoice("BREAKOUT", {"period": 37}),
        bias=GeneChoice("RSI", {"period": 9}),
    )
    ea = generate_ea(genome, "#BBB222", "EURUSD", "M15")
    assert "HighestHigh(37, 2)" in ea.source
    assert "iRSI(_Symbol, PERIOD_CURRENT, 9, PRICE_CLOSE)" in ea.source


def test_each_strategy_gets_its_own_magic_number():
    first = generate_ea(make_genome(), "#AAA111", "EURUSD", "M15").source
    second = generate_ea(make_genome(), "#BBB222", "EURUSD", "M15").source
    magic_of = lambda source: [line for line in source.splitlines() if "MagicNumber" in line and "input" in line][0]
    assert magic_of(first) != magic_of(second)


def test_a_genome_without_a_bias_gene_never_disagrees():
    ea = generate_ea(make_genome(bias=None), "#CCC333", "EURUSD", "M15")
    assert "no BIAS gene in this genome" in ea.source


def test_filters_become_entry_vetoes():
    genome = make_genome(filters=[GeneChoice("NO FRI OPEN", {"cutoff_hour": 15})])
    ea = generate_ea(genome, "#DDD444", "EURUSD", "M15")
    assert "IsFridayAfter(15)" in ea.source


def test_the_news_gene_is_refused_rather_than_silently_dropped():
    # The EA has no calendar feed: exporting it would trade without a
    # filter the backtest had.
    genome = make_genome(filters=[GeneChoice("EVENT", {"mode": "FTMO"})])
    with pytest.raises(UnsupportedGeneError, match="EVENT"):
        generate_ea(genome, "#EEE555", "EURUSD", "M15")


def test_management_gene_drives_the_position_management_block():
    partial = generate_ea(
        make_genome(management=GeneChoice("PARTIAL TP", {"r_multiple": 1.5, "close_fraction": 0.4})),
        "#FFF666", "EURUSD", "M15",
    )
    assert "PositionClosePartial" in partial.source
    assert "1.5" in partial.source

    none = generate_ea(make_genome(management=None), "#FFF777", "EURUSD", "M15")
    assert "no MANAGEMENT gene" in none.source


def test_execution_gene_picks_the_order_type():
    market = generate_ea(make_genome(execution=GeneChoice("MARKET2", {"confirm_bars": 2})), "#1", "EURUSD", "M15")
    assert "Trade.Buy(" in market.source

    limit = generate_ea(make_genome(execution=GeneChoice("LIMIT", {"offset_atr": 0.4})), "#2", "EURUSD", "M15")
    assert "Trade.BuyLimit(" in limit.source

    stop = generate_ea(make_genome(execution=GeneChoice("STOP", {"offset_atr": 0.25})), "#3", "EURUSD", "M15")
    assert "Trade.BuyStop(" in stop.source


def test_every_indicator_handle_is_declared_created_and_released():
    genome = make_genome(
        signal=GeneChoice("STOCH", {"k_period": 14, "d_period": 3}),
        bias=GeneChoice("PSAR", {"step": 0.02, "max_step": 0.2}),
        filters=[GeneChoice("REGIME", {"period": 14, "threshold": 20, "require": "trend"})],
    )
    source = generate_ea(genome, "#HANDLES", "EURUSD", "M15").source

    for handle in ("h_stoch_0", "h_sar_1", "h_adx_f_10"):
        assert f"int {handle} = INVALID_HANDLE;" in source
        assert f"   {handle} = i" in source
        assert f"IndicatorRelease({handle})" in source


def test_random_genomes_either_export_or_say_which_gene_stopped_them():
    random.seed(11)
    pool = GenePoolConfig()
    exported = 0
    for _ in range(120):
        genome = random_genome(pool)
        try:
            source = generate_ea(genome, "#RANDOM", "EURUSD", "M15").source
        except UnsupportedGeneError as exc:
            # The only gene without an MQL5 equivalent today.
            assert "EVENT" in str(exc)
            continue
        assert "OnTick" in source and "OnInit" in source
        source.encode("ascii")
        exported += 1
    assert exported > 0


def test_a_waiting_pending_order_blocks_another_entry():
    """The exported EA used to check only for open positions.

    A LIMIT/LIMIT2/STOP entry sits unfilled for up to
    PENDING_ORDER_WINDOW_BARS bars. While one waited, HasOpenPosition()
    was false, so every further bar whose signal still held placed
    another order on top of it. Several then filled, giving simultaneous
    positions at full risk each — the backtester has exactly one order in
    flight, ever. Exported from the Vault and run in the MetaTrader
    tester over the same year, that was 2,247 trades against the
    backtester's 638, with a 28.6% drawdown against 9.0%.
    """
    for execution in ("LIMIT", "LIMIT2", "STOP"):
        genome = make_genome(
            execution=GeneChoice(execution, dict(ALL_GENES[("execution", execution)].default_params))
        )
        source = generate_ea(genome, "#PENDING", "XAUUSD", "M15").source

        assert "bool HasPendingOrder()" in source, execution
        assert "OrdersTotal()" in source, execution
        assert "if(HasOpenPosition() || HasPendingOrder()) return;" in source, execution
