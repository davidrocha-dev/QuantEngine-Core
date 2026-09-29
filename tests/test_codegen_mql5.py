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


def test_time_based_management_waits_for_a_closed_bar():
    """The EA ran its MANAGEMENT gene on every tick.

    FRIDAY PROFIT closes any position in profit after lock_hour. Per
    tick, that closes at the first profitable price after the hour, which
    clears HasOpenPosition() and lets the next bar enter again — all
    Friday evening. Strategy #6F0A16 exported from the Vault and run over
    its own out-of-sample window put ~150 entries on Fridays against ~70
    on every other weekday, where the backtester, which evaluates this
    gene only when a bar closes, put 55 there. 433 trades against 315,
    at a win rate 9 points higher on much smaller average wins.

    PARTIAL TP is deliberately not gated: _run_partial_tp scans the bar's
    high and low, so the backtester really does act on it intrabar.
    """
    for management in ("FRIDAY PROFIT", "EOD CLOSE"):
        genome = make_genome(
            management=GeneChoice(management, dict(ALL_GENES[("management", management)].default_params))
        )
        source = generate_ea(genome, "#FRIDAY", "XAUUSD", "M15").source

        assert "void ManageOpenPosition(bool newBar)" in source, management
        body = source.split("void ManageOpenPosition(bool newBar)")[1].split("void TryEnter()")[0]
        assert "if(!newBar) return;" in body, management

    partial = generate_ea(
        make_genome(management=GeneChoice("PARTIAL TP", {"r_multiple": 1.0, "close_fraction": 0.5})),
        "#PARTIAL", "XAUUSD", "M15",
    ).source
    body = partial.split("void ManageOpenPosition(bool newBar)")[1].split("void TryEnter()")[0]
    assert "if(!newBar) return;" not in body

    # And management is still reached on every tick, ahead of the guard
    # that returns until a bar closes.
    assert "ManageOpenPosition(newBar);" in partial
    assert partial.index("ManageOpenPosition(newBar);") < partial.index("LastBarTime = currentBar;")


def test_market2_waits_out_its_confirmation_bars():
    """The EA ignored confirm_bars and entered on the bar after the signal.

    `_resolve_entry` fills MARKET2 at the open of signal_idx +
    confirm_bars and does not re-test the signal while it waits. The EA's
    MARKET2 branch bought at the next tick after the signal bar closed,
    whatever the gene said. On strategy #6F0A16 (confirm_bars 4) over its
    own out-of-sample window that alone was 368 trades against the
    engine's 313, and it is most of the 433 MetaTrader reported.
    """
    for confirm in (1, 4):
        genome = make_genome(execution=GeneChoice("MARKET2", {"confirm_bars": confirm}))
        source = generate_ea(genome, "#CONFIRM", "XAUUSD", "M15").source

        assert f"BarsUntilEntry = {confirm} - 1;" in source, confirm
        # One bar of confirmation is what entering immediately already
        # meant, so it needs no countdown in OnTick.
        assert ("if(BarsUntilEntry > 0)" in source) == (confirm > 1), confirm

    stop = generate_ea(
        make_genome(execution=GeneChoice("STOP", {"offset_atr": 0.25})), "#NOCONFIRM", "XAUUSD", "M15"
    ).source
    assert "BarsUntilEntry" not in stop


def test_time_of_day_rules_read_utc_not_the_broker_clock():
    """FRIDAY PROFIT, EOD CLOSE and the SESSION filter were bred against
    UTC bars and asked the broker what time it was. A UTC+3 server begins
    its Friday on Thursday at 21:00 UTC and reaches lock_hour three hours
    early: replaying #6F0A16 with that offset moved its Friday entries
    from 64 to 96 and its trade count from 368 to 394.

    TimeGMT() alone does not fix it. The exported EA was run on an
    IC Markets EU server and fired FRIDAY PROFIT 45 times against the
    engine's 12 — the behaviour of a clock three hours ahead, which is
    that server's summer time. The offset has to be resolvable by hand,
    because the tester will not tell the EA what it is.
    """
    genome = make_genome(
        management=GeneChoice("FRIDAY PROFIT", {"lock_hour": 19}),
        execution=GeneChoice("MARKET2", {"confirm_bars": 2}),
    )
    source = generate_ea(genome, "#UTC", "XAUUSD", "M15").source

    assert "input int    ServerGmtOffsetHours" in source
    assert "datetime UtcNow() { return TimeCurrent() - GmtOffsetSeconds; }" in source
    # Nothing reads a weekday or an hour off the broker's clock directly.
    assert "TimeToStruct(TimeCurrent(), now)" not in source
    assert "TimeToStruct(TimeGMT(), now)" not in source
    assert source.count("TimeToStruct(UtcNow(), now)") == 1  # IsFridayAfter
    # FRIDAY PROFIT judges the bar that just closed, in UTC, as the
    # backtester does - not the clock at the next tick.
    assert "TimeToStruct(ClosedBarUtc(), bar)" in source
    assert "iTime(_Symbol, PERIOD_CURRENT, 1) - GmtOffsetSeconds" in source


# --- Execution the backtest can reproduce ----------------------------------


def test_the_ea_has_no_spread_ceiling_unless_asked_for_one():
    """The export passed the *cost* spread (10 points by default) as the
    ceiling, and the EA skipped every entry above it - most of them on
    gold or a standard account. The backtest skips none."""
    source = generate_ea(make_genome(), "#SPREAD", "XAUUSD", "M15").source
    assert "input double MaxSpreadPoints  = 0.0;" in source


def test_the_ea_closes_a_trade_after_the_backtests_max_hold():
    from app.engine.backtest import MAX_HOLD_BARS

    source = generate_ea(make_genome(), "#HOLD", "EURUSD", "M5").source
    assert f"input int    MaxHoldBars      = {MAX_HOLD_BARS};" in source
    assert "if(newBar) CloseStalePositions();" in source


def test_resting_orders_expire_by_bar_count_and_fall_back_to_market_when_passed():
    for execution in ("LIMIT", "LIMIT2", "STOP"):
        genome = make_genome(execution=GeneChoice(execution, dict(ALL_GENES[("execution", execution)].default_params)))
        source = generate_ea(genome, "#PEND", "EURUSD", "M15").source
        assert "ORDER_TIME_SPECIFIED, expiry" not in source and "ORDER_TIME_GTC" in source, execution
        assert "#define PENDING_BARS 5" in source, execution
        assert "ExpirePendingOrders();" in source, execution
        # Expired before the next signal is read.
        assert source.index("ExpirePendingOrders();") < source.index("   TryEnter();\n}")
        assert "PlacePendingOrder(direction," in source, execution


def test_orders_are_normalised_and_use_the_symbols_filling_mode():
    source = generate_ea(make_genome(), "#HYGIENE", "XAUUSD", "M15").source
    assert "Trade.SetTypeFillingBySymbol(_Symbol);" in source
    assert "NormalizePrice(price - direction * stopDistance)" in source
    assert "StopsAllowed(stopDistance, targetDistance)" in source


def test_partial_tp_does_not_divide_by_a_breakeven_stop():
    source = generate_ea(
        make_genome(management=GeneChoice("PARTIAL TP", {"r_multiple": 1.0, "close_fraction": 0.5})),
        "#BE", "XAUUSD", "M15",
    ).source
    body = source.split("void ManageOpenPosition(bool newBar)")[1]
    assert body.index("if(stopDistance <= _Point) continue;") < body.index("/ stopDistance")


def test_indicators_are_mt5s_equivalents_of_the_python_genes():
    stoch = generate_ea(make_genome(signal=GeneChoice("STOCH", {"k_period": 14, "d_period": 3})), "#S", "EURUSD", "H1").source
    assert "iStochastic(_Symbol, PERIOD_CURRENT, 14, 3, 1, MODE_SMA, STO_LOWHIGH)" in stoch
    adx = generate_ea(make_genome(bias=GeneChoice("ADX", {"period": 14, "threshold": 20})), "#A", "EURUSD", "H1").source
    assert "iADXWilder(" in adx and "iADX(" not in adx


def test_consec_is_exported_as_the_no_op_it_was_backtested_as():
    """It used to veto on the EA's own losing streak, which never resets
    once the EA stops trading: N losses in a row switched it off for good."""
    genome = make_genome(filters=[GeneChoice("CONSEC", {"max_losses": 3})])
    source = generate_ea(genome, "#CONSEC", "EURUSD", "H1").source
    assert "ConsecutiveLosses" not in source


def test_htf_bias_follows_the_context_the_strategy_was_bred_with():
    genome = make_genome(bias=GeneChoice("HTF", {"period": 20}))
    off = generate_ea(genome, "#HTF", "EURUSD", "M15", higher_timeframe=None).source
    assert "HtfClose(1) > Val(" not in off  # neutral, as the backtest had it
    manual = generate_ea(genome, "#HTF", "EURUSD", "M15", higher_timeframe="H4").source
    assert "ENUM_TIMEFRAMES HigherTimeframe() { return PERIOD_H4; }" in manual
    assert "HtfClose(1) > Val(" in manual
