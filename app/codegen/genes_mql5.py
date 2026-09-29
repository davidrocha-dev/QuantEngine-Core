"""MQL5 expressions for each of the 46 genes — the export side of the gene
catalog in `app/genome/genes.py`.

Every emitter mirrors the Python gene it is named after, reading the same
bars: the generated EA evaluates on *closed* bars (shift 1 is the last
closed bar, shift 2 the one before), which is what the backtester's
`df.shift(1)` means. Where a gene needs an indicator, the emitter declares
the handle it needs and the template creates it once in OnInit.

A gene with no emitter here cannot be exported. That is deliberate: an EA
that silently trades differently from its backtest is worse than a refused
export, so `mql5.py` raises with the gene's name instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GeneCode:
    """MQL5 for one gene.

    `long_expr` / `short_expr` are boolean expressions for SIGNAL and BIAS
    genes; `veto_expr` is the boolean a FILTER blocks entry on. `handles`
    maps a variable name to the indicator constructor that fills it.
    """

    handles: dict[str, str] = field(default_factory=dict)
    long_expr: str = "false"
    short_expr: str = "false"
    veto_expr: str = "false"


def _i(name: str, index: int) -> str:
    return f"h_{name}_{index}"


# ---------------------------------------------------------------------------
# SIGNAL layer (16)
# ---------------------------------------------------------------------------

def _signal(name: str, params: dict, index: int) -> GeneCode:
    p = params
    if name == "ATR RUN":
        atr = _i("atr", index)
        lookback, mult = int(p.get("lookback", 10)), p.get("atr_mult", 1.5)
        move = f"(Cl(1) - Cl(1 + {lookback}))"
        threshold = f"({mult} * Val({atr}, 0, 1))"
        return GeneCode({atr: "iATR(_Symbol, PERIOD_CURRENT, 14)"}, f"{move} > {threshold}", f"{move} < -{threshold}")

    if name == "BREAKOUT":
        period = int(p.get("period", 20))
        return GeneCode(
            long_expr=f"Cl(1) > HighestHigh({period}, 2)",
            short_expr=f"Cl(1) < LowestLow({period}, 2)",
        )

    if name == "CCI":
        cci, period = _i("cci", index), int(p.get("period", 20))
        return GeneCode(
            {cci: f"iCCI(_Symbol, PERIOD_CURRENT, {period}, PRICE_TYPICAL)"},
            f"Val({cci}, 0, 1) > 100 && Val({cci}, 0, 2) <= 100",
            f"Val({cci}, 0, 1) < -100 && Val({cci}, 0, 2) >= -100",
        )

    if name == "ENGULFING":
        return GeneCode(long_expr="BullishEngulfing(1)", short_expr="BearishEngulfing(1)")

    if name == "FADE":
        rsi, period = _i("rsi", index), int(p.get("period", 14))
        oversold, overbought = p.get("oversold", 30), p.get("overbought", 70)
        return GeneCode(
            {rsi: f"iRSI(_Symbol, PERIOD_CURRENT, {period}, PRICE_CLOSE)"},
            f"Val({rsi}, 0, 1) < {oversold}",
            f"Val({rsi}, 0, 1) > {overbought}",
        )

    if name == "FIB":
        lookback, tolerance = int(p.get("lookback", 50)), p.get("tolerance", 0.1)
        swing_high, swing_low = f"HighestHigh({lookback}, 1)", f"LowestLow({lookback}, 1)"
        rng = f"({swing_high} - {swing_low})"
        level = f"({swing_high} - 0.618 * {rng})"
        near = f"(MathAbs(Cl(1) - {level}) <= {tolerance} * {rng})"
        uptrend = f"(Cl(1) > Cl(1 + {lookback}))"
        return GeneCode(
            long_expr=f"{near} && {uptrend} && Cl(1) > Cl(2)",
            short_expr=f"{near} && !{uptrend} && Cl(1) < Cl(2)",
        )

    if name == "INSIDE BREAKOUT":
        inside_prev = "(Hi(2) < Hi(3) && Lo(2) > Lo(3))"
        return GeneCode(
            long_expr=f"{inside_prev} && Cl(1) > Hi(2)",
            short_expr=f"{inside_prev} && Cl(1) < Lo(2)",
        )

    if name == "LIQSWEEP":
        period = int(p.get("period", 20))
        prior_high, prior_low = f"HighestHigh({period}, 2)", f"LowestLow({period}, 2)"
        return GeneCode(
            long_expr=f"Lo(1) < {prior_low} && Cl(1) > {prior_low}",
            short_expr=f"Hi(1) > {prior_high} && Cl(1) < {prior_high}",
        )

    if name == "MACD":
        macd = _i("macd", index)
        fast, slow, signal = int(p.get("fast", 12)), int(p.get("slow", 26)), int(p.get("signal", 9))
        return GeneCode(
            {macd: f"iMACD(_Symbol, PERIOD_CURRENT, {fast}, {slow}, {signal}, PRICE_CLOSE)"},
            f"Val({macd}, 0, 1) > Val({macd}, 1, 1) && Val({macd}, 0, 2) <= Val({macd}, 1, 2)",
            f"Val({macd}, 0, 1) < Val({macd}, 1, 1) && Val({macd}, 0, 2) >= Val({macd}, 1, 2)",
        )

    if name == "MOM BREAK":
        period, threshold = int(p.get("period", 10)), p.get("threshold", 0.003)
        roc = f"(Cl(1) / Cl(1 + {period}) - 1.0)"
        return GeneCode(long_expr=f"{roc} > {threshold}", short_expr=f"{roc} < -{threshold}")

    if name == "ORB":
        range_bars = int(p.get("range_bars", 4))
        return GeneCode(
            long_expr=f"BarOfDay(1) >= {range_bars} && Cl(1) > OpeningRangeHigh({range_bars})",
            short_expr=f"BarOfDay(1) >= {range_bars} && Cl(1) < OpeningRangeLow({range_bars})",
        )

    if name == "STOCH":
        stoch = _i("stoch", index)
        k_period, d_period = int(p.get("k_period", 14)), int(p.get("d_period", 3))
        return GeneCode(
            # slowing=1: the Python gene has no slowing, and a slowing of 3
            # here left one crossover in eleven in common with the backtest.
            {stoch: f"iStochastic(_Symbol, PERIOD_CURRENT, {k_period}, {d_period}, 1, MODE_SMA, STO_LOWHIGH)"},
            f"Val({stoch}, 0, 1) > Val({stoch}, 1, 1) && Val({stoch}, 0, 2) <= Val({stoch}, 1, 2) && Val({stoch}, 0, 1) < 20",
            f"Val({stoch}, 0, 1) < Val({stoch}, 1, 1) && Val({stoch}, 0, 2) >= Val({stoch}, 1, 2) && Val({stoch}, 0, 1) > 80",
        )

    if name == "THREE SOLDIERS":
        bullish = "(Cl(1) > Op(1) && Cl(2) > Op(2) && Cl(3) > Op(3))"
        rising = "(Cl(1) > Cl(2) && Cl(2) > Cl(3))"
        bearish = "(Cl(1) < Op(1) && Cl(2) < Op(2) && Cl(3) < Op(3))"
        falling = "(Cl(1) < Cl(2) && Cl(2) < Cl(3))"
        return GeneCode(long_expr=f"{bullish} && {rising}", short_expr=f"{bearish} && {falling}")

    if name == "VWAP":
        return GeneCode(
            long_expr="Cl(1) > SessionVwap(1) && Cl(2) <= SessionVwap(2)",
            short_expr="Cl(1) < SessionVwap(1) && Cl(2) >= SessionVwap(2)",
        )

    if name == "WICK REJECTION":
        ratio = p.get("wick_ratio", 2.0)
        return GeneCode(long_expr=f"LowerWickRatio(1) >= {ratio}", short_expr=f"UpperWickRatio(1) >= {ratio}")

    if name == "WILLIAMS":
        wpr, period = _i("wpr", index), int(p.get("period", 14))
        return GeneCode(
            {wpr: f"iWPR(_Symbol, PERIOD_CURRENT, {period})"},
            f"Val({wpr}, 0, 1) > -80 && Val({wpr}, 0, 2) <= -80",
            f"Val({wpr}, 0, 1) < -20 && Val({wpr}, 0, 2) >= -20",
        )

    raise KeyError(name)


# ---------------------------------------------------------------------------
# BIAS layer (11)
# ---------------------------------------------------------------------------

def _bias(name: str, params: dict, index: int) -> GeneCode:
    p = params
    if name == "ADX":
        adx, period, threshold = _i("adx", index), int(p.get("period", 14)), p.get("threshold", 20)
        strong = f"Val({adx}, 0, 1) > {threshold}"
        return GeneCode(
            # iADXWilder: the Python side is Wilder; plain iADX is an EMA.
            {adx: f"iADXWilder(_Symbol, PERIOD_CURRENT, {period})"},
            f"{strong} && Val({adx}, 1, 1) > Val({adx}, 2, 1)",
            f"{strong} && Val({adx}, 1, 1) < Val({adx}, 2, 1)",
        )

    if name == "CHANDELIER":
        atr, period, mult = _i("atr", index), int(p.get("period", 22)), p.get("atr_mult", 3.0)
        band = f"({mult} * Val({atr}, 0, 1))"
        return GeneCode(
            {atr: f"iATR(_Symbol, PERIOD_CURRENT, {period})"},
            f"Cl(1) > (HighestHigh({period}, 1) - {band})",
            f"Cl(1) < (LowestLow({period}, 1) + {band})",
        )

    if name == "DAILY MID":
        return GeneCode(long_expr="Cl(1) > PrevDayMid()", short_expr="Cl(1) < PrevDayMid()")

    if name == "DONCHIAN MID":
        period = int(p.get("period", 20))
        mid = f"((HighestHigh({period}, 1) + LowestLow({period}, 1)) / 2.0)"
        return GeneCode(long_expr=f"Cl(1) > {mid}", short_expr=f"Cl(1) < {mid}")

    if name == "HTF":
        ma, period = _i("htf_ma", index), int(p.get("period", 20))
        # The higher-timeframe bar must be closed — shift 1 on that series —
        # which is the same rule the Python gene's shift(1) enforces.
        return GeneCode(
            {ma: f"iMA(_Symbol, HigherTimeframe(), {period}, 0, MODE_SMA, PRICE_CLOSE)"},
            f"HtfClose(1) > Val({ma}, 0, 1)",
            f"HtfClose(1) < Val({ma}, 0, 1)",
        )

    if name == "MARKET STRUCTURE":
        lookback = int(p.get("lookback", 10))
        return GeneCode(
            long_expr=f"Hi(1) > Hi(1 + {lookback}) && Lo(1) > Lo(1 + {lookback})",
            short_expr=f"Hi(1) < Hi(1 + {lookback}) && Lo(1) < Lo(1 + {lookback})",
        )

    if name == "MOMENTUM":
        period = int(p.get("period", 20))
        roc = f"(Cl(1) / Cl(1 + {period}) - 1.0)"
        return GeneCode(long_expr=f"{roc} > 0", short_expr=f"{roc} < 0")

    if name == "PSAR":
        sar = _i("sar", index)
        step, max_step = p.get("step", 0.02), p.get("max_step", 0.2)
        return GeneCode(
            {sar: f"iSAR(_Symbol, PERIOD_CURRENT, {step}, {max_step})"},
            f"Cl(1) > Val({sar}, 0, 1)",
            f"Cl(1) < Val({sar}, 0, 1)",
        )

    if name == "RSI":
        rsi, period = _i("rsi_bias", index), int(p.get("period", 14))
        return GeneCode(
            {rsi: f"iRSI(_Symbol, PERIOD_CURRENT, {period}, PRICE_CLOSE)"},
            f"Val({rsi}, 0, 1) > 50",
            f"Val({rsi}, 0, 1) < 50",
        )

    if name == "SMA":
        ma, period = _i("sma_bias", index), int(p.get("period", 50))
        return GeneCode(
            {ma: f"iMA(_Symbol, PERIOD_CURRENT, {period}, 0, MODE_SMA, PRICE_CLOSE)"},
            f"Cl(1) > Val({ma}, 0, 1)",
            f"Cl(1) < Val({ma}, 0, 1)",
        )

    if name == "TRAILING":
        period = int(p.get("period", 14))
        return GeneCode(
            long_expr=f"Cl(1) > HighestHigh({period}, 2)",
            short_expr=f"Cl(1) < LowestLow({period}, 2)",
        )

    raise KeyError(name)


# ---------------------------------------------------------------------------
# FILTER layer (12) — veto_expr true blocks entry
# ---------------------------------------------------------------------------

def _filter(name: str, params: dict, index: int) -> GeneCode:
    p = params
    if name == "ADR EXHAUST":
        period, exhaust = int(p.get("period", 14)), p.get("exhaust_pct", 0.8)
        return GeneCode(veto_expr=f"TodayRange() >= {exhaust} * AverageDailyRange({period})")

    if name == "CCI":
        cci, period, extreme = _i("cci_f", index), int(p.get("period", 20)), p.get("extreme", 200)
        return GeneCode(
            {cci: f"iCCI(_Symbol, PERIOD_CURRENT, {period}, PRICE_TYPICAL)"},
            veto_expr=f"MathAbs(Val({cci}, 0, 1)) > {extreme}",
        )

    if name == "CONSEC":
        # The backtester has never applied this gene - its Python side
        # vetoes nothing - so every strategy carrying it was measured
        # without it, and that is what the EA must trade. It used to veto
        # on the EA's own losing streak, which never resets once trading
        # stops: after N losses in a row the EA was switched off for good.
        # Campaigns no longer draw it (main._gene_pool_for_campaign).
        return GeneCode(veto_expr="false")

    if name == "DOJI":
        return GeneCode(veto_expr=f"IsDoji(1, {p.get('body_ratio', 0.1)})")

    if name == "EVENT":
        # REQ-ENGINE-2 — the EA has no calendar feed of its own; exporting a
        # genome that relies on it would trade a filter that is not there.
        raise KeyError("EVENT")

    if name == "KELTNER":
        ma, atr = _i("kelt_ma", index), _i("kelt_atr", index)
        period, mult = int(p.get("period", 20)), p.get("atr_mult", 2.0)
        band = f"({mult} * Val({atr}, 0, 1))"
        return GeneCode(
            {
                ma: f"iMA(_Symbol, PERIOD_CURRENT, {period}, 0, MODE_EMA, PRICE_CLOSE)",
                atr: f"iATR(_Symbol, PERIOD_CURRENT, {period})",
            },
            veto_expr=f"(Cl(1) > Val({ma}, 0, 1) + {band} || Cl(1) < Val({ma}, 0, 1) - {band})",
        )

    if name == "NO FRI OPEN":
        return GeneCode(veto_expr=f"IsFridayAfter({int(p.get('cutoff_hour', 18))})")

    if name == "RECEDING":
        atr, lookback = _i("rec_atr", index), int(p.get("lookback", 5))
        return GeneCode(
            {atr: "iATR(_Symbol, PERIOD_CURRENT, 14)"},
            veto_expr=f"AtrDeclining({atr}, {lookback})",
        )

    if name == "REGIME":
        adx, period, threshold = _i("adx_f", index), int(p.get("period", 14)), p.get("threshold", 20)
        require = p.get("require", "trend")
        comparison = "<" if require == "trend" else ">="
        return GeneCode(
            {adx: f"iADXWilder(_Symbol, PERIOD_CURRENT, {period})"},
            veto_expr=f"Val({adx}, 0, 1) {comparison} {threshold}",
        )

    if name == "RSI":
        rsi, period = _i("rsi_f", index), int(p.get("period", 14))
        return GeneCode(
            {rsi: f"iRSI(_Symbol, PERIOD_CURRENT, {period}, PRICE_CLOSE)"},
            veto_expr=f"(Val({rsi}, 0, 1) > 80 || Val({rsi}, 0, 1) < 20)",
        )

    if name == "SMA":
        ma, atr = _i("sma_f", index), _i("sma_f_atr", index)
        period, mult = int(p.get("period", 50)), p.get("atr_mult", 3.0)
        return GeneCode(
            {
                ma: f"iMA(_Symbol, PERIOD_CURRENT, {period}, 0, MODE_SMA, PRICE_CLOSE)",
                atr: "iATR(_Symbol, PERIOD_CURRENT, 14)",
            },
            veto_expr=f"MathAbs(Cl(1) - Val({ma}, 0, 1)) > {mult} * Val({atr}, 0, 1)",
        )

    if name == "VOLATILITY":
        atr = _i("vol_atr", index)
        period = int(p.get("period", 14))
        min_mult, max_mult = p.get("min_mult", 0.5), p.get("max_mult", 2.0)
        baseline = f"AverageAtr({atr}, {period * 5})"
        return GeneCode(
            {atr: f"iATR(_Symbol, PERIOD_CURRENT, {period})"},
            veto_expr=f"(Val({atr}, 0, 1) < {min_mult} * {baseline} || Val({atr}, 0, 1) > {max_mult} * {baseline})",
        )

    raise KeyError(name)


EMITTERS = {"signal": _signal, "bias": _bias, "filter": _filter}


def emit(layer: str, name: str, params: dict, index: int = 0) -> GeneCode:
    """MQL5 for one gene, or KeyError(name) when it cannot be exported."""
    return EMITTERS[layer](name, params, index)
