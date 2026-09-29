"""MQL5 code generation for a Vault strategy — design.md B.1 (Jinja2).

Turns a stored genome back into an Expert Advisor that trades the rules the
backtester measured: same closed-bar evaluation, same ATR stop/target
multiples, same single-position-at-a-time risk sizing.

Two deliberate refusals, because a plausible-but-different EA is worse than
no EA:

  * a gene with no MQL5 emitter (`app/codegen/genes_mql5.py`) raises
    `UnsupportedGeneError` naming it, rather than being silently dropped;
  * the EVENT (news) gene cannot be exported at all — the EA has no
    calendar feed, so the exported strategy would trade without a filter
    its backtest had.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.codegen.genes_mql5 import GeneCode, emit
from app.data.broker_clock import DEFAULT_CLOCK, BrokerClock
from app.engine.backtest import MAX_HOLD_BARS, PENDING_ORDER_WINDOW_BARS
from app.genome.genome import Genome

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

# REQ-BREED-2.1's context-timeframe map, in MQL5 constants.
MQL5_TIMEFRAME = {
    "M1": "PERIOD_M1", "M5": "PERIOD_M5", "M15": "PERIOD_M15", "M30": "PERIOD_M30",
    "H1": "PERIOD_H1", "H2": "PERIOD_H2", "H4": "PERIOD_H4", "D1": "PERIOD_D1",
}
HIGHER_TIMEFRAME = {"M15": "H1", "M30": "H1", "H1": "H4", "H2": "D1", "H4": "D1", "D1": "D1", "M5": "H1", "M1": "M15"}

# `higher_timeframe` for a strategy whose HIGHER TF context is not known:
# fall back to the automatic mapping, which is what every export did before
# the context was recorded per strategy.
AUTO_HIGHER_TIMEFRAME = "AUTO"


class UnsupportedGeneError(RuntimeError):
    """A gene in this genome has no faithful MQL5 equivalent yet."""


@dataclass
class GeneratedEa:
    filename: str
    source: str


def _magic_number(hash_id: str) -> int:
    """Stable per-strategy magic number, so an EA only ever manages its own
    positions even when several are attached to the same terminal."""
    digest = hashlib.sha256(hash_id.encode()).hexdigest()
    return 100_000_000 + int(digest[:8], 16) % 899_999_999


def _safe_name(hash_id: str) -> str:
    return "QE_" + re.sub(r"[^A-Za-z0-9]", "", hash_id).upper()


def build_context(
    genome: Genome, hash_id: str, symbol: str, timeframe: str, risk_pct: float, max_spread_points: float,
    higher_timeframe: str | None = AUTO_HIGHER_TIMEFRAME,
    broker_clock: BrokerClock = DEFAULT_CLOCK,
) -> dict:
    """`higher_timeframe` is the context the strategy was measured with:
    a timeframe, None when HIGHER TF was off, or AUTO_HIGHER_TIMEFRAME
    when nobody knows. With HIGHER TF off the backtest's HTF bias is
    neutral on every bar, so the EA's is too - it used to read a real
    higher-timeframe average and filter trades the backtest had taken."""
    handles: dict[str, str] = {}
    if higher_timeframe == AUTO_HIGHER_TIMEFRAME:
        higher_timeframe = HIGHER_TIMEFRAME.get(timeframe, "H4")
    neutral_htf_bias = genome.bias is not None and genome.bias.name == "HTF" and higher_timeframe is None

    def collect(code: GeneCode) -> GeneCode:
        handles.update(code.handles)
        return code

    try:
        signal = collect(emit("signal", genome.signal.name, genome.signal.params, 0))
        bias = (
            collect(emit("bias", genome.bias.name, genome.bias.params, 1))
            if genome.bias and not neutral_htf_bias
            else None
        )
        filters = [collect(emit("filter", f.name, f.params, 10 + i)) for i, f in enumerate(genome.filters)]
    except KeyError as exc:
        gene = exc.args[0] if exc.args else "unknown"
        raise UnsupportedGeneError(
            f"{gene} has no MQL5 equivalent yet, so this strategy cannot be exported without changing how it trades"
        ) from exc

    execution = genome.execution.name
    execution_params = genome.execution.params
    management = genome.management.name if genome.management else None

    return {
        "hash_id": hash_id,
        "genome_summary": _summary(genome),
        "symbol": symbol,
        "timeframe": timeframe,
        "magic": _magic_number(hash_id),
        "risk_pct": round(risk_pct * 100, 2),
        "max_spread_points": max_spread_points,
        "handles": handles,
        "signal_long": signal.long_expr,
        "signal_short": signal.short_expr,
        "has_bias": bias is not None,
        "bias_long": bias.long_expr if bias else "false",
        "bias_short": bias.short_expr if bias else "false",
        "vetoes": [f.veto_expr for f in filters],
        "execution": execution,
        "limit_offset_atr": execution_params.get(
            "offset_atr", (execution_params.get("offset_atr_1", 0.5) + execution_params.get("offset_atr_2", 1.0)) / 2
        ),
        "stop_offset_atr": execution_params.get("offset_atr", 0.3),
        # MARKET2 waits this many bars after the signal before it buys.
        # The backtester fills at the open of signal_idx + confirm_bars;
        # the EA used to fill on the very next bar whatever this said.
        "confirm_bars": int(execution_params.get("confirm_bars", 2)),
        "pending_bars": PENDING_ORDER_WINDOW_BARS,
        "max_hold_bars": MAX_HOLD_BARS,
        "management": management,
        "management_params": genome.management.params if genome.management else {},
        "stop_atr_mult": genome.stop_atr_mult,
        "target_atr_mult": genome.target_atr_mult,
        "higher_timeframe": MQL5_TIMEFRAME[higher_timeframe] if higher_timeframe else "PERIOD_CURRENT",
        "server_time_mode": broker_clock.mode,
        "server_offset_hours": float(broker_clock.offset_hours),
        "server_time_description": broker_clock.describe(),
    }


def _summary(genome: Genome) -> str:
    parts = [genome.signal.name]
    if genome.bias:
        parts.append(f"bias {genome.bias.name}")
    if genome.filters:
        parts.append("filters " + "/".join(f.name for f in genome.filters))
    parts.append(genome.execution.name)
    if genome.management:
        parts.append(genome.management.name)
    return " | ".join(parts)


def generate_ea(
    genome: Genome,
    hash_id: str,
    symbol: str,
    timeframe: str,
    risk_pct: float = 0.01,
    max_spread_points: float = 0.0,
    higher_timeframe: str | None = AUTO_HIGHER_TIMEFRAME,
    broker_clock: BrokerClock = DEFAULT_CLOCK,
) -> GeneratedEa:
    environment = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR), undefined=StrictUndefined, trim_blocks=False, lstrip_blocks=False
    )
    template = environment.get_template("strategy_ea.mq5.j2")
    context = build_context(genome, hash_id, symbol, timeframe, risk_pct, max_spread_points, higher_timeframe, broker_clock)
    return GeneratedEa(filename=f"{_safe_name(hash_id)}.mq5", source=template.render(**context))
