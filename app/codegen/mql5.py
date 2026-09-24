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
from app.genome.genome import Genome

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

# REQ-BREED-2.1's context-timeframe map, in MQL5 constants.
MQL5_TIMEFRAME = {
    "M1": "PERIOD_M1", "M5": "PERIOD_M5", "M15": "PERIOD_M15", "M30": "PERIOD_M30",
    "H1": "PERIOD_H1", "H2": "PERIOD_H2", "H4": "PERIOD_H4", "D1": "PERIOD_D1",
}
HIGHER_TIMEFRAME = {"M15": "H1", "M30": "H1", "H1": "H4", "H2": "D1", "H4": "D1", "D1": "D1", "M5": "H1", "M1": "M15"}

# LIMIT/STOP orders expire unfilled after this many bars in the backtester
# (PENDING_ORDER_WINDOW_BARS); the EA expresses the same window in seconds.
PENDING_ORDER_WINDOW_BARS = 5
TIMEFRAME_SECONDS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H2": 7200, "H4": 14400, "D1": 86400,
}


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


def build_context(genome: Genome, hash_id: str, symbol: str, timeframe: str, risk_pct: float, max_spread_points: float) -> dict:
    handles: dict[str, str] = {}

    def collect(code: GeneCode) -> GeneCode:
        handles.update(code.handles)
        return code

    try:
        signal = collect(emit("signal", genome.signal.name, genome.signal.params, 0))
        bias = collect(emit("bias", genome.bias.name, genome.bias.params, 1)) if genome.bias else None
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
        "pending_expiry_seconds": PENDING_ORDER_WINDOW_BARS * TIMEFRAME_SECONDS.get(timeframe, 3600),
        "management": management,
        "management_params": genome.management.params if genome.management else {},
        "stop_atr_mult": genome.stop_atr_mult,
        "target_atr_mult": genome.target_atr_mult,
        "higher_timeframe": MQL5_TIMEFRAME[HIGHER_TIMEFRAME.get(timeframe, "H4")],
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
) -> GeneratedEa:
    environment = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR), undefined=StrictUndefined, trim_blocks=False, lstrip_blocks=False
    )
    template = environment.get_template("strategy_ea.mq5.j2")
    context = build_context(genome, hash_id, symbol, timeframe, risk_pct, max_spread_points)
    return GeneratedEa(filename=f"{_safe_name(hash_id)}.mq5", source=template.render(**context))
