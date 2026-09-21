"""Phase 0 screening gate and F1/F2 fitness formulas — REQ-ENGINE-3.2/3.3/3.4."""

from __future__ import annotations

from app.engine.backtest import BacktestResult
from app.engine.metrics import TradeStats

PHASE_0_MIN_PROFIT_FACTOR = 1.1
PHASE_0_MIN_TRADES = 100


def passes_phase_0_screening(stats: TradeStats, result: BacktestResult) -> tuple[bool, str | None]:
    """REQ-ENGINE-3.2: discard PF < 1.1, fewer than 100 trades, or a margin breach."""
    if result.margin_breach:
        return False, "margin breach"
    if stats.trade_count < PHASE_0_MIN_TRADES:
        return False, f"only {stats.trade_count} trades (min {PHASE_0_MIN_TRADES})"
    if stats.profit_factor < PHASE_0_MIN_PROFIT_FACTOR:
        return False, f"profit factor {stats.profit_factor:.2f} below {PHASE_0_MIN_PROFIT_FACTOR}"
    return True, None


def fitness_f1_stability(stats: TradeStats) -> float:
    """REQ-ENGINE-3.3 (Phase 1, Tribe A): F1 = (Net Profit / Max DD%) × Win Rate × Sharpe Ratio."""
    if stats.max_dd_pct <= 0:
        return 0.0
    value = (stats.net_profit_pct / stats.max_dd_pct) * stats.win_rate * stats.sharpe_ratio
    if stats.net_profit_pct <= 0 or stats.sharpe_ratio <= 0:
        # A losing strategy has both a negative Net Profit and a negative
        # Sharpe — their product is positive and would outrank winners.
        return -abs(value)
    return value


def fitness_f2_expansion(stats: TradeStats) -> float:
    """REQ-ENGINE-3.4 (Phase 2, Tribe B): F2 = CAGR% × Profit Factor × Payoff Ratio.

    CAGR is None on a window too short to annualise honestly (see
    metrics.cagr_pct). The total return stands in for it there: over a few
    weeks the raw return *is* the growth figure, and substituting zero
    would flatten every candidate in the tribe to the same score and turn
    the ranking into a coin toss.
    """
    profit_factor = stats.profit_factor if stats.profit_factor != float("inf") else 0.0
    growth = stats.cagr_pct if stats.cagr_pct is not None else stats.net_profit_pct
    return growth * profit_factor * stats.payoff_ratio
