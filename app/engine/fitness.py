"""Phase 0 screening gate and F1/F2 fitness formulas — REQ-ENGINE-3.2/3.3/3.4."""

from __future__ import annotations

from app.engine.backtest import BacktestResult
from app.engine.metrics import TradeStats

PHASE_0_MIN_PROFIT_FACTOR = 1.1
PHASE_0_MIN_TRADES = 100

# Profit factor in F2 is capped here. A candidate with no losing trade has
# an infinite one, which has to sort somewhere finite - at the top, as
# compute_score also gives it full marks - and beyond 3 a profit factor is
# more often a thin sample than a better strategy. A ranking cap, not a
# measurement.
FITNESS_MAX_PROFIT_FACTOR = 3.0


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
    """Phase 1, Tribe A: F1 = (Net Profit % / Max DD %) x SQN.

    REQ-ENGINE-3.3 wrote this as (Net Profit / Max DD%) x Win Rate x Sharpe.
    Win rate is the problem term: it pays for winning often, not for
    winning more than is lost, so the tribe bred towards small targets
    behind wide stops - a 1R stop against a 0.38R target needs a 72.4% win
    rate to break even, and a few points of spread or a slightly different
    fill in MetaTrader is the whole margin. SQN (metrics.system_quality_number)
    measures the edge per trade and the evidence behind it, which is what
    Win Rate x Sharpe was reaching for.
    """
    if stats.max_dd_pct <= 0:
        return 0.0
    value = (stats.net_profit_pct / stats.max_dd_pct) * stats.sqn
    if stats.net_profit_pct <= 0 or stats.sqn <= 0:
        # Two negatives multiply out positive; a loser must rank below winners.
        return -abs(value)
    return value


def fitness_f2_expansion(stats: TradeStats) -> float:
    """Phase 2, Tribe B: F2 = growth % x min(Profit Factor, 3).

    REQ-ENGINE-3.4 also multiplied by the payoff ratio. Growth, profit
    factor and payoff all rise together when a few outsized winners carry
    a thin sample, so their product rewarded exactly the strategies whose
    result depends most on trades that may not come again. The profit
    factor is capped for the same reason: past 3 it is usually a small
    sample talking.

    Growth is CAGR, or the total return where the window is too short to
    annualise honestly (metrics.cagr_pct returns None there).
    """
    profit_factor = min(stats.profit_factor, FITNESS_MAX_PROFIT_FACTOR)
    growth = stats.cagr_pct if stats.cagr_pct is not None else stats.net_profit_pct
    return growth * profit_factor
