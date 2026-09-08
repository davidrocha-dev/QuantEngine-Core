"""Vault Score / Grade / Archetype — REQ-VAULT-1.1/1.2, REQ-MAP-1.3.

Neither `requirements.md` nor `design.md` specifies the exact Score
formula or archetype-classification rule (only that GRADE = GOOD when
Score >= 57, and that archetypes are MULTI_SIGNAL / MEAN_REVERTER /
PATTERN_SPOTTER). Per the project owner's direction on REQ-ENGINE-1.2.5
("implement a reasonable, standard, well-documented version ... don't
block the rest of the architecture on this"), this module applies that
same approach here: a documented, provisional formula meant to be
reviewed/tuned rather than treated as fixed.

Archetype labeling here is a lightweight heuristic off the genome's Signal
gene family, used only until Phase 9's UMAP-based cluster auto-labeling
(REQ-MAP-1.3) exists as the richer, population-aware alternative.
"""

from __future__ import annotations

from app.engine.metrics import TradeStats
from app.engine.monte_carlo import RobustnessResult
from app.genome.genome import Genome

GOOD_GRADE_THRESHOLD = 57

_MEAN_REVERTER_SIGNALS = {"FADE", "CCI", "STOCH", "WILLIAMS"}
_PATTERN_SPOTTER_SIGNALS = {"ENGULFING", "THREE SOLDIERS", "WICK REJECTION", "INSIDE BREAKOUT", "LIQSWEEP", "FIB", "DOJI"}
# Everything else (BREAKOUT, MACD, MOM BREAK, ORB, VWAP, ATR RUN) defaults
# to MULTI_SIGNAL, matching the archetype's "combines trend + confirmation
# conditions" framing once a Bias/Filter is layered on top of the signal.


def classify_archetype(genome: Genome) -> str:
    if genome.signal.name in _MEAN_REVERTER_SIGNALS:
        return "MEAN_REVERTER"
    if genome.signal.name in _PATTERN_SPOTTER_SIGNALS:
        return "PATTERN_SPOTTER"
    return "MULTI_SIGNAL"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_score(stats_oos: TradeStats, robustness: RobustnessResult) -> int:
    """0-100 composite: profit factor, win rate, Sharpe, drawdown control,
    and Phase 4 robustness margin (headroom below the 25% degradation
    discard threshold), each weighted 20%."""
    pf_component = _clamp01(stats_oos.profit_factor / 2.0) if stats_oos.profit_factor != float("inf") else 1.0
    win_rate_component = _clamp01(stats_oos.win_rate)
    sharpe_component = _clamp01(stats_oos.sharpe_ratio / 2.0)
    drawdown_component = _clamp01(1 - stats_oos.max_dd_pct / 50.0)
    worst_degradation = max(robustness.monte_carlo_degradation_pct, robustness.noise_degradation_pct)
    robustness_component = _clamp01(1 - worst_degradation / 25.0)

    composite = (
        pf_component + win_rate_component + sharpe_component + drawdown_component + robustness_component
    ) / 5.0
    return round(composite * 100)


def compute_grade(score: int) -> str:
    return "GOOD" if score >= GOOD_GRADE_THRESHOLD else "WEAK"
