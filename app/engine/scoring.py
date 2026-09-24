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

# Out-of-sample trades for the sample-adequacy component to reach full
# marks. Below it, a score is a claim made on thin evidence.
#
# The precision of an estimated edge scales with the square root of the
# sample, so the component does too. 150 is where a profit factor or a
# win rate starts being worth quoting; the phase gate already refuses
# anything under 30 (phases.MIN_OOS_TRADES), so the realistic range this
# operates over is 30 to 150.
FULL_CONFIDENCE_TRADES = 150

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
    """0-100 composite, six equally weighted parts: profit factor, win
    rate, Sharpe, drawdown control, Phase 4 robustness margin, and how
    much out-of-sample evidence there is behind all five.

    The sixth is the one that was missing. Without it a strategy with 30
    out-of-sample trades and one with 400 scored identically, and since
    the Vault is sorted by score, the thin one could sit at the top of
    the list and be the one someone deploys. Everything else here
    measures how good the result was; this measures how much the result
    is worth believing.
    """
    # A profit factor of 1.0 is breakeven, not half a good result. Dividing
    # by two gave a strategy that loses money 45 marks out of 100 on this
    # component, and paired with the two components a consistent loser
    # scores full marks on - it degrades very little under noise, and 191
    # trades is plenty of evidence - that was enough to grade one GOOD.
    # The ramp now starts where the strategy starts making money.
    pf_component = (
        1.0 if stats_oos.profit_factor == float("inf")
        else _clamp01(stats_oos.profit_factor - 1.0)
    )
    win_rate_component = _clamp01(stats_oos.win_rate)
    sharpe_component = _clamp01(stats_oos.sharpe_ratio / 2.0)
    drawdown_component = _clamp01(1 - stats_oos.max_dd_pct / 50.0)
    worst_degradation = max(robustness.monte_carlo_degradation_pct, robustness.noise_degradation_pct)
    robustness_component = _clamp01(1 - worst_degradation / 25.0)
    sample_component = sample_adequacy(stats_oos.trade_count)

    composite = (
        pf_component
        + win_rate_component
        + sharpe_component
        + drawdown_component
        + robustness_component
        + sample_component
    ) / 6.0
    return round(composite * 100)


def sample_adequacy(trade_count: int) -> float:
    """0-1: how much the rest of the score is worth believing.

    Square root, because that is how the precision of an estimate scales
    with its sample. 30 trades scores 0.45, 75 scores 0.71, 150 and above
    scores 1.0.
    """
    if trade_count <= 0:
        return 0.0
    return _clamp01((trade_count / FULL_CONFIDENCE_TRADES) ** 0.5)


# No number of good-looking trades makes a dozen of them enough. This
# matches the phase gate (phases.MIN_OOS_TRADES) rather than inventing a
# second bar: if a sample is too thin to get into the Vault on its own
# merits, it is too thin to be called GOOD once it is there.
MIN_TRADES_FOR_GOOD = 30


def compute_grade(score: int, trade_count: int | None = None, net_profit_pct: float | None = None) -> str:
    """GOOD needs a good result, enough of it, and a profit.

    The score already carries a sample-adequacy component, but one part
    in six cannot hold back a result that is excellent on every other
    axis: a twelve-trade fluke still scores in the seventies. Weighting
    cannot express "not enough evidence to call this anything" - only a
    floor can.

    The same is true of losing money, and more bluntly. Nothing in the
    composite asked whether the strategy was profitable: robustness
    measures how little a result decays under noise, and a strategy that
    is reliably bad decays hardly at all. #6B8DCA came back from REASSESS
    at -5.3% over its out-of-sample window, -10.2% a year, and was graded
    GOOD at 64 on the strength of 100% robustness and 191 trades. No
    weighting fixes that; only a floor does.

    `trade_count` and `net_profit_pct` are optional so a caller with only
    a score still works; it grades on the score alone, which is the old
    behaviour.
    """
    if score < GOOD_GRADE_THRESHOLD:
        return "WEAK"
    if trade_count is not None and trade_count < MIN_TRADES_FOR_GOOD:
        return "WEAK"
    if net_profit_pct is not None and net_profit_pct <= 0:
        return "WEAK"
    return "GOOD"
