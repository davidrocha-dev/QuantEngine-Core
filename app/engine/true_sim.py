"""TRUE SIM — the portfolio on one account, with guardrails — REQ-PORT-2.

The HEALTH tab's combined curve adds up strategies that were each
simulated alone. That overstates what a real account would have done,
because it lets two members hold the same symbol at once and lets total
open risk reach whatever the members happen to want.

TRUE SIM replays every member's trades onto a single simulated account in
chronological order and applies the two guardrails that a live account
imposes anyway:

  * **Symbol Lock** — one position per symbol at a time. A second entry on
    a symbol already held is blocked, not sized down.
  * **Max aggregate risk** — total open risk is capped. An entry that
    would breach the cap is blocked.

Blocked entries are counted, not hidden: the gap between the combined
curve and this one is exactly what the guardrails cost, and that number is
the point of running it.

Equity is marked to market between entry and exit (each trade's unrealised
R, scaled by the size this account gave it), so the drawdown reported is
the floating one, which is what a margin call responds to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from app.engine.backtest import simulate_trades
from app.genome.gene_pool import GenePoolConfig
from app.genome.genome import Genome
from app.models import Strategy

FIXED, COMPOUNDING = "FIXED", "COMPOUNDING"


@dataclass
class BlockedEntry:
    timestamp: str
    strategy_id: str
    symbol: str
    reason: str


@dataclass
class TrueSimResult:
    strategy_count: int
    sizing: str
    initial_balance: float
    final_balance: float
    return_pct: float
    max_drawdown_pct: float
    peak_simultaneous_risk_pct: float
    peak_risk_at: str | None
    trades_taken: int
    trades_blocked: int
    blocked_by_symbol_lock: int
    blocked_by_aggregate_risk: int
    equity: list[dict] = field(default_factory=list)
    blocked_samples: list[BlockedEntry] = field(default_factory=list)
    computed_at: str = ""

    def to_dict(self) -> dict:
        data = {k: v for k, v in vars(self).items() if k != "blocked_samples"}
        data["blocked_samples"] = [vars(b) for b in self.blocked_samples]
        return data


@dataclass
class _Candidate:
    strategy_id: str
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    r_multiple: float
    unrealised: pd.Series  # per-bar R while the trade is open, indexed by time


def _member_candidates(strategy: Strategy, df: pd.DataFrame, gene_pool: GenePoolConfig, config) -> list[_Candidate]:
    """Every trade this member would have taken, with the R path it followed."""
    genome = Genome.from_dict(__import__("json").loads(strategy.genome_json))
    result = simulate_trades(
        df, genome, gene_pool,
        initial_balance=config.initial_balance,
        risk_pct=config.risk_pct,
        point_size=config.point_size,
        cost_points=config.cost_points,
        commission_per_lot=config.commission_per_lot,
        contract_size=config.contract_size,
        lot_step=getattr(config, "lot_step", 0.0),
        min_lot=getattr(config, "min_lot", 0.0),
        max_lot=getattr(config, "max_lot", 0.0),
        quote_to_account=getattr(config, "quote_to_account", None),
    )

    timestamps = pd.DatetimeIndex(df["timestamp"])
    closes = df["close"].to_numpy()
    candidates: list[_Candidate] = []

    for trade in result.trades:
        stop_distance = abs(trade.entry_price - trade.stop_price)
        if stop_distance <= 0 or trade.exit_idx <= trade.entry_idx:
            continue
        window = slice(trade.entry_idx, trade.exit_idx + 1)
        unrealised = (closes[window] - trade.entry_price) * trade.direction / stop_distance
        candidates.append(
            _Candidate(
                strategy_id=strategy.id,
                symbol=strategy.asset,
                entry_time=timestamps[trade.entry_idx],
                exit_time=timestamps[trade.exit_idx],
                r_multiple=trade.r_multiple,
                unrealised=pd.Series(unrealised, index=timestamps[window]),
            )
        )
    return candidates


def run_true_sim(
    strategies: list[Strategy],
    bars_by_market: dict[tuple[str, str], pd.DataFrame],
    gene_pool: GenePoolConfig,
    config,
    sizing: str = COMPOUNDING,
    max_aggregate_risk_pct: float = 5.0,
) -> TrueSimResult:
    """Replays every member onto one account. `bars_by_market` maps
    (symbol, timeframe) to the bars that market was bred on."""
    candidates: list[_Candidate] = []
    for strategy in strategies:
        df = bars_by_market.get((strategy.asset, strategy.timeframe))
        if df is None or df.empty:
            continue
        candidates.extend(_member_candidates(strategy, df, gene_pool, config))

    now = datetime.now().astimezone().isoformat()
    if not candidates:
        return TrueSimResult(
            strategy_count=len(strategies), sizing=sizing, initial_balance=config.initial_balance,
            final_balance=config.initial_balance, return_pct=0.0, max_drawdown_pct=0.0,
            peak_simultaneous_risk_pct=0.0, peak_risk_at=None, trades_taken=0, trades_blocked=0,
            blocked_by_symbol_lock=0, blocked_by_aggregate_risk=0, computed_at=now,
        )

    candidates.sort(key=lambda c: c.entry_time)

    balance = config.initial_balance
    open_trades: list[tuple[_Candidate, float]] = []  # (candidate, risk amount)
    accepted: list[tuple[_Candidate, float]] = []
    blocked_symbol_lock = blocked_aggregate = 0
    blocked_samples: list[BlockedEntry] = []

    def close_due(now_time: pd.Timestamp) -> None:
        nonlocal balance
        for trade, risk_amount in list(open_trades):
            if trade.exit_time <= now_time:
                balance += risk_amount * trade.r_multiple
                open_trades.remove((trade, risk_amount))

    for candidate in candidates:
        close_due(candidate.entry_time)

        if any(trade.symbol == candidate.symbol for trade, _ in open_trades):
            blocked_symbol_lock += 1
            if len(blocked_samples) < 20:
                blocked_samples.append(
                    BlockedEntry(candidate.entry_time.isoformat(), candidate.strategy_id, candidate.symbol, "symbol lock")
                )
            continue

        base = balance if sizing == COMPOUNDING else config.initial_balance
        risk_amount = base * config.risk_pct
        open_risk = sum(amount for _, amount in open_trades)
        if balance > 0 and (open_risk + risk_amount) / balance * 100 > max_aggregate_risk_pct:
            blocked_aggregate += 1
            if len(blocked_samples) < 20:
                blocked_samples.append(
                    BlockedEntry(
                        candidate.entry_time.isoformat(), candidate.strategy_id, candidate.symbol, "max aggregate risk"
                    )
                )
            continue

        open_trades.append((candidate, risk_amount))
        accepted.append((candidate, risk_amount))

    close_due(max(candidate.exit_time for candidate in candidates))

    equity_series, peak_risk_pct, peak_risk_at = _equity_path(accepted, config.initial_balance)
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max

    return TrueSimResult(
        strategy_count=len(strategies),
        sizing=sizing,
        initial_balance=config.initial_balance,
        final_balance=round(float(equity_series.iloc[-1]), 2),
        return_pct=round(float((equity_series.iloc[-1] - config.initial_balance) / config.initial_balance * 100), 2),
        max_drawdown_pct=round(float(-drawdown.min() * 100), 2),
        peak_simultaneous_risk_pct=round(peak_risk_pct, 2),
        peak_risk_at=peak_risk_at,
        trades_taken=len(accepted),
        trades_blocked=blocked_symbol_lock + blocked_aggregate,
        blocked_by_symbol_lock=blocked_symbol_lock,
        blocked_by_aggregate_risk=blocked_aggregate,
        equity=[
            {"t": timestamp.isoformat(), "equity": round(float(value), 2)}
            for timestamp, value in equity_series.resample("1D").last().dropna().items()
        ],
        blocked_samples=blocked_samples,
        computed_at=now,
    )


def _equity_path(
    accepted: list[tuple[_Candidate, float]], initial_balance: float
) -> tuple[pd.Series, float, str | None]:
    """Floating equity over the whole history, plus the peak simultaneous
    open risk and when it happened."""
    realised = pd.Series(
        [risk_amount * trade.r_multiple for trade, risk_amount in accepted],
        index=pd.DatetimeIndex([trade.exit_time for trade, _ in accepted]),
    ).groupby(level=0).sum()

    # Daily grid across the whole span: fine enough for a drawdown figure,
    # cheap enough for a 20-member portfolio over years of history.
    span = pd.date_range(
        min(trade.entry_time for trade, _ in accepted).normalize(),
        max(trade.exit_time for trade, _ in accepted).normalize(),
        freq="1D",
        tz=realised.index.tz,
    )
    realised_daily = realised.resample("1D").sum().reindex(span, fill_value=0.0)
    closed_equity = initial_balance + realised_daily.cumsum()

    floating = pd.Series(0.0, index=span)
    open_risk = pd.Series(0.0, index=span)
    for trade, risk_amount in accepted:
        daily_r = trade.unrealised.resample("1D").last().dropna()
        daily_r = daily_r[daily_r.index < trade.exit_time.normalize()]
        if daily_r.empty:
            continue
        aligned = daily_r.reindex(span).ffill().where(
            (span >= daily_r.index[0]) & (span < trade.exit_time.normalize()), 0.0
        ).fillna(0.0)
        floating = floating.add(aligned * risk_amount, fill_value=0.0)
        open_risk = open_risk.add(
            pd.Series(risk_amount, index=span).where(
                (span >= trade.entry_time.normalize()) & (span < trade.exit_time.normalize()), 0.0
            ),
            fill_value=0.0,
        )

    equity = closed_equity + floating
    risk_pct = (open_risk / equity.replace(0, np.nan) * 100).fillna(0.0)
    peak_index = int(risk_pct.to_numpy().argmax()) if len(risk_pct) else 0
    peak_pct = float(risk_pct.iloc[peak_index]) if len(risk_pct) else 0.0
    peak_at = risk_pct.index[peak_index].isoformat() if len(risk_pct) else None
    return equity, peak_pct, peak_at
