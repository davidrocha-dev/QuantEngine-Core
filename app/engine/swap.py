"""Overnight financing (swap), charged the way MetaTrader charges it.

A position held through the broker's midnight rollover is charged - or,
rarely, paid - the symbol's swap for that side, once per night, and three
times on one night of the week to cover the weekend. The backtest charged
nothing, which on gold, indices and high-yield crosses is not a rounding
error: a strategy that holds for days can make its whole edge in price
and hand it back in financing.

Two kinds of quote exist in practice:

  - "price": swap in price units per lot per night (MT5's points mode,
    already multiplied by the symbol's point). It becomes money through
    the contract size and the quote-to-account rate, like a price move.
  - "money": swap already in the account currency per lot per night.

Rollovers are counted on the broker's clock: every broker day a position
stays open past the end of is one night, and the day named by
`triple_day` (0 = Monday ... 6 = Sunday) counts three.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PRICE = "price"
MONEY = "money"


@dataclass(frozen=True)
class SwapSpec:
    long: float  # per lot per night, signed: negative is a charge
    short: float
    kind: str = PRICE
    triple_day: int = 2  # Wednesday, which is what nearly every FX broker uses

    def per_lot_night(self, direction: int) -> float:
        return self.long if direction > 0 else self.short


def nights_held(entry_day: np.datetime64, exit_day: np.datetime64, triple_day: int) -> int:
    """Rollovers between two broker dates (both day-truncated).

    One per trading day whose end the position is held past; the triple
    day counts three. Weekend days are skipped - the triple night already
    charges for them.
    """
    if exit_day <= entry_day:
        return 0
    days = np.arange(entry_day, exit_day, dtype="datetime64[D]")
    # 1970-01-01 was a Thursday: numpy's weekday is (days + 3) % 7 with Monday = 0.
    weekdays = (days.astype("int64") + 3) % 7
    trading = weekdays < 5
    return int(trading.sum() + 2 * (trading & (weekdays == triple_day)).sum())
