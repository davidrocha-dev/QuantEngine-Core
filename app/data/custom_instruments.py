"""Instruments the user adds themselves — beyond the built-in table.

The built-in list is verified: every code and divisor in `symbols.py` was
checked against the live feed. A code someone types in has had no such
check, and a wrong `point_divisor` does not fail - it silently produces a
backtest at the wrong order of magnitude.

So nothing is saved on trust. `probe` downloads a real hour of ticks and
hands back what the price would actually be, and the app shows that
number before anything is written. The user confirms a price they can
recognise rather than a divisor they had to guess.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from app.data.dukascopy_client import DukascopyClient, build_hour_url, decompress_bi5, parse_ticks
from app.data.symbols import INSTRUMENTS, Instrument

# Dukascopy's own point divisors. Anything else is a typo or a
# misunderstanding, and both produce prices that are wrong by a factor of
# ten or more.
VALID_DIVISORS = (10, 100, 1000, 10000, 100000, 1000000)
VALID_CATEGORIES = ("fx", "metal", "index", "commodity", "crypto")
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9._]{3,16}$")


class InstrumentRefused(ValueError):
    """The instrument cannot be added. The message is shown to the user."""


@dataclass(frozen=True)
class Probe:
    code: str
    point_divisor: int
    ticks: int
    median_price: float | None
    first_price: float | None
    last_price: float | None
    hour: str
    detail: str


def _sample_hour(now: datetime | None = None) -> datetime:
    """A weekday mid-session hour, recent but old enough to be published."""
    hour = (now or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0) - timedelta(days=3)
    while hour.weekday() > 3:
        hour -= timedelta(days=1)
    return hour.replace(hour=14)


async def probe(code: str, point_divisor: int, client: DukascopyClient | None = None) -> Probe:
    """Fetch one hour and report what this code and divisor actually give.

    Returns rather than raises on an empty or missing instrument: "that
    code returns nothing" is the answer to the user's question, not an
    error in asking it.
    """
    if point_divisor not in VALID_DIVISORS:
        raise InstrumentRefused(f"Point divisor must be one of {', '.join(str(d) for d in VALID_DIVISORS)}.")

    owns_client = client is None
    client = client or DukascopyClient(request_delay_seconds=0.25)
    hour = _sample_hour()
    instrument = Instrument(code, code, point_divisor, "fx")

    try:
        response = await client._client.get(build_hour_url(code, hour))
        if response.status_code == 404:
            return Probe(
                code, point_divisor, 0, None, None, None, hour.isoformat(),
                f"Dukascopy has no instrument called {code}. Check the code against their feed.",
            )
        response.raise_for_status()
        ticks = parse_ticks(decompress_bi5(response.content), instrument, hour)
    except InstrumentRefused:
        raise
    except Exception as error:
        return Probe(
            code, point_divisor, 0, None, None, None, hour.isoformat(),
            f"Could not reach the feed for {code}: {error}",
        )
    finally:
        if owns_client:
            await client.aclose()

    if ticks.empty:
        return Probe(
            code, point_divisor, 0, None, None, None, hour.isoformat(),
            f"{code} exists but published no ticks in this hour. It may be too illiquid to breed on.",
        )

    mid = (ticks["bid"] + ticks["ask"]) / 2
    return Probe(
        code=code,
        point_divisor=point_divisor,
        ticks=int(len(ticks)),
        median_price=float(mid.median()),
        first_price=float(mid.iloc[0]),
        last_price=float(mid.iloc[-1]),
        hour=hour.isoformat(),
        detail=(
            f"{len(ticks):,} ticks. At this divisor the price reads about "
            f"{mid.median():,.5f}. Save it only if that is what this instrument costs — "
            "if it is out by a factor of ten, the divisor is wrong."
        ),
    )


def _path(data_dir: Path) -> Path:
    return data_dir / "custom_instruments.json"


def load(data_dir: Path) -> dict[str, Instrument]:
    path = _path(data_dir)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A corrupt file must not take the engine down on startup; the
        # built-in instruments still work and the user can re-add.
        return {}
    return {
        symbol: Instrument(symbol, row["dukascopy_code"], int(row["point_divisor"]), row["category"])
        for symbol, row in raw.items()
        if isinstance(row, dict) and {"dukascopy_code", "point_divisor", "category"} <= row.keys()
    }


def save(data_dir: Path, symbol: str, code: str, point_divisor: int, category: str) -> Instrument:
    symbol = symbol.strip().upper()
    code = code.strip()
    if not SYMBOL_PATTERN.match(symbol):
        raise InstrumentRefused("A symbol is 3-16 characters, uppercase letters, digits, dot or underscore.")
    if symbol in INSTRUMENTS:
        raise InstrumentRefused(f"{symbol} is already built in and verified — no need to add it.")
    if point_divisor not in VALID_DIVISORS:
        raise InstrumentRefused(f"Point divisor must be one of {', '.join(str(d) for d in VALID_DIVISORS)}.")
    if category not in VALID_CATEGORIES:
        raise InstrumentRefused(f"Category must be one of {', '.join(VALID_CATEGORIES)}.")

    existing = load(data_dir)
    existing[symbol] = Instrument(symbol, code, point_divisor, category)
    _write(data_dir, existing)
    return existing[symbol]


def remove(data_dir: Path, symbol: str) -> bool:
    existing = load(data_dir)
    if symbol not in existing:
        return False
    del existing[symbol]
    _write(data_dir, existing)
    return True


def _write(data_dir: Path, instruments: dict[str, Instrument]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    _path(data_dir).write_text(
        json.dumps(
            {
                symbol: {
                    "dukascopy_code": instrument.dukascopy_code,
                    "point_divisor": instrument.point_divisor,
                    "category": instrument.category,
                }
                for symbol, instrument in sorted(instruments.items())
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def all_instruments(data_dir: Path) -> dict[str, Instrument]:
    """Built-ins first, so a custom entry can never shadow a verified one."""
    return {**load(data_dir), **INSTRUMENTS}
