"""Reading a MetaTrader 5 Strategy Tester report.

The report is what MetaTrader says the EA actually did, and the only
honest way to find where an exported strategy departs from its backtest
is to lay the two trade lists side by side. The tester saves it as HTML
(UTF-16, from the Backtest tab's "Report" context menu) and the history
tab can export the same deals as CSV; both carry a Deals table:

    Time | Deal | Symbol | Type | Direction | Volume | Price | Order |
    Commission | Swap | Profit | Balance | Comment

Deal times are the broker's server clock. Deals are paired into round
trips by the order in which they open and close a position: the EA holds
one position at a time, so an "in" is followed by one or more "out"s
(PARTIAL TP closes in two).

The parser is written against the stdlib's HTML parser so the engine does
not grow a dependency for one table.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

_HEADER_KEYS = {"time", "deal", "type", "direction", "price", "profit"}


class ReportUnreadable(ValueError):
    """The file is not a report this module understands. The message is shown to the user."""


@dataclass
class Mt5Deal:
    time: pd.Timestamp  # server clock, tz-naive
    symbol: str
    type: str  # buy | sell | balance | ...
    direction: str  # in | out | in/out
    volume: float
    price: float
    commission: float
    swap: float
    profit: float


@dataclass
class Mt5Trade:
    entry_time: pd.Timestamp
    direction: int  # 1 long, -1 short
    entry_price: float
    volume: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    net_profit: float = 0.0  # profit + commission + swap over every deal of the trade
    deals: list[Mt5Deal] = field(default_factory=list)


class _TableCollector(HTMLParser):
    """Every table row in the document, as lists of cell text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _decode(raw: bytes) -> str:
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    # The tester writes UTF-16 without a BOM on some builds: every other
    # byte of plain ASCII markup is then zero.
    if len(raw) > 64 and raw[1:64:2].count(0) > 24:
        return raw.decode("utf-16-le")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _rows_from_csv(text: str) -> list[list[str]]:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    return [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(text), dialect)]


def _number(text: str) -> float:
    cleaned = text.replace(" ", "").replace(" ", "")
    # A decimal comma (some locales) when there is no point; otherwise a
    # comma is a thousands separator.
    cleaned = cleaned.replace(",", ".") if "," in cleaned and "." not in cleaned else cleaned.replace(",", "")
    if not cleaned or cleaned == "-":
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _deals_from_rows(rows: list[list[str]]) -> list[Mt5Deal]:
    header_at = None
    for index, row in enumerate(rows):
        lowered = {cell.strip().lower() for cell in row}
        if _HEADER_KEYS <= lowered:
            header_at = index  # the Deals table is the last one with these columns
    if header_at is None:
        raise ReportUnreadable(
            "No Deals table found. Save the report from the Strategy Tester's Backtest tab "
            "(right-click -> Report -> HTML) or export the deals as CSV."
        )

    columns = [cell.strip().lower() for cell in rows[header_at]]
    col = {name: columns.index(name) for name in columns if name}
    deals: list[Mt5Deal] = []
    for row in rows[header_at + 1:]:
        if len(row) < len(columns) or not row[col["time"]].strip():
            if deals:
                break  # the table has ended
            continue
        try:
            time = pd.Timestamp(row[col["time"]].replace(".", "-", 2))
        except (ValueError, TypeError):
            continue
        deals.append(
            Mt5Deal(
                time=time,
                symbol=row[col["symbol"]] if "symbol" in col else "",
                type=row[col["type"]].strip().lower(),
                direction=row[col["direction"]].strip().lower(),
                volume=_number(row[col["volume"]]) if "volume" in col else 0.0,
                price=_number(row[col["price"]]),
                commission=_number(row[col["commission"]]) if "commission" in col else 0.0,
                swap=_number(row[col["swap"]]) if "swap" in col else 0.0,
                profit=_number(row[col["profit"]]),
            )
        )
    return deals


def trades_from_deals(deals: list[Mt5Deal]) -> list[Mt5Trade]:
    """Round trips, in the order they opened."""
    trades: list[Mt5Trade] = []
    open_trade: Mt5Trade | None = None
    for deal in deals:
        if deal.type not in ("buy", "sell"):
            continue  # balance, credit, ...
        if deal.direction == "in":
            open_trade = Mt5Trade(
                entry_time=deal.time,
                direction=1 if deal.type == "buy" else -1,
                entry_price=deal.price,
                volume=deal.volume,
                net_profit=deal.profit + deal.commission + deal.swap,
                deals=[deal],
            )
            trades.append(open_trade)
        elif deal.direction in ("out", "out by") and open_trade is not None:
            open_trade.exit_time = deal.time
            open_trade.exit_price = deal.price
            open_trade.net_profit += deal.profit + deal.commission + deal.swap
            open_trade.deals.append(deal)
    return trades


def read_report(path: str | Path) -> list[Mt5Trade]:
    path = Path(path)
    if not path.exists():
        raise ReportUnreadable(f"File not found: {path}")
    text = _decode(path.read_bytes())
    if "<table" in text.lower() or "<tr" in text.lower():
        collector = _TableCollector()
        collector.feed(text)
        rows = collector.rows
    else:
        rows = _rows_from_csv(text)
    trades = trades_from_deals(_deals_from_rows(rows))
    if not trades:
        raise ReportUnreadable("The report has a Deals table but no trades in it.")
    return trades
