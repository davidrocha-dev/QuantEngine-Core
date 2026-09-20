"""Reading the QuantEngine Pilot's report — REQ-PORT-4.

The Pilot EA (app/codegen/templates/pilot_ea.mq5) runs on one chart per
terminal and appends every closed QuantEngine deal to a CSV in MT5's
shared Files folder. This module installs it, finds that file and imports
the rows.

Two properties matter more than completeness here:

  * **Idempotent.** Rows are keyed by the broker's own deal ticket, so
    re-reading the file — which happens on every poll — can never
    double-count a trade.
  * **Attributed, or kept anyway.** A deal's magic number maps back to the
    strategy that generated it. A deal whose magic matches nothing in the
    Vault is still stored with a null strategy_id: a bot running from a
    deleted strategy is exactly the situation you want to see, not hide.
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.broker.mt5_install import install_ea
from app.codegen.mql5 import _magic_number
from app.models import LiveTrade, Strategy

PILOT_FILENAME = "QuantEngine_Pilot.mq5"
REPORT_FILENAME = "quantengine_live.csv"
PILOT_SOURCE = Path(__file__).resolve().parent.parent / "codegen" / "templates" / "pilot_ea.mq5"


@dataclass
class ImportResult:
    imported: int
    skipped_existing: int
    unattributed: int
    report_path: str | None
    detail: str

    def to_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}


def common_files_dir() -> Path | None:
    """MT5's shared Files folder — the one FILE_COMMON writes to."""
    if sys.platform != "win32":
        return None
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    path = Path(appdata) / "MetaQuotes" / "Terminal" / "Common" / "Files"
    return path if path.exists() else None


def report_path() -> Path | None:
    directory = common_files_dir()
    if directory is None:
        return None
    path = directory / REPORT_FILENAME
    return path if path.exists() else None


def install_pilot():
    """REQ-PORT-4.1 — put the Pilot in every terminal and compile it."""
    source = PILOT_SOURCE.read_text(encoding="utf-8")
    return install_ea(PILOT_FILENAME, source)


def _magic_to_strategy(session: Session) -> dict[int, str]:
    """Every Vault strategy's magic number, recomputed the same way the
    generated EA does, so a deal can be traced back to it."""
    return {
        _magic_number(row.id): row.id
        for row in session.scalars(select(Strategy).where(Strategy.deleted_at.is_(None)))
    }


def _parse_closed_at(value: str) -> datetime | None:
    # MQL5's TimeToString writes "YYYY.MM.DD HH:MM:SS".
    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def import_live_trades(session: Session, path: Path | None = None) -> ImportResult:
    """Imports the Pilot's report. Safe to call repeatedly."""
    path = path or report_path()
    if path is None or not path.exists():
        return ImportResult(
            imported=0, skipped_existing=0, unattributed=0, report_path=None,
            detail="No Pilot report found — install the Pilot and attach it to a chart in MT5.",
        )

    magic_map = _magic_to_strategy(session)
    existing = {ticket for ticket in session.scalars(select(LiveTrade.deal_ticket))}

    imported = skipped = unattributed = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            ticket = (row.get("deal_ticket") or "").strip()
            if not ticket or ticket == "deal_ticket":
                continue
            if ticket in existing:
                skipped += 1
                continue

            closed_at = _parse_closed_at(row.get("closed_at", ""))
            if closed_at is None:
                continue

            try:
                magic = int(row.get("magic", 0))
                profit = float(row.get("profit", 0.0))
            except ValueError:
                continue

            strategy_id = magic_map.get(magic)
            if strategy_id is None:
                unattributed += 1

            session.add(
                LiveTrade(
                    deal_ticket=ticket,
                    strategy_id=strategy_id,
                    magic=magic,
                    symbol=(row.get("symbol") or "").strip(),
                    closed_at=closed_at,
                    volume=_float_or_none(row.get("volume")),
                    profit=profit,
                    exit_price=_float_or_none(row.get("exit_price")),
                    # Older Pilot builds wrote no broker column; those rows
                    # simply cannot take part in the multi-broker check.
                    broker=(row.get("broker") or "").strip() or None,
                )
            )
            existing.add(ticket)
            imported += 1

    session.commit()
    return ImportResult(
        imported=imported,
        skipped_existing=skipped,
        unattributed=unattributed,
        report_path=str(path),
        detail=f"Imported {imported} new closed trades ({unattributed} from unknown strategies).",
    )


def _float_or_none(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None
