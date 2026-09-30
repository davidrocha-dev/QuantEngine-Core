"""Reading a Strategy Tester report and pairing it with the backtest."""

import pandas as pd
import pytest

from app.engine.backtest import Trade
from app.engine.mt5_report import ReportUnreadable, read_report
from app.engine.trade_parity import compare_trades

HEADER = ["Time", "Deal", "Symbol", "Type", "Direction", "Volume", "Price", "Order",
          "Commission", "Swap", "Profit", "Balance", "Comment"]


def _deal_rows(trades):
    """(entry 'YYYY.MM.DD HH:MM:SS', exit, 'buy'|'sell', entry price, exit price, profit)"""
    rows = [["2024.01.02 00:00:00", "1", "", "balance", "", "", "", "", "0.00", "0.00", "10 000.00", "10 000.00", ""]]
    deal = 2
    for entry, exit_, side, p_in, p_out, profit in trades:
        close_side = "sell" if side == "buy" else "buy"
        rows.append([entry, str(deal), "EURUSD", side, "in", "0.10", str(p_in), str(deal), "-0.35", "0.00", "0.00", "", ""])
        rows.append([exit_, str(deal + 1), "EURUSD", close_side, "out", "0.10", str(p_out), str(deal + 1),
                     "-0.35", "0.00", f"{profit:.2f}", "", "sl" if profit < 0 else "tp"])
        deal += 2
    return rows


def _html(rows) -> str:
    cells = lambda row, tag: "".join(f"<{tag}>{c}</{tag}>" for c in row)
    body = "".join(f"<tr>{cells(r, 'td')}</tr>" for r in rows)
    return (
        "<html><body><table><tr><td>Orders</td></tr></table>"
        f"<table><tr>{cells(HEADER, 'th')}</tr>{body}</table></body></html>"
    )


TRADES = [
    ("2024.01.03 10:00:03", "2024.01.03 14:22:10", "buy", 1.1000, 1.1030, 30.0),
    ("2024.01.04 09:00:01", "2024.01.04 11:05:40", "sell", 1.0950, 1.0970, -20.0),
]


def test_a_utf16_html_report_is_read_into_round_trips(tmp_path):
    path = tmp_path / "ReportTester.html"
    path.write_bytes(_html(_deal_rows(TRADES)).encode("utf-16"))

    trades = read_report(path)

    assert len(trades) == 2
    assert trades[0].direction == 1 and trades[1].direction == -1
    assert trades[0].entry_time == pd.Timestamp("2024-01-03 10:00:03")
    assert trades[0].net_profit == pytest.approx(30.0 - 0.70)
    assert trades[1].exit_price == pytest.approx(1.0970)


def test_a_csv_export_is_read_too(tmp_path):
    path = tmp_path / "deals.csv"
    lines = [";".join(HEADER)] + [";".join(row) for row in _deal_rows(TRADES)]
    path.write_text("\n".join(lines), encoding="utf-8")
    assert len(read_report(path)) == 2


def test_a_file_without_deals_says_so(tmp_path):
    path = tmp_path / "other.html"
    path.write_text("<html><table><tr><td>Nothing</td></tr></table></html>")
    with pytest.raises(ReportUnreadable, match="Deals"):
        read_report(path)


def _bars():
    stamps = pd.date_range("2024-01-03 00:00", periods=60, freq="1h")
    return pd.DataFrame({"timestamp": stamps.tz_localize("UTC"), "broker_time": stamps,
                         "open": 1.1, "high": 1.1, "low": 1.1, "close": 1.1, "volume": 1.0})


def _bt(entry_idx, direction, r):
    return Trade(entry_idx, entry_idx + 3, direction, 1.1, 1.1, 1.09, 1.12, r, r * 100, "target")


def test_matching_trades_pair_and_the_first_difference_is_named(tmp_path):
    path = tmp_path / "r.html"
    path.write_bytes(_html(_deal_rows(TRADES)).encode("utf-16"))
    mt5 = read_report(path)

    df = _bars()
    # 10:00 long (matches), 33:00 = next day 09:00 short (matches), 40:00 an extra long MT5 never took
    backtest = [_bt(10, 1, 1.0), _bt(33, -1, -1.0), _bt(40, 1, 0.5)]

    report = compare_trades(backtest, df, mt5, bar_minutes=60, point_size=0.00001)

    assert report.matched == 2
    assert report.clock_shift_hours == 0
    assert report.same_outcome_pct == 100.0
    # The extra backtest trade is after the report's last deal, so it is outside the period.
    assert report.backtest_trades == 2


def test_a_wrong_server_clock_shows_up_as_a_constant_shift(tmp_path):
    path = tmp_path / "r.html"
    path.write_bytes(_html(_deal_rows(TRADES)).encode("utf-16"))
    mt5 = read_report(path)

    # The backtest's clock is three hours behind the server's.
    df = _bars()
    backtest = [_bt(7, 1, 1.0), _bt(30, -1, -1.0)]

    report = compare_trades(backtest, df, mt5, bar_minutes=60, point_size=0.00001)

    assert report.clock_shift_hours == -3
    assert "Server time" in report.verdict
