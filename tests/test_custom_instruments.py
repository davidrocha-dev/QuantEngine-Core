"""Instruments the user adds — and the reason they cannot be added blind.

The built-in table was verified against the live feed. A code somebody
types has had no such check, and a wrong point divisor does not fail: it
produces a backtest at the wrong order of magnitude with every number
downstream of it quietly nonsense. So the flow is probe first, save
second, and these tests are about that order being enforced.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from app.data.custom_instruments import (
    InstrumentRefused,
    all_instruments,
    load,
    probe,
    remove,
    save,
)
from app.data.symbols import INSTRUMENTS, get_instrument


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


def test_the_built_in_table_covers_more_than_the_majors():
    # The screen used to offer eleven pairs and no way to want anything
    # else.
    categories = {row.category for row in INSTRUMENTS.values()}
    assert {"fx", "metal", "index", "commodity", "crypto"} <= categories
    assert len(INSTRUMENTS) > 50


def test_every_built_in_jpy_pair_uses_the_jpy_divisor():
    # The one divisor mistake that is easy to make by hand, and the one
    # that shifts a price by a factor of a hundred.
    for symbol, instrument in INSTRUMENTS.items():
        if instrument.category == "fx" and symbol.endswith("JPY"):
            assert instrument.point_divisor == 1000, symbol
        elif instrument.category == "fx":
            assert instrument.point_divisor == 100000, symbol


def test_a_saved_instrument_is_found_by_the_rest_of_the_engine(data_dir, monkeypatch):
    from app.config import EngineConfig

    monkeypatch.setattr("app.config.get_config", lambda: EngineConfig(data_dir=data_dir))
    save(data_dir, "TESTFX", "TESTFX", 100000, "fx")

    found = get_instrument("TESTFX")

    assert found.dukascopy_code == "TESTFX"
    assert found.point_divisor == 100000


def test_a_custom_entry_cannot_shadow_a_verified_one(data_dir):
    # Someone re-adding EURUSD with a wrong divisor would otherwise break
    # every existing strategy on it.
    with pytest.raises(InstrumentRefused, match="already built in"):
        save(data_dir, "EURUSD", "EURUSD", 10, "fx")


def test_the_built_ins_win_even_if_the_file_is_edited_by_hand(data_dir):
    data_dir.mkdir(parents=True)
    (data_dir / "custom_instruments.json").write_text(
        '{"EURUSD": {"dukascopy_code": "NONSENSE", "point_divisor": 10, "category": "fx"}}',
        encoding="utf-8",
    )

    assert all_instruments(data_dir)["EURUSD"].dukascopy_code == "EURUSD"


def test_a_divisor_that_is_not_a_real_one_is_refused(data_dir):
    with pytest.raises(InstrumentRefused, match="Point divisor"):
        save(data_dir, "WEIRD", "WEIRD", 12345, "fx")


def test_a_nonsense_symbol_is_refused(data_dir):
    with pytest.raises(InstrumentRefused, match="3-16 characters"):
        save(data_dir, "no lower case or spaces", "X", 100000, "fx")


def test_a_corrupt_file_does_not_take_the_engine_down(data_dir):
    data_dir.mkdir(parents=True)
    (data_dir / "custom_instruments.json").write_text("{ not json", encoding="utf-8")

    assert load(data_dir) == {}
    assert "EURUSD" in all_instruments(data_dir)


def test_removing_what_was_added(data_dir):
    save(data_dir, "TESTFX", "TESTFX", 100000, "fx")

    assert remove(data_dir, "TESTFX") is True
    assert remove(data_dir, "TESTFX") is False
    assert "TESTFX" not in load(data_dir)


# --- the probe ------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int, content: bytes = b""):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeClient:
    """Stands in for the feed. `_client.get` is what probe reaches for."""

    def __init__(self, response):
        self._client = self
        self._response = response

    async def get(self, url):
        return self._response

    async def aclose(self):
        return None


@pytest.mark.asyncio
async def test_a_code_the_feed_does_not_know_says_so_plainly():
    result = await probe("MADEUP", 100000, client=FakeClient(FakeResponse(404)))

    assert result.ticks == 0
    assert result.median_price is None
    assert "no instrument called MADEUP" in result.detail


@pytest.mark.asyncio
async def test_the_probe_reports_the_price_the_divisor_produces(monkeypatch):
    # The whole point: the user confirms a price they recognise instead
    # of a divisor they had to guess.
    def fake_parse(decompressed, instrument, hour_start):
        # Two ticks around 1.0850 at the divisor under test.
        return pd.DataFrame(
            {
                "timestamp": [hour_start, hour_start],
                "bid": [1.08495, 1.08505],
                "ask": [1.08505, 1.08515],
                "bid_volume": [1.0, 1.0],
                "ask_volume": [1.0, 1.0],
            }
        )

    monkeypatch.setattr("app.data.custom_instruments.parse_ticks", fake_parse)
    monkeypatch.setattr("app.data.custom_instruments.decompress_bi5", lambda raw: raw)

    result = await probe("SOMEFX", 100000, client=FakeClient(FakeResponse(200, b"payload")))

    assert result.ticks == 2
    assert result.median_price == pytest.approx(1.08505, abs=1e-5)
    assert "1.08505" in result.detail
    assert "out by a factor of ten" in result.detail


@pytest.mark.asyncio
async def test_an_instrument_with_no_ticks_is_called_out_as_untradeable(monkeypatch):
    monkeypatch.setattr(
        "app.data.custom_instruments.parse_ticks",
        lambda decompressed, instrument, hour_start: pd.DataFrame(
            columns=["timestamp", "bid", "ask", "bid_volume", "ask_volume"]
        ),
    )
    monkeypatch.setattr("app.data.custom_instruments.decompress_bi5", lambda raw: raw)

    result = await probe("THINONE", 100000, client=FakeClient(FakeResponse(200, b"payload")))

    assert result.ticks == 0
    assert "too illiquid to breed on" in result.detail


@pytest.mark.asyncio
async def test_the_probe_refuses_an_impossible_divisor_before_touching_the_network():
    with pytest.raises(InstrumentRefused, match="Point divisor"):
        await probe("EURUSD", 3, client=FakeClient(FakeResponse(200)))
