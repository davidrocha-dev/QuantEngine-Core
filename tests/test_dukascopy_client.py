import lzma
import struct
from datetime import datetime, timezone

import httpx
import pandas as pd
import pytest

from app.data.dukascopy_client import (
    DukascopyClient,
    DukascopyDownloadError,
    build_hour_url,
    decompress_bi5,
    parse_ticks,
    resample_ticks_to_ohlc,
    utc_hour_range,
)
from app.data.symbols import get_instrument


def make_bi5(records: list[tuple[int, int, int, float, float]]) -> bytes:
    """Build a synthetic .bi5 payload — keeps this suite fast and
    deterministic, with no live network dependency on the real Dukascopy
    feed (which the engine does hit for real; see
    app/data/build_bundled_subset.py and its generated .parquet files)."""
    raw = b"".join(struct.pack(">IIIff", *r) for r in records)
    compressor = lzma.LZMACompressor(format=lzma.FORMAT_ALONE)
    return compressor.compress(raw) + compressor.flush()


def test_build_hour_url_month_is_zero_indexed():
    hour = datetime(2024, 3, 5, 14, tzinfo=timezone.utc)
    url = build_hour_url("EURUSD", hour)
    assert url == "https://datafeed.dukascopy.com/datafeed/EURUSD/2024/02/05/14h_ticks.bi5"


def test_decompress_and_parse_round_trip():
    instrument = get_instrument("EURUSD")
    hour_start = datetime(2024, 3, 5, 14, tzinfo=timezone.utc)
    records = [
        (0, 108234, 108220, 1.5, 2.0),
        (500, 108236, 108222, 1.0, 1.0),
        (1500, 108235, 108221, 0.5, 0.5),
    ]
    payload = make_bi5(records)

    decompressed = decompress_bi5(payload)
    ticks = parse_ticks(decompressed, instrument, hour_start)

    assert len(ticks) == 3
    assert ticks.iloc[0]["ask"] == pytest.approx(1.08234)
    assert ticks.iloc[0]["bid"] == pytest.approx(1.08220)
    assert ticks.iloc[1]["timestamp"] == datetime(2024, 3, 5, 14, 0, 0, 500000, tzinfo=timezone.utc)


def test_decompress_empty_payload_is_empty_not_error():
    assert decompress_bi5(b"") == b""


def test_parse_ticks_rejects_truncated_payload():
    instrument = get_instrument("EURUSD")
    hour_start = datetime(2024, 3, 5, 14, tzinfo=timezone.utc)
    with pytest.raises(Exception):
        parse_ticks(b"\x00" * 7, instrument, hour_start)


def test_resample_ticks_to_ohlc():
    hour_start = datetime(2024, 3, 5, 14, tzinfo=timezone.utc)
    ticks = pd.DataFrame(
        {
            "timestamp": [
                hour_start,
                hour_start.replace(second=30),
                hour_start.replace(minute=1),
            ],
            "bid": [1.0800, 1.0802, 1.0810],
            "ask": [1.0802, 1.0804, 1.0812],
            "bid_volume": [1.0, 1.0, 1.0],
            "ask_volume": [1.0, 1.0, 1.0],
        }
    )
    bars = resample_ticks_to_ohlc(ticks, "M1")
    assert len(bars) == 2
    assert bars.iloc[0]["open"] == pytest.approx(1.0801)
    assert bars.iloc[1]["close"] == pytest.approx(1.0811)


def test_resample_empty_ticks_returns_empty_frame():
    empty = pd.DataFrame(columns=["timestamp", "bid", "ask", "bid_volume", "ask_volume"])
    bars = resample_ticks_to_ohlc(empty, "M1")
    assert bars.empty


def test_resample_unsupported_timeframe_raises():
    ticks = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "bid": [1.0],
            "ask": [1.0],
            "bid_volume": [1.0],
            "ask_volume": [1.0],
        }
    )
    with pytest.raises(ValueError):
        resample_ticks_to_ohlc(ticks, "M3")


def test_fetch_hour_retries_on_429_then_succeeds():
    hour_start = datetime(2024, 3, 5, 14, tzinfo=timezone.utc)
    payload = make_bi5([(0, 108234, 108220, 1.5, 2.0)])
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, content=payload)

    client = DukascopyClient()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    import asyncio

    result = asyncio.run(client.fetch_hour("EURUSD", hour_start))
    assert calls["n"] == 2
    assert len(result.ticks) == 1


def test_fetch_hour_gives_up_after_max_retries():
    hour_start = datetime(2024, 3, 5, 14, tzinfo=timezone.utc)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "0"})

    client = DukascopyClient(max_retries=3)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    import asyncio

    with pytest.raises(DukascopyDownloadError):
        asyncio.run(client.fetch_hour("EURUSD", hour_start))


def test_fetch_hour_404_is_empty_not_error():
    hour_start = datetime(2024, 3, 5, 14, tzinfo=timezone.utc)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = DukascopyClient()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    import asyncio

    result = asyncio.run(client.fetch_hour("EURUSD", hour_start))
    assert result.ticks.empty


def test_utc_hour_range():
    start = datetime(2024, 1, 1, 0, 30, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 3, 0, tzinfo=timezone.utc)
    hours = list(utc_hour_range(start, end))
    assert hours == [
        datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 2, tzinfo=timezone.utc),
    ]
