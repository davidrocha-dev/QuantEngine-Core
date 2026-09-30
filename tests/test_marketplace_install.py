"""Installing a Marketplace EA — REQ-MKT-1.3.

The engine's half of the flow: the shell fetches the compiled binary
(it holds the licence key), the engine writes it where MetaTrader will
find it (it knows where that is).
"""

import base64
import hashlib

import pytest

from app.broker import mt5_install
from app.broker.mt5_install import install_compiled_ea

COMPILED = b"MQ5\x00 not really a binary, but it is bytes and it is not source"


@pytest.fixture
def fake_terminals(tmp_path, monkeypatch):
    first = tmp_path / "terminal-a"
    second = tmp_path / "terminal-b"
    for terminal in (first, second):
        (terminal / "MQL5").mkdir(parents=True)
    monkeypatch.setattr(mt5_install, "terminal_data_dirs", lambda: [first, second])
    return [first, second]


def test_a_compiled_ea_lands_in_every_terminal(fake_terminals):
    result = install_compiled_ea("QE_MKT_ABC.ex5", COMPILED)

    assert len(result.installed_paths) == 2
    for terminal in fake_terminals:
        installed = terminal / mt5_install.EXPERTS_SUBDIR / "QE_MKT_ABC.ex5"
        assert installed.read_bytes() == COMPILED


def test_a_corrupted_download_is_not_installed(fake_terminals):
    # This binary is about to be given permission to trade. If it is not
    # byte-for-byte what the listing published, it does not get written.
    result = install_compiled_ea("QE_MKT_ABC.ex5", COMPILED, expected_sha256="0" * 64)

    assert result.installed_paths == []
    assert "Checksum mismatch" in result.detail
    for terminal in fake_terminals:
        assert not (terminal / mt5_install.EXPERTS_SUBDIR / "QE_MKT_ABC.ex5").exists()


def test_a_matching_checksum_installs(fake_terminals):
    result = install_compiled_ea("QE_MKT_ABC.ex5", COMPILED, expected_sha256=hashlib.sha256(COMPILED).hexdigest())

    assert len(result.installed_paths) == 2


def test_source_is_refused_even_here(fake_terminals):
    # The server will not serve one, but the engine does not take that on
    # trust: this is the process that writes to the trading terminal.
    result = install_compiled_ea("QE_MKT_ABC.mq5", b"// source")

    assert result.installed_paths == []
    assert ".ex5" in result.detail


def test_no_terminal_is_a_clear_message_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(mt5_install, "terminal_data_dirs", lambda: [])

    result = install_compiled_ea("QE_MKT_ABC.ex5", COMPILED)

    assert result.installed_paths == []
    assert "No MetaTrader 5 terminal" in result.detail


def test_the_endpoint_rejects_a_payload_that_is_not_base64(fake_terminals):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        bad = client.post(
            "/marketplace/install", json={"filename": "QE.ex5", "content_base64": "not base64!!"}
        )
        good = client.post(
            "/marketplace/install",
            json={
                "filename": "QE.ex5",
                "content_base64": base64.b64encode(COMPILED).decode(),
                "sha256": hashlib.sha256(COMPILED).hexdigest(),
            },
        )

    assert bad.status_code == 422
    assert good.status_code == 200
    assert len(good.json()["installed_paths"]) == 2


def test_a_name_that_climbs_out_of_the_experts_folder_is_refused(tmp_path, monkeypatch):
    """The file name comes from whoever published the listing."""
    from app.broker import mt5_install

    monkeypatch.setattr(mt5_install, "terminal_data_dirs", lambda: [tmp_path / "terminal"])
    for name in ("..\\..\\Startup\\evil.ex5", "../evil.ex5", str(tmp_path / "abs.ex5"), "sub/evil.ex5"):
        result = install_compiled_ea(name, COMPILED)
        assert not result.installed_paths, name
    assert not (tmp_path / "abs.ex5").exists()
