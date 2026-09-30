"""The engine answers the desktop app and nothing else on the machine."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_a_foreign_host_is_refused_even_without_a_token(client):
    """DNS rebinding makes a page same-origin with 127.0.0.1, but the
    browser still sends the attacker's own host name."""
    response = client.get("/status", headers={"Host": "attacker.example"})
    assert response.status_code == 403


def test_loopback_hosts_are_allowed(client):
    for host in ("127.0.0.1:7831", "localhost:7831"):
        assert client.get("/status", headers={"Host": host}).status_code == 200


def test_with_a_token_configured_every_request_must_carry_it(client, monkeypatch):
    monkeypatch.setenv("QUANTENGINE_API_TOKEN", "s3cret")

    # A cross-origin "simple" POST - what any web page can send without a
    # preflight - used to reach the handler and run it.
    forged = client.post(
        "/campaign/pause",
        headers={"Origin": "https://evil.example", "Content-Type": "text/plain"},
        content="x",
    )
    assert forged.status_code == 403
    assert client.get("/status", headers={"X-QuantEngine-Token": "wrong"}).status_code == 403
    assert client.get("/status", headers={"X-QuantEngine-Token": "s3cret"}).status_code == 200


def test_a_cors_preflight_still_gets_its_answer(client, monkeypatch):
    monkeypatch.setenv("QUANTENGINE_API_TOKEN", "s3cret")
    response = client.options(
        "/status",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "x-quantengine-token",
        },
    )
    assert response.status_code == 200


def test_the_event_socket_takes_the_token_as_a_query_parameter(client, monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.setenv("QUANTENGINE_API_TOKEN", "s3cret")
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/events") as socket:
            socket.receive_text()
    with client.websocket_connect("/events?token=s3cret"):
        pass
