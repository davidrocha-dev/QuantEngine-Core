import httpx

from app.engine import notifications
from app.engine.notifications import notify_event, send_telegram


class _Response:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


def settings_with(**overrides) -> dict:
    notifications_settings = {
        "os_enabled": True,
        "telegram_enabled": True,
        "telegram_bot_token": "token",
        "telegram_chat_id": "chat",
        "events": {"campaign_complete": {"os": True, "telegram": True}},
    }
    notifications_settings.update(overrides)
    return {"notifications": notifications_settings}


def test_a_missing_token_or_chat_id_is_reported_not_attempted(monkeypatch):
    called = False

    def fail(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(notifications.httpx, "post", fail)

    result = send_telegram("", "chat", "hello")

    assert result.sent is False
    assert "bot token" in result.detail
    assert called is False


def test_a_successful_send_reports_sent(monkeypatch):
    monkeypatch.setattr(notifications.httpx, "post", lambda *a, **k: _Response(200))
    assert send_telegram("token", "chat", "hello").sent is True


def test_telegrams_own_explanation_is_surfaced(monkeypatch):
    # "chat not found" tells the user what to fix; "400" does not.
    monkeypatch.setattr(
        notifications.httpx, "post", lambda *a, **k: _Response(400, {"description": "Bad Request: chat not found"})
    )

    result = send_telegram("token", "chat", "hello")

    assert result.sent is False
    assert "chat not found" in result.detail


def test_a_network_failure_never_raises(monkeypatch):
    def boom(*args, **kwargs):
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(notifications.httpx, "post", boom)

    result = send_telegram("token", "chat", "hello")

    assert result.sent is False
    assert "Telegram" in result.detail


def test_event_delivery_respects_the_channel_toggle(monkeypatch):
    monkeypatch.setattr(notifications.httpx, "post", lambda *a, **k: _Response(200))

    assert notify_event(settings_with(), "campaign_complete", "done").sent is True
    assert notify_event(settings_with(telegram_enabled=False), "campaign_complete", "done").sent is False


def test_event_delivery_respects_the_per_event_toggle(monkeypatch):
    # REQ-SET-7.4 — per event, per channel.
    monkeypatch.setattr(notifications.httpx, "post", lambda *a, **k: _Response(200))
    off = settings_with(events={"campaign_complete": {"os": True, "telegram": False}})

    result = notify_event(off, "campaign_complete", "done")

    assert result.sent is False
    assert "campaign_complete" in result.detail


def test_an_unknown_event_is_not_sent(monkeypatch):
    monkeypatch.setattr(notifications.httpx, "post", lambda *a, **k: _Response(200))
    assert notify_event(settings_with(), "something_else", "x").sent is False
