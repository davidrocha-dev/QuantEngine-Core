"""Notification channels — REQ-SET-7.

The OS notification is the renderer's job (Electron owns that API). The
Telegram channel belongs here instead: a campaign finishing is an engine
event, and the whole point of the channel is that it reaches the user when
the app is not in front of them — including when the window is closed.

Delivery is best-effort and never raises into the caller: a campaign that
finished successfully must not be reported as failed because a chat id was
wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"
TIMEOUT_SECONDS = 10.0


@dataclass
class DeliveryResult:
    sent: bool
    detail: str


def send_telegram(bot_token: str, chat_id: str, text: str) -> DeliveryResult:
    """Sends one message. Returns why it failed rather than raising."""
    if not bot_token or not chat_id:
        return DeliveryResult(False, "Telegram needs both a bot token and a chat id.")

    try:
        response = httpx.post(
            f"{TELEGRAM_API}/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
            timeout=TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        logger.warning("Telegram delivery failed: %s", exc)
        return DeliveryResult(False, f"Could not reach Telegram: {exc.__class__.__name__}")

    if response.status_code == 200:
        return DeliveryResult(True, "Message sent.")

    # Telegram explains rejections in the body ("chat not found", "Unauthorized"),
    # which is far more useful to the user than the status code alone.
    description = ""
    try:
        description = response.json().get("description", "")
    except ValueError:
        pass
    return DeliveryResult(False, description or f"Telegram rejected the message ({response.status_code}).")


def notify_event(settings: dict, event: str, text: str) -> DeliveryResult:
    """Sends `text` on the Telegram channel if this event is enabled for it
    (REQ-SET-7.4 — per event, per channel)."""
    notifications = settings.get("notifications", {})
    if not notifications.get("telegram_enabled"):
        return DeliveryResult(False, "Telegram channel is off.")
    if not (notifications.get("events", {}).get(event, {}) or {}).get("telegram"):
        return DeliveryResult(False, f"Telegram is off for {event}.")

    return send_telegram(notifications.get("telegram_bot_token", ""), notifications.get("telegram_chat_id", ""), text)
