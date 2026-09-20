"""Rotating local logs and opt-in crash reporting — REQ-NFR-2.

When the engine dies, the app sees a child process exit and nothing else.
These logs are what turns "it stopped working" into something diagnosable,
so they live next to the Vault in AppData, rotate rather than grow without
bound, and capture the traceback of anything that escapes.

Crash telemetry is opt-in and off by default (REQ-NFR-2.2). Sentry is
imported only when it is both enabled and installed, so the engine has no
hard dependency on it and a build without it behaves normally.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

LOG_FILENAME = "engine.log"
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3
FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def log_dir(data_dir: Path) -> Path:
    directory = data_dir / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def configure_logging(data_dir: Path, level: int = logging.INFO) -> Path:
    """Logs to a rotating file and to stdout (which Electron captures)."""
    path = log_dir(data_dir) / LOG_FILENAME

    root = logging.getLogger()
    root.setLevel(level)
    # Re-running this (tests, reloads) must not stack duplicate handlers.
    for handler in list(root.handlers):
        if getattr(handler, "_quantengine", False):
            root.removeHandler(handler)

    file_handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(FORMAT))
    file_handler._quantengine = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(FORMAT))
    stream_handler._quantengine = True  # type: ignore[attr-defined]
    root.addHandler(stream_handler)

    _install_excepthook()
    return path


def _install_excepthook() -> None:
    """Anything that escapes still reaches the log before the process dies."""
    previous = sys.excepthook

    def handler(exc_type, exc_value, exc_traceback):
        if not issubclass(exc_type, KeyboardInterrupt):
            logging.getLogger("quantengine").critical(
                "Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback)
            )
        previous(exc_type, exc_value, exc_traceback)

    sys.excepthook = handler


def configure_crash_reporting(enabled: bool) -> bool:
    """REQ-NFR-2.2 — opt-in, anonymised, and only when a DSN exists.

    Returns whether reporting is actually active, so the caller can tell
    the user the truth rather than assuming the toggle did something.
    """
    if not enabled:
        return False

    dsn = os.environ.get("QUANTENGINE_SENTRY_DSN", "").strip()
    if not dsn:
        logging.getLogger("quantengine").info(
            "Crash reporting is enabled but no DSN is configured — nothing will be sent."
        )
        return False

    try:
        import sentry_sdk
    except ImportError:
        logging.getLogger("quantengine").info("Crash reporting is enabled but sentry-sdk is not installed.")
        return False

    sentry_sdk.init(
        dsn=dsn,
        # REQ-NFR-2.2 / design.md B.3 — telemetry is anonymous: no account
        # identifiers, no request bodies, nothing about the user's trades.
        send_default_pii=False,
        traces_sample_rate=0.0,
    )
    return True
