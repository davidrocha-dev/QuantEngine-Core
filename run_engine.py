"""Frozen-binary entry point for the engine — design.md B.9.

PyInstaller bundles *this* file (not `app/main.py`, which is an ASGI module
with no entry point of its own: running the packaged binary would have
imported it, defined the FastAPI app and exited without ever serving).

`freeze_support()` comes before any project import on purpose. Phases 0-3
evaluate candidates across a ProcessPoolExecutor, and a frozen build starts
each worker by re-running this executable — without this call the worker
would boot a second copy of the whole engine instead of running its task.
"""

from __future__ import annotations

import multiprocessing
import os


def main() -> None:
    multiprocessing.freeze_support()

    import uvicorn

    # design.md B.2 — localhost only; the Electron main process is the
    # security boundary and spawns this binary with no arguments.
    host = os.environ.get("QUANTENGINE_HOST", "127.0.0.1")
    port = int(os.environ.get("QUANTENGINE_PORT", "7831"))
    uvicorn.run("app.main:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
