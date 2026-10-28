"""Who may call the engine.

The engine listens on 127.0.0.1, which keeps other machines out and
nothing else. Any web page open in any browser on this machine can send it
requests: CORS only stops the page reading the answer, not the request
being carried out, so a plain cross-origin POST reached /deploy,
/incubation/{id}/graduate or /vault/purge and did what it said. And with
DNS rebinding a page can make itself same-origin with 127.0.0.1 and read
the answers too.

Two checks close that:

  - the Host header must name the loopback interface, which a rebound
    domain cannot (it carries its own name);
  - every request must carry the per-launch token the Electron shell
    generated and handed to both the engine (QUANTENGINE_API_TOKEN) and
    its own renderer. A web page never sees it, and a custom header
    cannot be sent cross-origin without a preflight this server does not
    grant. The event socket, which cannot carry headers from a browser,
    takes it as the `token` query parameter.

Without QUANTENGINE_API_TOKEN (the test suite, a bare `uvicorn` in a
terminal) the token check is off and says so once at startup; the Host
check is not.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
from urllib.parse import parse_qs


TOKEN_ENV = "QUANTENGINE_API_TOKEN"
TOKEN_HEADER = b"x-quantengine-token"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
# Starlette's TestClient identifies itself this way; only a process that
# imports this code can send it.
_TEST_HOSTS = {"testserver"}


def configured_token() -> str | None:
    return os.environ.get(TOKEN_ENV) or None


def _host_allowed(host: str | None) -> bool:
    if not host:
        return False
    name = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
    return name.lower() in LOOPBACK_HOSTS or name.lower() in _TEST_HOSTS


class LocalAccessMiddleware:
    """Pure ASGI, so it covers the websocket as well as HTTP."""

    def __init__(self, app) -> None:
        self.app = app
        if configured_token() is None:
            logging.getLogger("quantengine").warning(
                "%s is not set: the engine API accepts any local caller. The desktop app always sets it.",
                TOKEN_ENV,
            )

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)

        headers = dict(scope.get("headers") or [])
        host = headers.get(b"host", b"").decode("latin-1")
        if not _host_allowed(host):
            return await self._refuse(scope, send, "Host not allowed")

        token = configured_token()
        # A CORS preflight carries no custom headers by design; it only
        # asks permission, and CORSMiddleware answers it.
        if scope["type"] == "http" and scope.get("method") == "OPTIONS":
            return await self.app(scope, receive, send)
        if token is None:
            # The packaged engine is only ever started by the app, which
            # always sets a token. One running without is someone else.
            return await self.app(scope, receive, send)

        presented = headers.get(TOKEN_HEADER, b"").decode("latin-1")
        if not presented and scope["type"] == "websocket":
            query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
            presented = (query.get("token") or [""])[0]
        if not hmac.compare_digest(presented, token):
            return await self._refuse(scope, send, "Missing or wrong engine token")
        return await self.app(scope, receive, send)

    @staticmethod
    async def _refuse(scope, send, reason: str, status: int = 403) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008, "reason": reason})
            return
        body = json.dumps({"detail": reason}).encode()
        await send({
            "type": "http.response.start", "status": status,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        })
        await send({"type": "http.response.body", "body": body})
