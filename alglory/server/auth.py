"""Per-run bearer-token auth for the local API surface.

ALGLORY is a local-first, single-user tool with no login flow, but every
/api/* route (including arming live trading) was reachable with zero
credentials -- any other page open in the same browser could POST to it
(CSRF), and nothing stopped a --host 0.0.0.0 run from being wide open on
a LAN. A random token generated per server run closes both gaps without
adding accounts: the browser only ever sees it because __main__ puts it
in the URL it opens.
"""

from __future__ import annotations

import secrets
from urllib.parse import parse_qsl

from starlette.responses import JSONResponse

_PROTECTED_PREFIXES = ("/api/",)
_PROTECTED_EXACT = ("/ws/events",)


def _is_protected(path: str) -> bool:
    return path in _PROTECTED_EXACT or any(path.startswith(p) for p in _PROTECTED_PREFIXES)


class TokenAuthMiddleware:
    """Raw ASGI middleware: requires X-Alglory-Token (HTTP) or ?token= (either)."""

    def __init__(self, app, *, token: str) -> None:
        self.app = app
        self.token = token

    @staticmethod
    def _supplied(scope) -> bytes:
        """The raw token bytes the caller sent, or b"" — never raises.

        Everything read here is attacker-controlled and can be *any* byte
        string, so this stays in bytes end to end. Decoding as UTF-8 would
        raise UnicodeDecodeError on invalid bytes (e.g. ?token=%FF), and
        secrets.compare_digest raises TypeError on a non-ASCII str (e.g.
        ?token=%C3%A9). This is raw ASGI middleware sitting above FastAPI's
        exception handling, so either would escape as an unhandled 500 from
        the one module whose job is gatekeeping live trading. latin-1 is a
        byte-exact round trip for every possible input, so no token — however
        malformed — can do anything but fail closed with a 401.
        """
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            supplied = headers.get(b"x-alglory-token", b"")
            if supplied:
                return supplied
        query = (scope.get("query_string") or b"").decode("latin-1")
        for key, value in parse_qsl(query, encoding="latin-1"):
            if key == "token":
                return value.encode("latin-1")
        return b""

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or not _is_protected(scope["path"]):
            await self.app(scope, receive, send)
            return

        if not secrets.compare_digest(self._supplied(scope), self.token.encode("utf-8")):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 4401})
            else:
                response = JSONResponse(
                    {"detail": "Missing or invalid X-Alglory-Token."}, status_code=401
                )
                await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
