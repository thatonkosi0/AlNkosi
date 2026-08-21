"""Alglory launcher: `python -m alglory [--port 8777] [--no-browser]`."""

from __future__ import annotations

import argparse
import secrets
import sys
import threading
import webbrowser

import uvicorn

from alglory.config import AppConfig
from alglory.server.app import create_app

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _check_host(host: str, allow_remote: bool) -> None:
    """Refuse a non-loopback bind unless the user explicitly opts in.

    Every /api/* route is token-gated (see alglory.server.auth), but binding
    to a LAN/public address is still a much bigger blast radius than the
    loopback default ALGLORY ships with -- require a deliberate flag, not
    just a --host typo, before exposing it.
    """
    if host not in _LOOPBACK_HOSTS and not allow_remote:
        raise SystemExit(
            f"Refusing to bind to {host!r}: that is not a loopback address. "
            "ALGLORY places real trading commands behind this API. If you really "
            "want it reachable from outside this machine, pass --allow-remote "
            "and make sure the network it's on is trusted."
        )


def main() -> None:
    parser = argparse.ArgumentParser(prog="alglory", description="Alglory bot factory")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--allow-remote", action="store_true",
        help="allow binding to a non-loopback --host (off by default; real risk of exposure)",
    )
    args = parser.parse_args()

    _check_host(args.host, args.allow_remote)

    token = secrets.token_urlsafe(24)
    app = create_app(AppConfig.default(), token=token)
    url = f"http://{args.host}:{args.port}/?token={token}"
    print(f"ALGLORY listening on {url}", file=sys.stderr)

    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
