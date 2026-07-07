"""Alglory launcher: `python -m alglory [--port 8777] [--no-browser]`."""

from __future__ import annotations

import argparse
import threading
import webbrowser

import uvicorn

from alglory.config import AppConfig
from alglory.server.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="alglory", description="Alglory bot factory")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    app = create_app(AppConfig.default())

    if not args.no_browser:
        url = f"http://{args.host}:{args.port}"
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
