"""uvicorn entrypoint for the scaled point viewer sidecar.

Prints ``PORT=<n>`` to stdout once the server is listening. The Tauri shell
reads this line to discover the bound port (random by default).
"""
from __future__ import annotations

import argparse
import asyncio
import socket
import sys

import uvicorn

from .app import app


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaled point viewer sidecar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="0 = pick a free port (default); otherwise bind to the given port",
    )
    parser.add_argument("--log-level", default="warning")
    args = parser.parse_args()

    port = args.port if args.port > 0 else _pick_free_port()

    print(f"PORT={port}", flush=True)

    config = uvicorn.Config(
        app=app,
        host=args.host,
        port=port,
        log_level=args.log_level,
        access_log=False,
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
