"""FastAPI application for the analysis viewer sidecar."""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import config as cfg
from .routers import compare, media, runs, session, sidecars


def create_app() -> FastAPI:
    app = FastAPI(
        title="Analysis Viewer Sidecar",
        version="1.0.0",
        docs_url="/docs",
        redoc_url=None,
    )

    # Desktop (Tauri) origins by default. In `--serve` mode the UI is served
    # same-origin from this backend, so CORS is moot; extra origins can still be
    # allowed via GIFT_VIEWER_ALLOWED_ORIGINS (comma-separated) for dev setups.
    origins = [
        "tauri://localhost",
        "https://tauri.localhost",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    extra = os.environ.get("GIFT_VIEWER_ALLOWED_ORIGINS", "")
    origins += [o.strip() for o in extra.split(",") if o.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["Range", "Content-Type"],
        expose_headers=["Content-Range", "Content-Length", "Accept-Ranges"],
    )

    app.include_router(session.router)
    app.include_router(sidecars.router)
    app.include_router(media.router)
    app.include_router(runs.router)
    app.include_router(compare.router)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "service": "analysis-viewer-sidecar"}

    @app.get("/config")
    def get_config() -> dict:
        """Runtime config the frontend reads at startup. In server mode it learns
        the outputs root so Compare can browse every run under it."""
        root = cfg.outputs_root()
        return {"outputs_root": root, "serve_mode": root is not None}

    # In server mode, serve the built Vue app from the same origin so a single
    # URL works from any device on the network. Mounted LAST so all API routes
    # above take precedence; `html=True` serves index.html at `/`.
    dist = cfg.dist_dir()
    if dist:
        app.mount("/", StaticFiles(directory=dist, html=True), name="static")

    return app


app = create_app()
