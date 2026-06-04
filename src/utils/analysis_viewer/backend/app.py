"""FastAPI application for the analysis viewer sidecar."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import compare, media, runs, session, sidecars


def create_app() -> FastAPI:
    app = FastAPI(
        title="Analysis Viewer Sidecar",
        version="1.0.0",
        docs_url="/docs",
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "tauri://localhost",
            "https://tauri.localhost",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
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

    return app


app = create_app()
