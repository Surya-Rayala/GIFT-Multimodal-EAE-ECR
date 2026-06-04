"""FastAPI application for the mapper viewer sidecar."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import artifacts, media


def create_app() -> FastAPI:
    app = FastAPI(
        title="Mapper Viewer Sidecar",
        version="1.0.0",
        docs_url="/docs",
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        # Mapper viewer's Vite dev server runs on 5174 (analysis viewer uses
        # 5173). Both must be allowed so the frontend can talk to its sidecar
        # both in dev mode (Vite) and in the bundled Tauri webview.
        allow_origins=[
            "tauri://localhost",
            "https://tauri.localhost",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Range", "Content-Type"],
        expose_headers=["Content-Range", "Content-Length", "Accept-Ranges"],
    )

    app.include_router(media.router)
    app.include_router(artifacts.router)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "service": "mapper-viewer-sidecar"}

    return app


app = create_app()
