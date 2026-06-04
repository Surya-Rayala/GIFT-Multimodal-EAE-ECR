"""Media endpoints for the scaled point viewer.

- ``GET /image`` — stream a static image (the room map).

Lightweight whitelist: we resolve all paths to absolute realpaths and require
them to point at an existing file with a reasonable extension. The Tauri shell
already restricts the user to picking files via the native dialog, so this is
defence-in-depth rather than a hard security boundary.
"""
from __future__ import annotations

import mimetypes
import os

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

router = APIRouter()

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _resolve_file(path: str, allowed_exts: set[str]) -> str:
    if not path:
        raise HTTPException(status_code=400, detail="path query parameter is required")
    abs_path = os.path.realpath(os.path.expanduser(path))
    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail=f"File not found: {abs_path}")
    ext = os.path.splitext(abs_path)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension {ext!r}. Expected one of {sorted(allowed_exts)}.",
        )
    return abs_path


@router.get("/image")
def get_image(path: str = Query(..., description="Absolute path to an image file")):
    abs_path = _resolve_file(path, IMAGE_EXTS)
    mime, _ = mimetypes.guess_type(abs_path)
    mime = mime or "image/png"

    def _iter():
        with open(abs_path, "rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    return
                yield chunk

    return StreamingResponse(_iter(), media_type=mime)
