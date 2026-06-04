"""Media endpoints for the mapper viewer.

- ``GET /image``       — stream a static image (map or camera frame).
- ``GET /video_probe`` — return frame count / fps for a video file.
- ``GET /video_frame`` — extract a specific frame from a video and stream it
  back as a PNG. Used by the Setup step's video-frame scrubber.

Lightweight whitelist: we resolve all paths to absolute realpaths and require
them to point at an existing file with a reasonable extension. The Tauri
shell already restricts the user to picking files via the native dialog, so
this is defence-in-depth rather than a hard security boundary.
"""
from __future__ import annotations

import io
import mimetypes
import os
from typing import Optional

import cv2
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

router = APIRouter()

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
# Anything cv2.VideoCapture will reasonably try. The extension check is a
# friendliness filter, not a hard guarantee — if cv2 can't open the file
# after this gate, /video_probe surfaces a clear error to the caller.
VIDEO_EXTS = {
    ".mp4", ".mov", ".avi", ".mkv", ".m4v",
    ".webm", ".wmv", ".flv",
    ".mpg", ".mpeg", ".mts", ".m2ts", ".ts",
    ".3gp",
}


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


@router.get("/video_probe")
def video_probe(path: str = Query(..., description="Absolute path to a video file")) -> dict:
    abs_path = _resolve_file(path, VIDEO_EXTS)
    cap = cv2.VideoCapture(abs_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail=f"Could not open video: {abs_path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if frame_count <= 0:
        raise HTTPException(status_code=400, detail="Video has zero frames.")
    return {
        "path": abs_path,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
    }


@router.get("/video_frame")
def video_frame(
    path: str = Query(..., description="Absolute path to a video file"),
    frame: int = Query(0, ge=0, description="Zero-based frame index"),
) -> StreamingResponse:
    abs_path = _resolve_file(path, VIDEO_EXTS)
    cap = cv2.VideoCapture(abs_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail=f"Could not open video: {abs_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx = max(0, min(total - 1, int(frame)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
    finally:
        cap.release()
    if not ok or bgr is None:
        raise HTTPException(status_code=500, detail=f"Could not read frame {frame} from {abs_path}")

    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode frame as PNG")
    return StreamingResponse(
        io.BytesIO(buf.tobytes()),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )
