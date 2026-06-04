"""Media streaming endpoints — /video and /image, both with HTTP Range support.

Range support is essential for HTML5 <video> seeking on every platform.
"""
from __future__ import annotations

import mimetypes
import os
import re
from pathlib import Path
from typing import AsyncIterator, Optional, Tuple

import aiofiles
from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..security import WHITELIST

router = APIRouter()

CHUNK_SIZE = 1024 * 1024  # 1 MiB


@router.get("/video")
async def get_video(
    path: str = Query(..., description="Absolute path to the video file"),
    session: str = Query(..., description="Path to the session JSON that authorizes this file"),
    range_header: Optional[str] = Header(None, alias="Range"),
):
    abs_path = WHITELIST.check(session, path)
    return await _stream_file(abs_path, range_header, default_mime="video/mp4")


@router.get("/image")
async def get_image(
    path: str = Query(..., description="Absolute path to the image file"),
    session: str = Query(..., description="Path to the session JSON that authorizes this file"),
):
    abs_path = WHITELIST.check(session, path)
    return await _stream_file(abs_path, range_header=None, default_mime="image/jpeg")


async def _stream_file(abs_path: str, range_header: Optional[str], default_mime: str) -> StreamingResponse:
    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail=f"File not found: {abs_path}")

    file_size = os.path.getsize(abs_path)
    mime, _ = mimetypes.guess_type(abs_path)
    mime = mime or default_mime

    start, end = _parse_range(range_header, file_size) if range_header else (0, file_size - 1)
    length = end - start + 1

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Content-Type": mime,
    }
    status = 200
    if range_header:
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        status = 206

    return StreamingResponse(
        _iter_range(abs_path, start, length),
        status_code=status,
        headers=headers,
        media_type=mime,
    )


_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def _parse_range(header: str, file_size: int) -> Tuple[int, int]:
    match = _RANGE_RE.fullmatch(header.strip())
    if not match:
        raise HTTPException(status_code=400, detail=f"Invalid Range header: {header!r}")
    start_str, end_str = match.groups()
    if start_str == "" and end_str == "":
        raise HTTPException(status_code=400, detail="Range must specify at least one bound")
    if start_str == "":
        suffix = int(end_str)
        start = max(0, file_size - suffix)
        end = file_size - 1
    else:
        start = int(start_str)
        end = int(end_str) if end_str else file_size - 1
    if start > end or end >= file_size:
        raise HTTPException(
            status_code=416,
            detail=f"Range not satisfiable: {start}-{end}/{file_size}",
            headers={"Content-Range": f"bytes */{file_size}"},
        )
    return start, end


async def _iter_range(abs_path: str, start: int, length: int) -> AsyncIterator[bytes]:
    remaining = length
    async with aiofiles.open(abs_path, "rb") as fh:
        await fh.seek(start)
        while remaining > 0:
            chunk = await fh.read(min(CHUNK_SIZE, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk
