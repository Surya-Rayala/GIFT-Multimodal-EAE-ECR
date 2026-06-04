"""Session loading endpoints — /session and /resolve_video.

Backed by the v2 loader at ``src/utils/analysis_viewer/backend/loader.py``.
"""
from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query

from ..loader import SessionLoadError, collect_whitelist_paths, load_session
from ..security import WHITELIST

router = APIRouter()


@router.get("/session")
def get_session(path: str = Query(..., description="Run folder (with RunInfo.json) or {basename}_Analysis.json")) -> Dict[str, Any]:
    try:
        data = load_session(path)
    except SessionLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    json_path = data["session_json_path"]
    whitelist = collect_whitelist_paths(data, json_path)
    WHITELIST.register(json_path, whitelist, root_dir=os.path.dirname(json_path))
    return data


@router.get("/resolve_video")
def resolve_video(path: str = Query(..., description="Run folder or session JSON")) -> Dict[str, Any]:
    try:
        data = load_session(path)
    except SessionLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    resolved = data.get("resolved_video_path")
    return {
        "session_json_path": data.get("session_json_path"),
        "resolved_video_path": resolved,
        "exists": bool(resolved and os.path.exists(resolved)),
    }


def _require_existing_file(path: str) -> str:
    if not path:
        raise HTTPException(status_code=400, detail="path query parameter is required")
    abs_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail=f"File not found: {abs_path}")
    return abs_path
