"""Transcription and drill-window sidecar endpoints."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from ..config import enforce_under_root

router = APIRouter()


@router.get("/transcription")
def get_transcription(path: str = Query(..., description="Path to {basename}_Transcription.json")) -> Dict[str, Any]:
    return _load_json(path)


@router.get("/drillwindow")
def get_drillwindow(
    path: str = Query(..., description="Path to {basename}_DrillWindow.json or session JSON containing the embedded drill_window block"),
) -> Optional[Dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, dict) and "drill_window" in payload:
        return payload["drill_window"]
    return payload


def _load_json(path: str) -> Dict[str, Any]:
    # Restricted to the configured outputs root in server mode (no-op on desktop).
    abs_path = enforce_under_root(path)
    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail=f"File not found: {abs_path}")
    try:
        with open(abs_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc
