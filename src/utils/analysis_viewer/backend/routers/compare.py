"""Comparison endpoint — ``GET /compare``.

Generates a side-by-side comparison of two run folders for a single metric.
Artifacts are written into a process-lifetime temp directory (cleaned up at
sidecar exit) so the run folders themselves are never touched. The same
pairing+metric is cached on disk so re-clicking a metric is cheap.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

from src.processing_engine import ProcessingEngine
from src.utils.run_info import load_run_info

from ..config import enforce_under_root
from ..security import WHITELIST

router = APIRouter()


# Process-lifetime temp directory for all comparison artifacts. Created lazily.
# Cleaned up at sidecar shutdown via atexit.
_COMPARE_TMP_DIR: Path | None = None
_CLEANUP_REGISTERED = False


def _get_temp_root() -> Path:
    global _COMPARE_TMP_DIR, _CLEANUP_REGISTERED
    if _COMPARE_TMP_DIR is None:
        _COMPARE_TMP_DIR = Path(tempfile.gettempdir()) / f"gift_viewer_{os.getpid()}"
        _COMPARE_TMP_DIR.mkdir(parents=True, exist_ok=True)
    if not _CLEANUP_REGISTERED:
        atexit.register(lambda: shutil.rmtree(_COMPARE_TMP_DIR, ignore_errors=True))
        _CLEANUP_REGISTERED = True
    return _COMPARE_TMP_DIR


def _pair_subdir(current_run_id: str, other_run_id: str, metric_id: str) -> Path:
    """Deterministic per-pairing subdir for caching."""
    safe = f"{current_run_id}__{other_run_id}__{metric_id}"
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in safe)
    return _get_temp_root() / safe


@router.get("/compare")
def compare(
    current_run: str = Query(..., description="Absolute path to current run folder"),
    other_run: str = Query(..., description="Absolute path to the run to compare against"),
    metric_id: str = Query(..., description="Metric identifier (upper-snake, e.g. ENTRANCE_VECTORS)"),
    session: str = Query(..., description="Path to the current session's Analysis.json (whitelist key)"),
) -> Dict[str, Any]:
    # Both run folders must be under the served outputs root in server mode
    # (no-op on desktop). Compare can thus reach any run beside the open one.
    cur = enforce_under_root(current_run)
    oth = enforce_under_root(other_run)
    if not os.path.isdir(cur):
        raise HTTPException(status_code=404, detail=f"Current run folder not found: {cur}")
    if not os.path.isdir(oth):
        raise HTTPException(status_code=404, detail=f"Other run folder not found: {oth}")

    cur_info = load_run_info(cur)
    oth_info = load_run_info(oth)
    if cur_info is None:
        raise HTTPException(status_code=400, detail=f"Current run has no RunInfo.json: {cur}")
    if oth_info is None:
        raise HTTPException(status_code=400, detail=f"Other run has no RunInfo.json: {oth}")

    pair_dir = _pair_subdir(cur_info.get("run_id") or os.path.basename(cur),
                             oth_info.get("run_id") or os.path.basename(oth),
                             metric_id)
    pair_dir.mkdir(parents=True, exist_ok=True)

    engine = ProcessingEngine(force_transcode=False)
    raw = engine.compare_expert(metric_id, cur, oth, output_dir=str(pair_dir))
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Comparison engine returned invalid JSON: {exc}") from exc

    extra_paths: List[str] = []
    for viz in payload.get("visualizations") or []:
        p = viz.get("image_path") if isinstance(viz, dict) else None
        if isinstance(p, str) and p:
            extra_paths.append(p)
    if extra_paths:
        WHITELIST.extend(session, extra_paths)

    return payload
