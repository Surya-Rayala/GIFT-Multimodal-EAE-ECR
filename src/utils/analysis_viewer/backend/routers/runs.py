"""Run-folder listing endpoint — ``GET /runs?outputs_root=<abs>``.

Lists every subdirectory of ``outputs_root`` that has a valid ``RunInfo.json``
manifest. Folders without a manifest are silently skipped (filters out legacy
flat outputs and junk). Used by the Compare-mode dropdown.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

from src.utils.run_info import list_runs

router = APIRouter()


@router.get("/runs")
def get_runs(outputs_root: str = Query(..., description="Absolute path to the outputs root directory")) -> List[Dict[str, Any]]:
    if not outputs_root:
        raise HTTPException(status_code=400, detail="outputs_root query parameter is required")
    abs_root = os.path.abspath(os.path.expanduser(outputs_root))
    if not os.path.isdir(abs_root):
        raise HTTPException(status_code=404, detail=f"Outputs root not found: {abs_root}")
    return list_runs(abs_root)
