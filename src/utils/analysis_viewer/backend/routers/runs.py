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

from ..config import enforce_under_root, outputs_root as configured_root

router = APIRouter()


@router.get("/runs")
def get_runs(
    outputs_root: str = Query(
        default="",
        description="Absolute path to the outputs root directory. Optional in server "
        "mode — defaults to the configured outputs root.",
    ),
) -> List[Dict[str, Any]]:
    # In server mode the configured root is the default and the ceiling: a passed
    # root must resolve under it (enforce_under_root). In desktop mode the query
    # param is required and any path is accepted.
    root = outputs_root or (configured_root() or "")
    if not root:
        raise HTTPException(status_code=400, detail="outputs_root query parameter is required")
    abs_root = enforce_under_root(root)
    if not os.path.isdir(abs_root):
        raise HTTPException(status_code=404, detail=f"Outputs root not found: {abs_root}")
    return list_runs(abs_root)
