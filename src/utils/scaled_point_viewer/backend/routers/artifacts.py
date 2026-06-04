"""Artifact saving endpoint for the scaled point viewer.

A single ``POST /save`` writes three files to the user-chosen project folder:

  * ``<name>_scaled_points.txt``   — one ``x, y`` map-pixel coordinate per line.
    Same format the engine / config builder + mapper viewer consume, so these
    reference points can drive homography setup later.
  * ``<name>_scaled_points.png``   — the map image with the points drawn on it,
    as a visual reference (no walls).
  * ``<name>_scaler_project.json`` — the full editor state (walls, real lengths,
    points, measurements, scale) so a project can be reloaded and edited.

The point pixel coordinates are computed in the frontend (see
``frontend/src/utils/scaling.ts``); the backend just persists them.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import cv2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class SaveRequest(BaseModel):
    output_dir: str = Field(..., description="Absolute output directory")
    project_name: str = Field(..., description="Project name; used as filename prefix")
    map_image_path: str = Field(..., description="Absolute path to the map image")
    points: List[Tuple[float, float]] = Field(..., description="Computed map-pixel points")
    project: Dict[str, Any] = Field(
        default_factory=dict, description="Full editor state for reload/edit"
    )


def _validate_output_dir(p: str) -> str:
    if not p:
        raise HTTPException(status_code=400, detail="output_dir is required")
    abs_p = os.path.realpath(os.path.expanduser(p))
    if not os.path.isdir(abs_p):
        raise HTTPException(status_code=404, detail=f"Output directory does not exist: {abs_p}")
    return abs_p


def _sanitize_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return "project"
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)
    return safe or "project"


def _draw_points_png(map_image_path: str, points: List[Tuple[float, float]], out_path: str) -> None:
    abs_img = os.path.realpath(os.path.expanduser(map_image_path))
    img = cv2.imread(abs_img, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail=f"Could not read map image: {abs_img}")
    h, w = img.shape[:2]
    for (x, y) in points:
        cx = int(round(max(0, min(w - 1, x))))
        cy = int(round(max(0, min(h - 1, y))))
        # White halo then a solid dark dot, so points are visible on any map.
        cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 3, (0, 0, 0), -1, lineType=cv2.LINE_AA)
    if not cv2.imwrite(out_path, img):
        raise HTTPException(status_code=500, detail=f"Failed to write image: {out_path}")


@router.post("/save")
def save(req: SaveRequest) -> dict:
    out_dir = _validate_output_dir(req.output_dir)
    name = _sanitize_name(req.project_name)
    if not req.points:
        raise HTTPException(status_code=400, detail="No points to save.")

    txt_path = os.path.join(out_dir, f"{name}_scaled_points.txt")
    png_path = os.path.join(out_dir, f"{name}_scaled_points.png")
    json_path = os.path.join(out_dir, f"{name}_scaler_project.json")

    # Points file — integer map pixels, "x, y" per line.
    lines = [f"{int(round(x))}, {int(round(y))}" for (x, y) in req.points]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Annotated reference PNG.
    _draw_points_png(req.map_image_path, req.points, png_path)

    # Reloadable project state.
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(req.project, f, indent=2)

    return {
        "ok": True,
        "points_path": txt_path,
        "image_path": png_path,
        "project_path": json_path,
        "count": len(req.points),
    }
