"""Artifact saving endpoint.

Writes the four mapper output files to the user-chosen project directory.
The mapper UI POSTs the current state for the artifact it wants to save;
the backend formats the lines and writes them to disk. The four kinds match
exactly what the engine reads from per-room configs (see ``input/*.json``).
"""
from __future__ import annotations

import os
from typing import List, Literal, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


ArtifactKind = Literal["mapping", "entry_polygons", "pod_points", "room_boundary"]

_FILENAME_BY_KIND = {
    "mapping": "{name}_mapping.txt",
    "entry_polygons": "{name}_entry_polygons.txt",
    "pod_points": "{name}_POD_points.txt",
    "room_boundary": "{name}_room_boundary.txt",
}


class MappingPair(BaseModel):
    fx: int
    fy: int
    mx: int
    my: int


class SaveRequest(BaseModel):
    output_dir: str = Field(..., description="Absolute output directory")
    project_name: str = Field(..., description="Project name; used as filename prefix")
    kind: ArtifactKind
    mapping: List[MappingPair] | None = None
    polygons: List[List[Tuple[int, int]]] | None = None
    points: List[Tuple[int, int]] | None = None


def _validate_output_dir(p: str) -> str:
    if not p:
        raise HTTPException(status_code=400, detail="output_dir is required")
    abs_p = os.path.realpath(os.path.expanduser(p))
    if not os.path.isdir(abs_p):
        raise HTTPException(
            status_code=404, detail=f"Output directory does not exist: {abs_p}"
        )
    return abs_p


def _sanitize_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return "project"
    # Replace any character that's awkward in a filename; keep [A-Za-z0-9_-]
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)
    return safe or "project"


@router.post("/save_artifact")
def save_artifact(req: SaveRequest) -> dict:
    out_dir = _validate_output_dir(req.output_dir)
    name = _sanitize_name(req.project_name)
    out_path = os.path.join(out_dir, _FILENAME_BY_KIND[req.kind].format(name=name))

    if req.kind == "mapping":
        if not req.mapping:
            raise HTTPException(status_code=400, detail="mapping is empty")
        lines = [f"{p.fx}, {p.fy}, {p.mx}, {p.my}" for p in req.mapping]

    elif req.kind == "entry_polygons":
        if not req.polygons:
            raise HTTPException(status_code=400, detail="polygons is empty")
        lines = [
            ", ".join(f"{x},{y}" for (x, y) in poly)
            for poly in req.polygons
            if len(poly) >= 3
        ]
        if not lines:
            raise HTTPException(status_code=400, detail="all polygons have <3 points")

    elif req.kind == "pod_points":
        if not req.points:
            raise HTTPException(status_code=400, detail="points is empty")
        lines = [f"{x}, {y}" for (x, y) in req.points]

    elif req.kind == "room_boundary":
        if not req.polygons or not req.polygons[0]:
            raise HTTPException(status_code=400, detail="boundary polygon is empty")
        first = req.polygons[0]
        if len(first) < 3:
            raise HTTPException(status_code=400, detail="boundary needs ≥3 points")
        lines = [", ".join(f"{x},{y}" for (x, y) in first)]

    else:  # pragma: no cover - Literal exhaustiveness
        raise HTTPException(status_code=400, detail=f"unknown kind {req.kind!r}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return {"ok": True, "path": out_path, "lines": len(lines)}
