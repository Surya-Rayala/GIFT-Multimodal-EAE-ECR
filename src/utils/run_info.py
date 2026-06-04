"""Per-run manifest (``RunInfo.json``).

Each processing-engine run writes its outputs into a dedicated folder under the
outputs root. The manifest at the top of that folder labels the run (role,
title, video basename, source vmeta, creation time) and tells consumers which
file inside the folder is the schema-v2 analysis JSON.

The manifest is the single source of truth that lets the analysis viewer, the
expert-compare endpoint, and the ``/runs`` listing discover and disambiguate
runs without filename heuristics.

Schema:

    {
      "schema_version": "1.0",
      "run_id": "expert_Test_Video_20260521_143022",
      "role": "expert",          # one of {"expert", "trainee"}
      "title": "Test_Video",     # human-readable label from vmeta
      "video_basename": "3-TrimmedV2",
      "vmeta_path": "/abs/path/to/test.vmeta.xml",
      "created_at": "2026-05-21T14:30:22Z",
      "analysis_file": "3-TrimmedV2_Analysis.json"   # relative to run folder
    }
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from typing import Any, Dict, List, Optional


_MANIFEST_NAME = "RunInfo.json"
_SCHEMA_VERSION = "1.0"


def manifest_path(run_dir: str) -> str:
    """Canonical path for a run's manifest."""
    return os.path.join(run_dir, _MANIFEST_NAME)


def slugify(text: str, max_len: int = 40) -> str:
    """Filesystem-safe slug: keep [A-Za-z0-9_-], collapse others to underscore."""
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", text or "")
    s = s.strip("_") or "untitled"
    return s[:max_len]


def make_run_id(role: str, title: str, when: Optional[_dt.datetime] = None) -> str:
    """Build the run-folder name: ``{role}_{slug(title)}_{YYYYMMDD_HHMMSS}``."""
    when = when or _dt.datetime.now(_dt.timezone.utc)
    return f"{role}_{slugify(title)}_{when.strftime('%Y%m%d_%H%M%S')}"


def save_run_info(
    run_dir: str,
    *,
    run_id: str,
    role: str,
    title: str,
    video_basename: str,
    vmeta_path: str,
    analysis_file: str,
    when: Optional[_dt.datetime] = None,
) -> str:
    """Write ``RunInfo.json`` at the top of ``run_dir``. Returns the written path."""
    when = when or _dt.datetime.now(_dt.timezone.utc)
    payload: Dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "run_id": run_id,
        "role": role,
        "title": title,
        "video_basename": video_basename,
        "vmeta_path": os.path.abspath(vmeta_path) if vmeta_path else None,
        "created_at": when.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "analysis_file": analysis_file,
    }
    os.makedirs(run_dir, exist_ok=True)
    path = manifest_path(run_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_run_info(run_dir: str) -> Optional[Dict[str, Any]]:
    """Read ``RunInfo.json`` from ``run_dir``. ``None`` on missing/malformed."""
    if not run_dir or not os.path.isdir(run_dir):
        return None
    path = manifest_path(run_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def list_runs(outputs_root: str) -> List[Dict[str, Any]]:
    """List every run folder under ``outputs_root`` that has a valid ``RunInfo.json``.

    Each entry: ``{run_id, title, role, path, created_at, analysis_file_exists}``.
    Subdirs without a manifest are silently skipped (filters out legacy flat
    outputs and any junk).
    """
    if not outputs_root or not os.path.isdir(outputs_root):
        return []
    out: List[Dict[str, Any]] = []
    for entry in sorted(os.listdir(outputs_root)):
        run_dir = os.path.join(outputs_root, entry)
        if not os.path.isdir(run_dir):
            continue
        info = load_run_info(run_dir)
        if info is None:
            continue
        analysis_rel = info.get("analysis_file") or ""
        analysis_abs = os.path.join(run_dir, analysis_rel) if analysis_rel else ""
        out.append(
            {
                "run_id": info.get("run_id") or entry,
                "title": info.get("title") or entry,
                "role": info.get("role") or "trainee",
                "path": os.path.abspath(run_dir),
                "created_at": info.get("created_at"),
                "analysis_file_exists": bool(analysis_abs) and os.path.isfile(analysis_abs),
            }
        )
    return out
