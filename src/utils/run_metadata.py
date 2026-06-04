"""Per-session run-metadata sidecar.

Live processing in :class:`ProcessingEngine` resolves fps (and other runtime
values) from the actual video file. That value is written here as a small JSON
sidecar alongside the other per-session artifacts (``*_PositionCache.txt``,
``*_TrackerOutput.json`` etc.).

Offline expert comparison opens the cached caches without ever touching the
video, so without this sidecar the comparison would fall back to whatever
``frame_rate`` happens to be in the map JSON — which can drift from the actual
video. Loading the sidecar in ``compare_expert`` restores a single source of
truth: the fps the live run actually used.

Schema (keys are best-effort; missing fields are tolerated):

    {
      "schema_version": 1,
      "fps": 60.0,                       # video-derived frame rate
      "video_path": "/abs/path/to.mp4",  # source video (may be transcoded)
      "video_basename": "3-Trimmed",     # basename used for sidecar files
      "produced_at": "2026-05-08T...",   # ISO8601 UTC timestamp
      "engine_version": "schema_v1"      # for future schema migrations
    }
"""

from __future__ import annotations

import datetime as _dt
import glob
import json
import os
from typing import Any, Dict, Optional


_SIDECAR_SUFFIX = "_RunMetadata.json"
_SCHEMA_VERSION = 1


def sidecar_path(output_dir: str, video_basename: str) -> str:
    """Canonical path for this session's run-metadata sidecar."""
    return os.path.join(output_dir, f"{video_basename}{_SIDECAR_SUFFIX}")


def save_run_metadata(
    output_dir: str,
    video_basename: str,
    *,
    fps: float,
    video_path: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Write the run-metadata sidecar. Returns the path written.

    Errors are not raised — a missing sidecar is recoverable downstream
    (offline comparison falls back to the JSON's ``frame_rate``). We only
    log a warning by raising; callers should wrap if they need silence.
    """
    if not output_dir or not video_basename:
        return ""
    payload: Dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "fps": float(fps) if fps is not None else None,
        "video_path": str(video_path) if video_path else None,
        "video_basename": str(video_basename),
        "produced_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        payload.update({k: v for k, v in extra.items() if k not in payload})

    path = sidecar_path(output_dir, video_basename)
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_run_metadata(folder: str) -> Optional[Dict[str, Any]]:
    """Read the most-recently-modified ``*_RunMetadata.json`` from ``folder``.

    Returns the parsed dict, or ``None`` if no sidecar exists or the file is
    unreadable / malformed.
    """
    if not folder or not os.path.isdir(folder):
        return None
    matches = glob.glob(os.path.join(folder, f"*{_SIDECAR_SUFFIX}"))
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    try:
        with open(matches[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except (OSError, json.JSONDecodeError):
        return None


def resolve_fps_from_metadata(folder: str, fallback: Optional[float] = None) -> Optional[float]:
    """Convenience: look up fps from the sidecar; return ``fallback`` if absent.

    The fallback is returned without modification — caller decides whether to
    treat ``None`` as an error or substitute a default.
    """
    md = load_run_metadata(folder)
    if md is None:
        return fallback
    fps = md.get("fps")
    try:
        return float(fps) if fps is not None else fallback
    except (TypeError, ValueError):
        return fallback
