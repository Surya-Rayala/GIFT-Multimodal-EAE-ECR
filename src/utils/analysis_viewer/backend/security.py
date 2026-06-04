"""Per-session file whitelist.

The sidecar runs on 127.0.0.1, but defense-in-depth: we don't expose arbitrary
filesystem reads. When a session JSON is loaded, its artifact paths + sidecars
+ resolved video path are recorded. Subsequent /video, /image, /transcription,
/drillwindow requests must reference one of those whitelisted files (compared
via inode equality so case-insensitive filesystems and symlinks work) or a
file under the session JSON's directory tree.
"""
from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Dict, List

from fastapi import HTTPException


class SessionWhitelist:
    """Maps a session JSON path to the list of files that session is allowed to expose."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._allowed: Dict[str, List[str]] = {}
        self._roots: Dict[str, str] = {}

    def register(self, session_path: str, paths: List[str], root_dir: str) -> None:
        session_key = _key(session_path)
        resolved_root = os.path.realpath(root_dir)
        existing: List[str] = []
        seen: set[str] = set()
        for p in paths + [session_path]:
            if not p:
                continue
            real = os.path.realpath(p)
            if real not in seen:
                seen.add(real)
                existing.append(real)
        with self._lock:
            self._allowed[session_key] = existing
            self._roots[session_key] = resolved_root

    def extend(self, session_path: str, additional_paths: List[str]) -> None:
        """Add extra allowed paths to an already-registered session.

        Used by ``/compare`` to whitelist the per-comparison image files that
        live outside the session root (in a process-lifetime temp directory).
        Silently no-ops if the session hasn't been registered yet.
        """
        session_key = _key(session_path)
        with self._lock:
            existing = self._allowed.get(session_key)
            if existing is None:
                return
            seen: set[str] = set(existing)
            for p in additional_paths:
                if not p:
                    continue
                try:
                    real = os.path.realpath(p)
                except OSError:
                    continue
                if real not in seen:
                    seen.add(real)
                    existing.append(real)
            self._allowed[session_key] = existing

    def check(self, session_path: str, requested_path: str) -> str:
        session_key = _key(session_path)
        with self._lock:
            allowed = self._allowed.get(session_key)
            root = self._roots.get(session_key)
        if allowed is None:
            raise HTTPException(status_code=400, detail="Session not loaded; call /session first")

        resolved_request = os.path.realpath(requested_path)
        if not os.path.exists(resolved_request):
            raise HTTPException(status_code=404, detail=f"File not found: {resolved_request}")

        for entry in allowed:
            if _same_file_or_path(entry, resolved_request):
                return resolved_request

        if root and _is_under(resolved_request, root):
            return resolved_request

        raise HTTPException(status_code=403, detail="Path not in session whitelist")


def _key(session_path: str) -> str:
    """Canonical key for a session: realpath, then case-folded so macOS/Windows
    case-insensitive filesystems hash to the same bucket."""
    return os.path.normcase(os.path.realpath(session_path))


def _same_file_or_path(a: str, b: str) -> bool:
    if os.path.normcase(a) == os.path.normcase(b):
        return True
    try:
        return os.path.samefile(a, b)
    except (OSError, ValueError):
        return False


def _is_under(child: str, parent: str) -> bool:
    try:
        rel = os.path.relpath(child, parent)
    except ValueError:
        return False
    return not rel.startswith("..") and not os.path.isabs(rel)


WHITELIST = SessionWhitelist()
