"""Server-mode configuration for the analysis viewer backend.

When the viewer is hosted over a network (the launcher's ``--serve`` mode), it
exports ``GIFT_VIEWER_OUTPUTS_ROOT`` so the backend only ever exposes session
folders **under that root** — a remote browser can't deep-link
``?run=/etc/passwd`` to read arbitrary files. In the default desktop (Tauri)
mode the variable is unset and any absolute path is accepted, exactly as before.

``GIFT_VIEWER_DIST`` (optional) points at the built Vue ``dist/`` directory so a
single origin serves both the UI and the API.

Both are read at call time (not import time) so the launcher can set them before
the app is constructed regardless of import order.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import HTTPException

ENV_OUTPUTS_ROOT = "GIFT_VIEWER_OUTPUTS_ROOT"
ENV_DIST = "GIFT_VIEWER_DIST"


def outputs_root() -> Optional[str]:
    """The configured outputs root (realpath), or None in desktop mode."""
    raw = os.environ.get(ENV_OUTPUTS_ROOT, "").strip()
    if not raw:
        return None
    return os.path.realpath(os.path.expanduser(raw))


def serve_mode() -> bool:
    return outputs_root() is not None


def dist_dir() -> Optional[str]:
    raw = os.environ.get(ENV_DIST, "").strip()
    if not raw:
        return None
    p = os.path.realpath(os.path.expanduser(raw))
    return p if os.path.isdir(p) else None


def _is_under(child: str, parent: str) -> bool:
    try:
        rel = os.path.relpath(child, parent)
    except ValueError:
        return False
    return rel == "." or (not rel.startswith("..") and not os.path.isabs(rel))


def enforce_under_root(path: str) -> str:
    """Resolve ``path`` and, when an outputs root is configured, require it to be
    under that root. Returns the resolved absolute path; raises 403 otherwise.

    In desktop mode (no configured root) this is just a resolve — any path is
    accepted, preserving the previous behaviour.
    """
    resolved = os.path.realpath(os.path.expanduser(path))
    root = outputs_root()
    if root is not None and not _is_under(resolved, root):
        raise HTTPException(
            status_code=403,
            detail="Path is outside the served outputs root.",
        )
    return resolved
