"""Server-mode configuration for the analysis viewer backend.

Access restriction is opt-in. When the launcher's ``--serve`` mode is given
``--outputs-root``, it exports ``GIFT_VIEWER_OUTPUTS_ROOT`` and the backend
only ever exposes session folders **under that root** — a remote browser
can't deep-link ``?run=/etc/passwd`` to read arbitrary files. Without
``--outputs-root`` (and in the desktop/Tauri mode) the variable is unset and
any absolute path is accepted.

``GIFT_VIEWER_SERVE=1`` marks network-serve mode independently of any
restriction. ``GIFT_VIEWER_BROWSE_ROOT`` (optional) is only the folder Compare
lists by default — it restricts nothing.

``GIFT_VIEWER_DIST`` (optional) points at the built Vue ``dist/`` directory so a
single origin serves both the UI and the API.

All are read at call time (not import time) so the launcher can set them before
the app is constructed regardless of import order.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import HTTPException

ENV_OUTPUTS_ROOT = "GIFT_VIEWER_OUTPUTS_ROOT"
ENV_BROWSE_ROOT = "GIFT_VIEWER_BROWSE_ROOT"
ENV_SERVE = "GIFT_VIEWER_SERVE"
ENV_DIST = "GIFT_VIEWER_DIST"


def _env_path(name: str) -> Optional[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return os.path.realpath(os.path.expanduser(raw))


def outputs_root() -> Optional[str]:
    """The restriction root (realpath), or None when access is unrestricted."""
    return _env_path(ENV_OUTPUTS_ROOT)


def browse_root() -> Optional[str]:
    """Default folder for Compare's run list: the restriction root when set,
    else the optional non-restricting browse hint."""
    return outputs_root() or _env_path(ENV_BROWSE_ROOT)


def serve_mode() -> bool:
    return os.environ.get(ENV_SERVE, "").strip() == "1" or outputs_root() is not None


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

    With no configured root (desktop mode, or ``--serve`` without
    ``--outputs-root``) this is just a resolve — any path is accepted.
    """
    resolved = os.path.realpath(os.path.expanduser(path))
    root = outputs_root()
    if root is not None and not _is_under(resolved, root):
        raise HTTPException(
            status_code=403,
            detail="Path is outside the served outputs root.",
        )
    return resolved
