"""Path resolution helpers.

Cross-platform, anchor-aware path handling for everything that flows through
config JSONs. The codebase has two distinct "relative path" concepts:

* **config-anchored** — relative to the config JSON's own directory. Used
  for room-specific assets (the map image, the entry-polygon file, the
  pixel-mapping file).
* **project-anchored** — relative to the repository root (where
  ``run_engine.py`` lives). Used for shared model weights.

This module is the single source of truth for both anchors. Add a new path
key by adding it to :data:`PATH_ANCHORS`; nothing else needs to change.

In JSON we always *store* relative, forward-slash, no-tilde-expanded paths.
At load time we *resolve* to absolute paths so downstream code never has to
think about anchors.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Repository root — the directory that contains run_engine.py / src/ / models/.
# Computed from this file's location: src/utils/paths.py → parents[2] is the repo root.
PROJECT_ROOT: str = str(pathlib.Path(__file__).resolve().parents[2])


# Each path-bearing config key maps to its anchor. Update this when you add
# a new path key — every other helper here pulls from this table.
#
# - "config":  relative to the directory of the JSON config file.
# - "project": relative to PROJECT_ROOT.
PATH_ANCHORS: Dict[str, str] = {
    "MapPath":            "config",
    "entry_polys_path":   "config",
    "point_mapping_path": "config",
    "det_weights":        "project",
    "pose2d_weights":     "project",
}


# ---------------------------------------------------------------------------
# Resolution (relative → absolute) and the inverse.
# ---------------------------------------------------------------------------


def _normalize_for_os(value: str) -> str:
    """Convert forward/back slashes into the host OS form and tidy up.

    JSON paths are stored with forward slashes for portability. Python's
    ``os.path`` handles either separator on Windows, but normalizing once
    here means downstream string ops don't trip on mixed forms.
    """
    if not value:
        return ""
    expanded = os.path.expanduser(value)
    normalized = expanded.replace("/", os.sep) if os.sep != "/" else expanded
    return os.path.normpath(normalized)


def resolve_config_path(
    value: Any,
    *,
    config_dir: Optional[str],
    anchor: str = "config",
) -> str:
    """Resolve a path string to an absolute filesystem path.

    Parameters
    ----------
    value : the path as it appears in the config (may be ``""``, ``None``,
            relative, absolute, or starting with ``~``).
    config_dir : directory of the JSON config file (used when ``anchor`` is
            ``"config"``). Ignored for project-anchored paths.
    anchor : ``"config"`` (default) or ``"project"``.

    Returns
    -------
    Absolute path with the host OS's separator, normalized. Empty string
    when ``value`` is empty.
    """
    if not isinstance(value, str) or not value:
        return ""

    p = _normalize_for_os(value)
    if os.path.isabs(p):
        return p

    if anchor == "project":
        base = PROJECT_ROOT
    else:
        # "config" anchor (default). When config_dir isn't given, fall back to
        # the project root rather than the current working directory — that
        # would silently produce different answers depending on where the
        # caller was invoked from.
        base = config_dir or PROJECT_ROOT

    return os.path.normpath(os.path.join(base, p))


def relativize_path(abs_path: str, *, base_dir: str) -> str:
    """Inverse of :func:`resolve_config_path`. Returns a portable, JSON-safe
    relative path (forward slashes, no leading ``./``).

    Falls back to the absolute path when:

    * ``base_dir`` and ``abs_path`` live on different Windows drives
      (``relpath`` raises ``ValueError`` then),
    * ``base_dir`` is empty,
    * the relative form would escape with leading ``..`` to a degree the
      caller would rather see absolute (we keep the ``..`` form anyway —
      it's still valid — but consumers can choose to fall back if they
      want).
    """
    if not isinstance(abs_path, str) or not abs_path:
        return ""
    if not base_dir:
        return abs_path.replace(os.sep, "/")
    try:
        rel = os.path.relpath(abs_path, base_dir)
    except ValueError:
        # Different drive on Windows.
        return abs_path.replace(os.sep, "/")
    return rel.replace(os.sep, "/")


def resolve_all_config_paths(config: Dict[str, Any], config_path: str) -> None:
    """Walk a loaded config dict and replace every known path key with its
    absolute resolution. Mutates ``config`` in place.

    Unknown keys are left untouched. Empty values are left as ``""``.
    """
    if not isinstance(config, dict):
        return
    config_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else None
    for key, anchor in PATH_ANCHORS.items():
        if key not in config:
            continue
        config[key] = resolve_config_path(
            config.get(key, ""),
            config_dir=config_dir,
            anchor=anchor,
        )


def relativize_for_save(
    config: Dict[str, Any],
    config_save_dir: str,
) -> Dict[str, Any]:
    """Return a shallow copy of ``config`` with every path key rewritten as a
    portable relative path appropriate for serializing to JSON.

    The original dict is not mutated. Use this just before ``json.dump``.
    """
    out = dict(config or {})
    if not config_save_dir:
        return out
    for key, anchor in PATH_ANCHORS.items():
        if key not in out:
            continue
        v = out[key]
        if not isinstance(v, str) or not v:
            continue
        # If value is still relative (e.g. user typed a relative path into the
        # builder) leave it alone; just normalize separators.
        if not os.path.isabs(_normalize_for_os(v)):
            out[key] = v.replace(os.sep, "/")
            continue
        base = config_save_dir if anchor == "config" else PROJECT_ROOT
        out[key] = relativize_path(v, base_dir=base)
    return out
