"""Shared portable-Node bootstrap for the Tauri + Vue desktop viewers.

The three desktop tools — ``scaled_point_viewer``, ``mapper_viewer``, and
``analysis_viewer`` — are Tauri + Vue apps whose frontends need Node.js to
build. Rather than making Node a manual prerequisite, this module gives every
launcher a single ``prepare_runtime()`` call that:

  * reuses a system Node/npm already on PATH (if it is >= ``MIN_NODE_MAJOR``), or
  * downloads a pinned, **portable** Node into a per-user cache on first launch
    (verified against the official ``SHASUMS256.txt``), and

  * wires up **shared build caches** so the three apps don't each carry their
    own multi-GB copies:
      - ``CARGO_TARGET_DIR`` -> ``<repo>/.build/cargo-target`` (one Rust cache), and
      - the frontends are npm *workspaces* of the repo-root ``package.json``, so a
        single hoisted ``<repo>/node_modules`` serves all three. ``npm install``
        therefore runs once, at the repo root.

Pure standard library — no third-party dependencies, so it can run before any
``npm install`` has happened.

Environment overrides:
  * ``GIFT_NODE_DIR``  — where to cache the portable Node (default
    ``~/.cache/gift-meae/node``).
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Optional

from .certs import install_certifi_https

# Pinned Node LTS. Bump deliberately. The matching SHASUMS256.txt is fetched at
# download time, so the version and its checksums always agree.
NODE_VERSION = "20.18.1"
MIN_NODE_MAJOR = 20

# <repo>/src/utils/node_runtime.py -> parents[2] == <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]
CARGO_TARGET_DIR = REPO_ROOT / ".build" / "cargo-target"
NODE_DIST_BASE = "https://nodejs.org/dist"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _log(label: str, msg: str) -> None:
    print(f"[{label}] {msg}", flush=True)


def _cache_root() -> Path:
    override = os.environ.get("GIFT_NODE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".cache" / "gift-meae" / "node"


def _platform_asset() -> "tuple[str, str, str]":
    """Return (os_tag, arch_tag, ext) for the running platform.

    os_tag  -> one of {darwin, linux, win}
    arch_tag-> one of {x64, arm64, armv7l}
    ext     -> archive extension ('tar.gz' or 'zip')
    """
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Darwin":
        os_tag, ext = "darwin", "tar.gz"
    elif system == "Linux":
        os_tag, ext = "linux", "tar.gz"
    elif system == "Windows":
        os_tag, ext = "win", "zip"
    else:
        raise RuntimeError(f"Unsupported OS for portable Node: {system!r}")

    if machine in ("x86_64", "amd64", "x64"):
        arch_tag = "x64"
    elif machine in ("arm64", "aarch64"):
        arch_tag = "arm64"
    elif machine in ("armv7l",):
        arch_tag = "armv7l"
    else:
        raise RuntimeError(f"Unsupported CPU arch for portable Node: {machine!r}")

    return os_tag, arch_tag, ext


def _node_major(node_exe: str) -> Optional[int]:
    try:
        out = subprocess.run(
            [node_exe, "--version"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    m = re.match(r"v?(\d+)\.", out)
    return int(m.group(1)) if m else None


def _system_node_ok() -> bool:
    node = shutil.which("node")
    npm = shutil.which("npm")
    if not node or not npm:
        return False
    major = _node_major(node)
    return major is not None and major >= MIN_NODE_MAJOR


# --------------------------------------------------------------------------- #
# Portable Node download
# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "gift-meae-node-bootstrap"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:  # noqa: S310
        shutil.copyfileobj(resp, out, length=1024 * 1024)
    tmp.replace(dest)


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected_sha(version: str, asset_name: str) -> Optional[str]:
    url = f"{NODE_DIST_BASE}/v{version}/SHASUMS256.txt"
    req = urllib.request.Request(url, headers={"User-Agent": "gift-meae-node-bootstrap"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        text = resp.read().decode("utf-8", "replace")
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == asset_name:
            return parts[0]
    return None


def _extract(archive: Path, into: Path) -> None:
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(into)
    else:
        with tarfile.open(archive) as tf:
            # The archive is checksum-verified above, so a plain extractall is
            # safe. (The 3.12+ `filter=` kwarg isn't available on 3.11.)
            tf.extractall(into)


def _portable_bin_dir(label: str) -> Path:
    """Ensure a portable Node exists in the cache and return its bin directory."""
    os_tag, arch_tag, ext = _platform_asset()
    dirname = f"node-v{NODE_VERSION}-{os_tag}-{arch_tag}"
    asset_name = f"{dirname}.{ext}"
    cache = _cache_root()
    target_dir = cache / dirname
    # On Windows the executables sit at the package root; elsewhere under bin/.
    bin_dir = target_dir if os_tag == "win" else target_dir / "bin"
    node_exe = bin_dir / ("node.exe" if os_tag == "win" else "node")

    if node_exe.exists():
        return bin_dir

    _log(label, f"First launch — downloading portable Node {NODE_VERSION} ({os_tag}-{arch_tag})…")
    install_certifi_https()  # HTTPS via certifi, not the (possibly broken) OS cert store
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / asset_name
    url = f"{NODE_DIST_BASE}/v{NODE_VERSION}/{asset_name}"
    _download(url, archive)

    expected = _expected_sha(NODE_VERSION, asset_name)
    if expected:
        actual = _sha256(archive)
        if actual != expected:
            archive.unlink(missing_ok=True)
            raise RuntimeError(
                f"Node download checksum mismatch for {asset_name}: "
                f"expected {expected}, got {actual}."
            )
    else:
        _log(label, "Warning: could not fetch SHASUMS256.txt; skipping checksum verification.")

    _log(label, "Extracting Node…")
    _extract(archive, cache)
    archive.unlink(missing_ok=True)

    if not node_exe.exists():
        raise RuntimeError(f"Portable Node extraction did not produce {node_exe}.")
    return bin_dir


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _require_cargo(label: str) -> None:
    if shutil.which("cargo") is None:
        sys.stderr.write(
            f"[{label}] Missing `cargo` on PATH. It should come from the conda env:\n"
            "    conda env update --file environment.yml\n"
            "  then re-activate: conda activate gift-meae\n"
        )
        sys.exit(1)


def prepare_runtime(label: str) -> Dict[str, object]:
    """Resolve a usable Node/npm + a build env shared across all three viewers.

    Returns a dict ``{"env", "npm", "node_bin_dir"}`` where ``env`` is a copy of
    the process environment with the chosen Node ``bin`` dir prepended to PATH
    and ``CARGO_TARGET_DIR`` pointed at the shared repo-level cache. ``npm`` is
    the npm executable to invoke. On any unrecoverable failure the process
    exits with a clear message.
    """
    _require_cargo(label)

    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(CARGO_TARGET_DIR)

    win = platform.system() == "Windows"
    npm_name = "npm.cmd" if win else "npm"

    if _system_node_ok():
        return {"env": env, "npm": npm_name, "node_bin_dir": None}

    try:
        bin_dir = _portable_bin_dir(label)
    except Exception as exc:  # offline / unsupported platform / checksum
        sys.stderr.write(
            f"[{label}] Could not provision Node.js automatically: {exc}\n"
            "  Install Node.js 20+ manually (https://nodejs.org) and re-run, "
            "or set GIFT_NODE_DIR to a writable cache location.\n"
        )
        sys.exit(1)

    # Prepend the portable bin dir so node/npm and the Tauri-spawned Vite all
    # resolve the right runtime.
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    npm_path = bin_dir / npm_name
    return {"env": env, "npm": str(npm_path), "node_bin_dir": str(bin_dir)}


def ensure_workspace_packages(label: str, env: Dict[str, str], npm: str) -> None:
    """Run ``npm install`` once at the workspace root (first launch only).

    The three frontends are npm workspaces of the repo-root ``package.json``, so
    a single hoisted ``<repo>/node_modules`` covers all of them. We treat the
    presence of that directory as the "already installed" signal.
    """
    node_modules = REPO_ROOT / "node_modules"
    if node_modules.is_dir():
        return
    _log(label, "Installing npm packages (workspace root, first launch)…")
    subprocess.run([npm, "install"], cwd=str(REPO_ROOT), env=env, check=True)
