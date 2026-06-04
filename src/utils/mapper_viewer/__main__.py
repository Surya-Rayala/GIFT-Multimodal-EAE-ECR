"""Mapper viewer launcher.

A single command — orchestrates Vite + the Tauri shell + a Python sidecar
that handles image / video / artifact I/O:

    conda activate gift-meae
    python -m src.utils.mapper_viewer

Replaces the legacy PyQt5 mapper_app.py. Uses the same architecture as the
analysis viewer (Tauri + Vue + FastAPI sidecar) so neither one inherits the
cv2 / Qt6 library collision that plagues PyQt5 on conda macOS.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ..node_runtime import ensure_workspace_packages, prepare_runtime

HERE = Path(__file__).resolve().parent
FRONTEND_DIR = HERE / "frontend"
LABEL = "mapper-viewer"


def fail(msg: str, code: int = 1) -> "None":
    sys.stderr.write(f"[{LABEL}] {msg}\n")
    sys.exit(code)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.utils.mapper_viewer",
        description=(
            "Launch the mapper viewer (Tauri + Vue + FastAPI sidecar). "
            "Wizard-style tool for authoring the four per-room artifacts: "
            "homography mapping, entry-zone polygons, POD points, and the "
            "room-boundary polygon."
        ),
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build a distributable bundle instead of launching dev mode.",
    )
    parser.add_argument(
        "--project-dir",
        help=(
            "Optional default output directory. Pre-fills the Save Folder "
            "field in the Setup step. Can still be changed in-app."
        ),
    )
    args = parser.parse_args()

    if not FRONTEND_DIR.is_dir():
        fail(f"Frontend directory missing: {FRONTEND_DIR}")

    runtime = prepare_runtime(LABEL)
    env = dict(runtime["env"])
    npm = str(runtime["npm"])
    ensure_workspace_packages(LABEL, env, npm)

    if args.project_dir:
        abs_dir = str(Path(args.project_dir).expanduser().resolve())
        env["VITE_INITIAL_PROJECT_DIR"] = abs_dir
        print(f"[{LABEL}] default save folder: {abs_dir}")

    cmd = [npm, "run", "tauri:build" if args.build else "tauri:dev"]
    try:
        subprocess.run(cmd, cwd=FRONTEND_DIR, env=env, check=True)
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
