"""Scaled point viewer launcher.

A single command — orchestrates Vite + the Tauri shell + a Python sidecar that
handles map-image I/O and artifact saving:

    conda activate gift-meae
    python -m src.utils.scaled_point_viewer

Standalone tool for placing real-world reference points onto a room map. You
draw the walls and type each wall's real length; the tool derives one
consistent map scale; then for each point you type how far it is (perpendicular
distance) from its two nearest adjacent walls and it is dropped on the map in
the right spot — robustly, even with corridors or short walls. The saved
points feed the mapper viewer / homography setup later.

Same architecture as the mapper / analysis viewers (Tauri + Vue + FastAPI
sidecar). Node.js is auto-provisioned on first launch via ``node_runtime``.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ..node_runtime import ensure_workspace_packages, prepare_runtime

HERE = Path(__file__).resolve().parent
FRONTEND_DIR = HERE / "frontend"
LABEL = "scaled-point-viewer"


def fail(msg: str, code: int = 1) -> "None":
    sys.stderr.write(f"[{LABEL}] {msg}\n")
    sys.exit(code)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.utils.scaled_point_viewer",
        description=(
            "Launch the scaled point viewer (Tauri + Vue + FastAPI sidecar). "
            "Place real-world reference points on a room map from wall "
            "measurements; saves a points file, an annotated map image, and a "
            "reloadable project file."
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
