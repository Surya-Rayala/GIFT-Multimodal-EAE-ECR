"""Analysis viewer launcher.

One command — orchestrates Vite + Tauri shell + the Python sidecar:

    conda activate gift-meae
    python -m src.utils.analysis_viewer                       # pick a run via Open... in-app
    python -m src.utils.analysis_viewer path/to/run_folder/   # auto-load on startup

The run folder is one of the per-run directories produced by the processing
engine — it must contain a ``RunInfo.json`` manifest. The launcher resolves
the embedded ``analysis_file`` and primes the Compare-mode dropdown with the
run's outputs root.

Hides the npm / cargo plumbing behind a single Python entry point so the
project keeps its Python-first flavour. No node-modules ceremony for the
caller; first run installs them automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Tuple

from ..node_runtime import REPO_ROOT, ensure_workspace_packages, prepare_runtime

HERE = Path(__file__).resolve().parent
FRONTEND_DIR = HERE / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"
LABEL = "analysis-viewer"


def fail(msg: str, code: int = 1) -> "None":
    sys.stderr.write(f"[{LABEL}] {msg}\n")
    sys.exit(code)


def _lan_ip() -> str:
    """Best-effort primary LAN IP (no packets are actually sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _serve(host: str, port: int, outputs_root: Path, rebuild: bool) -> None:
    """Host the viewer as a LAN web app served from a single origin.

    Builds the Vue frontend once (Node auto-provisioned), then runs the FastAPI
    sidecar bound to ``host:port`` serving both the UI and the API. Access is
    restricted to session folders under ``outputs_root``; people open a specific
    session by deep-linking its host path: ``/?run=/abs/path/to/<run_folder>``.
    """
    if rebuild or not (DIST_DIR / "index.html").is_file():
        print(f"[{LABEL}] building frontend (one-time)…")
        runtime = prepare_runtime(LABEL)
        build_env = dict(runtime["env"])
        npm = str(runtime["npm"])
        ensure_workspace_packages(LABEL, build_env, npm)
        subprocess.run([npm, "run", "build"], cwd=FRONTEND_DIR, env=build_env, check=True)

    env = os.environ.copy()
    env["GIFT_VIEWER_OUTPUTS_ROOT"] = str(outputs_root)
    env["GIFT_VIEWER_DIST"] = str(DIST_DIR)

    lan = _lan_ip()
    print(f"\n[{LABEL}] serving the Analysis Viewer over the network")
    print(f"    outputs root (accessible): {outputs_root}")
    print(f"    this machine:  http://127.0.0.1:{port}")
    if host == "0.0.0.0":
        print(f"    other devices: http://{lan}:{port}")
    host_str = lan if host == "0.0.0.0" else host
    print(f"    open a session by appending its folder path; the path picks the view:")
    print(f"        http://{host_str}:{port}/analysis/?run=<run_folder_under_outputs_root>   # Analysis view")
    print(f"        http://{host_str}:{port}/compare/?run=<run_folder_under_outputs_root>    # Compare view")
    print(f"    (only folders under the outputs root above can be opened)\n")

    cmd = [
        sys.executable,
        "-m",
        "src.utils.analysis_viewer.backend.sidecar_main",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "info",
    ]
    try:
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)


def _resolve_run(path: str) -> Tuple[Path, Path]:
    """Resolve a run-folder path → (run_dir_abs, analysis_json_abs).

    The path must be a directory containing ``RunInfo.json``. The manifest's
    ``analysis_file`` field names the schema-v2 analysis JSON inside the run.
    Fails fast (sys.exit) on any structural problem so the user sees a clear
    error rather than a confusing downstream traceback.
    """
    run_dir = Path(path).expanduser().resolve()
    if not run_dir.is_dir():
        fail(
            f"Run folder not found or not a directory: {run_dir}\n"
            "  Pass the path to a per-run output directory (the one with "
            "RunInfo.json inside)."
        )
    manifest = run_dir / "RunInfo.json"
    if not manifest.is_file():
        fail(
            f"No RunInfo.json in {run_dir}.\n"
            "  This isn't a v1 run folder. Re-run the processing engine "
            "(run_engine_local.py) on the source vmeta to generate one."
        )
    try:
        with manifest.open("r", encoding="utf-8") as fh:
            info = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"Could not read RunInfo.json: {exc}")
    analysis_rel = (info or {}).get("analysis_file") or ""
    if not analysis_rel:
        fail(f"RunInfo.json at {manifest} is missing the 'analysis_file' field.")
    analysis_path = run_dir / analysis_rel
    if not analysis_path.is_file():
        fail(f"Analysis file referenced by RunInfo.json does not exist: {analysis_path}")
    return run_dir, analysis_path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.utils.analysis_viewer",
        description=(
            "Launch the cross-platform Analysis Viewer (Tauri + Vue + FastAPI sidecar)."
        ),
    )
    parser.add_argument(
        "run",
        nargs="?",
        help=(
            "Optional path to a run folder (containing RunInfo.json). If "
            "omitted, the app starts empty — use the Open... button to pick "
            "a run."
        ),
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build a distributable bundle instead of launching dev mode.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help=(
            "Host the viewer as a LAN web app (no desktop window). People on the "
            "network open sessions by deep-linking a run folder: /?run=<path>."
        ),
    )
    parser.add_argument("--host", default="0.0.0.0", help="Serve bind host (default 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8000, help="Serve port (default 8000).")
    parser.add_argument(
        "--outputs-root",
        help=(
            "Folder the server may expose (only sessions under it are openable). "
            "Defaults to the run's parent if a run is given, else ./output."
        ),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a frontend rebuild before serving (otherwise reuse dist/).",
    )
    args = parser.parse_args()

    if not FRONTEND_DIR.is_dir():
        fail(f"Frontend directory missing: {FRONTEND_DIR}")

    if args.serve:
        if args.outputs_root:
            root = Path(args.outputs_root).expanduser().resolve()
        elif args.run:
            root = Path(args.run).expanduser().resolve().parent
        else:
            root = (REPO_ROOT / "output").resolve()
        if not root.is_dir():
            fail(f"Outputs root not found or not a directory: {root}")
        _serve(args.host, args.port, root, args.rebuild)
        return

    runtime = prepare_runtime(LABEL)
    env = dict(runtime["env"])
    npm = str(runtime["npm"])
    ensure_workspace_packages(LABEL, env, npm)

    if args.run:
        run_dir, analysis_path = _resolve_run(args.run)
        # Vite exposes VITE_*-prefixed env vars to client code at dev/build time.
        env["VITE_INITIAL_RUN_DIR"] = str(run_dir)
        env["VITE_INITIAL_SESSION"] = str(analysis_path)
        env["VITE_OUTPUTS_ROOT"] = str(run_dir.parent)
        print(f"[{LABEL}] launching with run: {run_dir}")
        print(f"[{LABEL}]            analysis: {analysis_path}")
        print(f"[{LABEL}]        outputs root: {run_dir.parent}")

    cmd = [npm, "run", "tauri:build" if args.build else "tauri:dev"]
    try:
        subprocess.run(cmd, cwd=FRONTEND_DIR, env=env, check=True)
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
