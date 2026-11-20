"""Run the EAE on a live UDP stream of an existing video, under controlled impairments.

Builds a GIFT-style ``stream-<name>.vmeta.xml`` (location replaced by a UDP
URL, like ``VideoProcessing.setupVideoProcessing`` does), starts the EAE
(``run_engine_local.py``) and an ffmpeg sender (like
``VideoProcessing.streamVideoFile``), optionally through
``udp_impair_proxy.py``, waits for the EAE, and writes ``summary.json``.

Example (GIFT-like: libx264 + AAC re-encode, MPEG-TS over UDP)::

    python tools/stream_sim/run_scenario.py --name gift_like --out /tmp/streamtest \\
        --source input/Videos/Iguana/3-TrimmedV2.mp4 --vmeta input/test.vmeta.xml \\
        --encode "-c:v libx264 -preset veryfast -c:a aac -b:a 128k"

Sender segments (``--segments``, JSON) allow restarts / discontinuities::

    [{"ss": 0, "t": 15, "encode": "..."}, {"pause": 0.3}, {"ss": 15.3, "encode": "..."}]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from src.utils.paths import PATH_ANCHORS, resolve_config_path  # noqa: E402

NS = "{http://ltsc.ieee.org/xsd/LOM}"


def _bin(name: str) -> str:
    cand = os.path.join(os.path.dirname(sys.executable), name)
    return cand if os.path.isfile(cand) else name


def make_config(template_vmeta: str, out_dir: str, overrides: dict) -> str:
    """Copy the room config next to the scenario with paths made absolute."""
    tree = ET.parse(template_vmeta)
    cfg_rel = None
    for ident in tree.getroot().iter(f"{NS}identifier"):
        if ident.findtext(f"{NS}catalog") == "space_metadata_file":
            cfg_rel = ident.findtext(f"{NS}entry")
    cfg_src = os.path.join(os.path.dirname(os.path.abspath(template_vmeta)), cfg_rel)
    with open(cfg_src) as fh:
        cfg = json.load(fh)
    for key, anchor in PATH_ANCHORS.items():
        if key in cfg and cfg[key]:
            cfg[key] = resolve_config_path(cfg[key], config_dir=os.path.dirname(cfg_src), anchor=anchor)
    cfg.update(overrides or {})
    path = os.path.join(out_dir, "scenario_config.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    return path


def make_vmeta(template_vmeta: str, out_dir: str, name: str, location: str, config_path: str,
               title: str = None) -> str:
    """GIFT-style stream vmeta: same metadata, location replaced by the URL."""
    tree = ET.parse(template_vmeta)
    root = tree.getroot()
    root.find(f"{NS}technical").find(f"{NS}location").text = location
    for ident in root.iter(f"{NS}identifier"):
        cat = ident.findtext(f"{NS}catalog")
        if cat == "space_metadata_file":
            ident.find(f"{NS}entry").text = config_path
        elif cat == "title" and title:
            ident.find(f"{NS}entry").text = title
    ET.register_namespace("", NS.strip("{}"))
    path = os.path.join(out_dir, f"stream-{name}.vmeta.xml")
    tree.write(path, encoding="UTF-8", xml_declaration=True)
    return path


def sender_cmd(source: str, url: str, seg: dict, default_encode: str) -> list:
    cmd = [_bin("ffmpeg"), "-hide_banner", "-loglevel", "error", "-re"]
    if seg.get("ss"):
        cmd += ["-ss", str(seg["ss"])]
    cmd += ["-i", source]
    if seg.get("t"):
        cmd += ["-t", str(seg["t"])]
    cmd += shlex.split(seg.get("encode", default_encode))
    cmd += ["-f", "mpegts", url]
    return cmd


def wait_until_listening(port: int, timeout: float = 120.0) -> bool:
    """True once something has bound UDP ``port`` (the EAE's recorder)."""
    import socket

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return True
        finally:
            s.close()
        time.sleep(0.2)
    return False


def read_scores(run_dir: str) -> dict:
    out = {}
    for p in glob.glob(os.path.join(run_dir, "*_Metrics.csv")):
        with open(p, newline="") as fh:
            for row in csv.DictReader(fh):
                out[row["metric_name"]] = float(row["score"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True, help="scenario parent folder")
    ap.add_argument("--source", required=True, help="video file to stream")
    ap.add_argument("--vmeta", required=True, help="template vmeta (room config, title, role)")
    ap.add_argument("--port", type=int, default=1234, help="EAE receive port")
    ap.add_argument("--encode", default="-map 0:v:0 -map 0:a:0? -c copy",
                    help="ffmpeg output options of the sender")
    ap.add_argument("--segments", default="", help="JSON list of sender segments")
    ap.add_argument("--proxy", default="", help="udp_impair_proxy.py options (enables the proxy)")
    ap.add_argument("--sender-delay", type=float, default=3.0,
                    help="seconds between starting the EAE and the sender")
    ap.add_argument("--eae-delay", type=float, default=0.0,
                    help="start the EAE this long AFTER the sender (late join)")
    ap.add_argument("--config", default="{}", help="JSON config overrides")
    ap.add_argument("--title", default=None)
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    sdir = os.path.abspath(os.path.join(args.out, args.name))
    run_dir = os.path.join(sdir, "run")
    os.makedirs(run_dir, exist_ok=True)
    cfg_path = make_config(args.vmeta, sdir, json.loads(args.config))
    url = f"udp://127.0.0.1:{args.port}?pkt_size=1316"
    vmeta = make_vmeta(args.vmeta, sdir, args.name, url, cfg_path, args.title)
    segments = json.loads(args.segments) if args.segments else [{}]

    send_port = args.port
    proxy = None
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")

    def start_eae():
        return subprocess.Popen(
            [sys.executable, os.path.join(REPO, "run_engine_local.py"), vmeta, "-o", run_dir, "-v"],
            cwd=REPO, env=env, stdout=open(os.path.join(sdir, "eae.log"), "w"), stderr=subprocess.STDOUT,
        )

    t_start = time.monotonic()
    eae = None
    if args.proxy:
        send_port = args.port + 1000
        proxy = subprocess.Popen(
            [sys.executable, os.path.join(os.path.dirname(__file__), "udp_impair_proxy.py"),
             "--listen", f"127.0.0.1:{send_port}", "--forward", f"127.0.0.1:{args.port}",
             "--stats", os.path.join(sdir, "proxy_stats.json"), *shlex.split(args.proxy)],
            stdout=subprocess.DEVNULL, stderr=open(os.path.join(sdir, "proxy.log"), "w"),
        )
    if args.eae_delay <= 0:
        eae = start_eae()
        # UDP is fire-and-forget: wait for the recorder to bind before sending,
        # then the requested extra delay (EAE waiting for a stream to begin).
        if not wait_until_listening(args.port):
            print("warning: EAE never bound the receive port", file=sys.stderr)
        time.sleep(args.sender_delay)
    send_url = f"udp://127.0.0.1:{send_port}?pkt_size=1316"
    sender_log = open(os.path.join(sdir, "sender.log"), "w")
    sender_events = []
    t_sender = time.monotonic()
    for i, seg in enumerate(segments):
        if "pause" in seg:
            time.sleep(float(seg["pause"]))
            continue
        cmd = sender_cmd(args.source, send_url, seg, args.encode)
        sender_log.write(" ".join(cmd) + "\n")
        sender_log.flush()
        p = subprocess.Popen(cmd, stdout=sender_log, stderr=sender_log)
        sender_events.append({"segment": i, "start_sec": round(time.monotonic() - t_sender, 3)})
        if eae is None:
            time.sleep(args.eae_delay)
            eae = start_eae()
        p.wait()
        sender_events[-1]["end_sec"] = round(time.monotonic() - t_sender, 3)
        sender_events[-1]["rc"] = p.returncode
    try:
        rc = eae.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        eae.kill()
        rc = "timeout"
    wall = time.monotonic() - t_start
    if proxy is not None:
        try:
            proxy.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proxy.kill()

    report = {}
    for p in glob.glob(os.path.join(run_dir, "*_StreamReport.json")):
        with open(p) as fh:
            report = json.load(fh)
    dw = {}
    for p in glob.glob(os.path.join(run_dir, "*_DrillWindow.json")):
        with open(p) as fh:
            dw = json.load(fh)
    proxy_stats = {}
    if os.path.isfile(os.path.join(sdir, "proxy_stats.json")):
        with open(os.path.join(sdir, "proxy_stats.json")) as fh:
            proxy_stats = json.load(fh)
    summary = {
        "name": args.name,
        "eae_exit": rc,
        "wall_sec": round(wall, 1),
        "sender_events": sender_events,
        "scores": read_scores(run_dir),
        "drill_window": {k: dw.get(k) for k in ("start_frame", "end_frame", "start_time_sec",
                                                 "end_time_sec", "end_uncertain", "decision_reason")},
        "stream": {k: report.get(k) for k in ("stop_reason", "frame_rate", "processed_frames",
                                               "processed_duration_sec", "wait_for_first_data_sec",
                                               "recorder_issues")},
        "video": {k: (report.get("video") or {}).get(k) for k in (
            "codec", "fps_source", "metadata_fps", "measured_fps", "source_size", "frame_size",
            "decoded_frames", "frames_delivered", "frames_duplicated", "frames_dropped", "gap_count",
            "longest_gap_sec", "reanchors", "decode_errors")},
        "audio": report.get("audio"),
        "proxy": proxy_stats,
        "artifacts": sorted(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "*"))),
    }
    with open(os.path.join(sdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: summary[k] for k in ("name", "eae_exit", "wall_sec", "scores", "drill_window")}, indent=1))


if __name__ == "__main__":
    main()
