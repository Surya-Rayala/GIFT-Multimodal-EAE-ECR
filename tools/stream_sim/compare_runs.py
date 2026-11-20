"""Compare two EAE run folders — e.g. a stream run against the static run of the same video.

    python tools/stream_sim/compare_runs.py REF_RUN TEST_RUN [--offset-sec S] [--json out.json]

Reports, per run pair: metric scores; frame rate / frame count / duration;
drill window (frames and seconds) and the speech segment that ended it;
every timeline item and flag matched by id, with the time difference and any
difference in content (kind, label, entry/track numbers, direction, flag type,
message, ...); the transcript word by word (what was recognised differently,
and the time difference of the words both runs recognised); tracker agreement
(ids and keypoints on the common frames, when both runs share a frame rate);
and the artifact videos (frame count, duration, audio track). ``--offset-sec``
is the TEST run's timeline shift relative to REF (e.g. a stream joined late):
it is subtracted before comparing times.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import glob
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, Optional

import numpy as np

_LOCATOR_LOG = re.compile(r"Streaming drill-end: .*\(([\d.]+)s audio, (\d+) passes")


def _ffprobe() -> str:
    cand = os.path.join(os.path.dirname(sys.executable), "ffprobe")
    return cand if os.path.isfile(cand) else "ffprobe"


def load_run(run_dir: str) -> Dict[str, Any]:
    with open(os.path.join(run_dir, "RunInfo.json")) as fh:
        info = json.load(fh)
    with open(os.path.join(run_dir, info["analysis_file"])) as fh:
        analysis = json.load(fh)
    base = analysis["session"]["video_basename"]
    scores = {}
    with open(os.path.join(run_dir, f"{base}_Metrics.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            scores[row["metric_name"]] = float(row["score"])
    tracker_path = os.path.join(run_dir, f"{base}_TrackerOutput.json")
    tracker = None
    if os.path.isfile(tracker_path):
        with open(tracker_path) as fh:
            tracker = json.load(fh)
    transcript = None
    tr_path = os.path.join(run_dir, f"{base}_Transcription.json")
    if os.path.isfile(tr_path):
        with open(tr_path) as fh:
            transcript = json.load(fh)
    return {"dir": run_dir, "info": info, "analysis": analysis, "base": base,
            "scores": scores, "tracker": tracker, "transcript": transcript,
            "asr_span_end_sec": _asr_span_end(run_dir, transcript)}


def _asr_span_end(run_dir: str, transcript: Optional[Dict[str, Any]]) -> Optional[float]:
    """How far the streaming drill-end locator transcribed.

    The ``_denoised.wav`` artifact is exactly the audio the ASR consumed, so
    its start + length is the span; the run log is the fallback.
    """
    den = (transcript or {}).get("denoise") or {}
    wav = os.path.join(run_dir, den.get("audio_artifact") or "")
    if den.get("audio_artifact") and os.path.isfile(wav) and den.get("start_sec") is not None:
        import wave
        try:
            with wave.open(wav, "rb") as w:
                return float(den["start_sec"]) + w.getnframes() / float(w.getframerate())
        except (wave.Error, EOFError):
            pass
    for log in (os.path.join(os.path.dirname(run_dir), "eae.log"), run_dir.rstrip("/") + ".log",
                *glob.glob(os.path.join(run_dir, "*.log"))):
        if os.path.isfile(log):
            with open(log, errors="replace") as fh:
                for line in fh:
                    m = _LOCATOR_LOG.search(line)
                    if m:
                        return float(m.group(1))
    return None


def _norm_word(w: str) -> str:
    return "".join(ch for ch in w.lower() if ch.isalnum() or ch == "'")


def compare_transcripts(ref: Dict[str, Any], test: Dict[str, Any], offset_sec: float = 0.0) -> Optional[Dict[str, Any]]:
    """Word-level comparison over the audio span both runs transcribed."""
    rt, tt = ref.get("transcript"), test.get("transcript")
    if rt is None and tt is None:
        return None

    def words(t, shift):
        return [(w["word"], round(float(w["start"]) - shift, 3), round(float(w["end"]) - shift, 3))
                for s in (t or {}).get("segments") or [] for w in s.get("words") or []]
    rw, tw = words(rt, 0.0), words(tt, offset_sec)
    ends = [e for e in (ref.get("asr_span_end_sec"),
                        None if test.get("asr_span_end_sec") is None else test["asr_span_end_sec"] - offset_sec)
            if e is not None]
    span_end = min(ends) if ends else None
    if span_end is not None:
        rw = [w for w in rw if w[1] < span_end]
        tw = [w for w in tw if w[1] < span_end]
    sm = difflib.SequenceMatcher(a=[_norm_word(w[0]) for w in rw], b=[_norm_word(w[0]) for w in tw],
                                 autojunk=False)
    pairs = [(rw[b.a + k], tw[b.b + k]) for b in sm.get_matching_blocks() for k in range(b.size)]
    dts = [t[1] - r[1] for r, t in pairs]
    moved = [{"word": r[0], "ref_sec": r[1], "test_sec": t[1]} for r, t in pairs if abs(t[1] - r[1]) > 1e-6]
    edits = []
    errors = 0
    for op, a1, a2, b1, b2 in sm.get_opcodes():
        if op == "equal":
            continue
        errors += max(a2 - a1, b2 - b1)
        edits.append({"op": op, "time_sec": (rw[a1][1] if a1 < a2 else tw[b1][1]),
                      "ref": " ".join(w[0] for w in rw[a1:a2]), "test": " ".join(w[0] for w in tw[b1:b2])})
    return {
        "span_end_sec": span_end,
        "ref_words": len(rw), "test_words": len(tw),
        "matched_words": len(dts),
        "identical": [w[0] for w in rw] == [w[0] for w in tw] and all(abs(d) < 1e-6 for d in dts),
        "word_error_rate": round(errors / len(rw), 3) if rw else None,
        "matched_max_abs_dt_sec": round(max((abs(d) for d in dts), default=0.0), 3),
        "matched_mean_dt_sec": round(float(np.mean(dts)), 4) if dts else None,
        "matched_moved": moved,
        "edits": edits,
    }


# Item/flag fields whose value is a measurement (pixels, map units) rather
# than an identity: excluded from the content comparison.
_MEASURED = {"start_xy", "end_xy", "dx", "dy", "z_cross", "L_map", "sample_count"}


def _content(obj: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in obj.items():
        if k in ("time_sec", "start_time_sec", "end_time_sec", "frame", "start_frame", "end_frame", "data"):
            continue
        out[k] = sorted(v) if isinstance(v, list) and all(isinstance(x, str) for x in v) else v
    for k, v in (obj.get("data") or {}).items():
        if k in _MEASURED or isinstance(v, (dict, list)):
            continue
        out[f"data.{k}"] = round(v, 3) if isinstance(v, float) else v
    return out


def _content_diff(r: Dict[str, Any], t: Dict[str, Any]) -> Dict[str, Any]:
    rc, tc = _content(r), _content(t)
    return {k: {"ref": rc.get(k), "test": tc.get(k)} for k in sorted(set(rc) | set(tc)) if rc.get(k) != tc.get(k)}


def _video_facts(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return None
    out = subprocess.run(
        [_ffprobe(), "-v", "error", "-count_packets", "-show_entries",
         "stream=codec_type,nb_read_packets,duration,r_frame_rate", "-of", "json", path],
        capture_output=True, text=True,
    )
    try:
        streams = json.loads(out.stdout).get("streams", [])
    except ValueError:
        return None
    facts: Dict[str, Any] = {"audio": False}
    for s in streams:
        if s.get("codec_type") == "video":
            facts["frames"] = int(s.get("nb_read_packets") or 0)
            facts["duration_sec"] = round(float(s.get("duration") or 0.0), 3)
        elif s.get("codec_type") == "audio":
            facts["audio"] = True
            facts["audio_duration_sec"] = round(float(s.get("duration") or 0.0), 3)
    return facts


def compare(ref: Dict[str, Any], test: Dict[str, Any], offset_sec: float = 0.0) -> Dict[str, Any]:
    ra, ta = ref["analysis"], test["analysis"]
    rf, tf = float(ra["video"]["frame_rate"]), float(ta["video"]["frame_rate"])
    res: Dict[str, Any] = {"ref": ref["dir"], "test": test["dir"], "offset_sec": offset_sec}

    res["scores"] = {
        m: {"ref": ref["scores"].get(m), "test": test["scores"].get(m),
            "diff": (None if ref["scores"].get(m) is None or test["scores"].get(m) is None
                     else round(test["scores"][m] - ref["scores"][m], 4))}
        for m in sorted(set(ref["scores"]) | set(test["scores"]))
    }
    res["video"] = {
        "ref": {k: ra["video"].get(k) for k in ("frame_rate", "total_frames", "duration_sec")},
        "test": {k: ta["video"].get(k) for k in ("frame_rate", "total_frames", "duration_sec")},
        "test_source": ta["video"].get("source"),
    }

    def _dw(a):
        d = a.get("drill_window") or {}
        out = {k: d.get(k) for k in ("start_frame", "end_frame", "start_time_sec", "end_time_sec",
                                      "end_uncertain", "decision_reason")}
        seg = d.get("matched_segment") or {}
        anchor = d.get("end_anchor") or {}
        out["end_word"] = anchor.get("word")
        out["end_word_end_sec"] = anchor.get("end_sec")
        out["end_phrase"] = seg.get("text")
        out["end_phrase_start_sec"] = seg.get("start_sec")
        out["end_phrase_end_sec"] = seg.get("end_sec")
        return out
    rdw, tdw = _dw(ra), _dw(ta)

    def _d(key):
        return (None if rdw[key] is None or tdw[key] is None
                else round(tdw[key] - offset_sec - rdw[key], 4))
    res["drill_window"] = {
        "ref": rdw, "test": tdw,
        "start_diff_sec": _d("start_time_sec"),
        "end_diff_sec": _d("end_time_sec"),
        "end_phrase_start_diff_sec": _d("end_phrase_start_sec"),
        "end_phrase_end_diff_sec": _d("end_phrase_end_sec"),
        "end_word_diff_sec": _d("end_word_end_sec"),
    }

    def _items(a, key):
        return {i.get("item_id") or i.get("flag_id"): i for i in a.get(key) or []}
    for key in ("timeline", "flags"):
        rs = _items(ra["timeline"], "items") if key == "timeline" else _items(ra, "flags")
        ts = _items(ta["timeline"], "items") if key == "timeline" else _items(ta, "flags")
        rows = []
        for k in sorted(set(rs) | set(ts)):
            r, t = rs.get(k), ts.get(k)
            row = {"id": k, "in_ref": r is not None, "in_test": t is not None}
            if r is not None and t is not None:
                for fld in ("time_sec", "start_time_sec", "end_time_sec"):
                    if isinstance(r.get(fld), (int, float)) and isinstance(t.get(fld), (int, float)):
                        row[f"{fld}_diff"] = round(t[fld] - offset_sec - r[fld], 4)
                if r.get("frame") is not None and t.get("frame") is not None:
                    row["frame_ref"], row["frame_test"] = r["frame"], t["frame"]
                content = _content_diff(r, t)
                if content:
                    row["content"] = content
            rows.append(row)
        diffs = [abs(v) for row in rows for f, v in row.items() if f.endswith("_diff") and v is not None]
        res[key] = {"matched": sum(1 for r in rows if r["in_ref"] and r["in_test"]),
                    "only_ref": [r["id"] for r in rows if not r["in_test"]],
                    "only_test": [r["id"] for r in rows if not r["in_ref"]],
                    "max_abs_time_diff_sec": round(max(diffs), 4) if diffs else 0.0,
                    "content_mismatches": {r["id"]: r["content"] for r in rows if r.get("content")},
                    "rows": rows}

    res["transcript"] = compare_transcripts(ref, test, offset_sec)

    if ref["tracker"] is not None and test["tracker"] is not None and abs(rf - tf) < 1e-6:
        shift = int(round(offset_sec * rf))
        rmap = {e["frame"]: e["objects"] for e in ref["tracker"]}
        same_ids = 0
        compared = 0
        max_kp = 0.0
        first_mismatch = None
        for e in test["tracker"]:
            rf_objs = rmap.get(e["frame"] - shift)
            if rf_objs is None:
                continue
            compared += 1
            rids = sorted(o["id"] for o in rf_objs)
            tids = sorted(o["id"] for o in e["objects"])
            if rids == tids:
                same_ids += 1
                rk = {o["id"]: o["keypoints"] for o in rf_objs}
                for o in e["objects"]:
                    a, b = np.asarray(rk[o["id"]], float), np.asarray(o["keypoints"], float)
                    if a.shape == b.shape and a.size:
                        max_kp = max(max_kp, float(np.abs(a - b).max()))
            elif first_mismatch is None:
                first_mismatch = e["frame"]
        res["tracker"] = {"frames_compared": compared, "frames_same_ids": same_ids,
                          "first_id_mismatch_frame": first_mismatch,
                          "max_keypoint_diff_px": round(max_kp, 3)}
    arts = {}
    for run, label in ((ref, "ref"), (test, "test")):
        for p in sorted(glob.glob(os.path.join(run["dir"], f"{run['base']}*.mp4"))):
            arts.setdefault(os.path.basename(p)[len(run["base"]):], {})[label] = _video_facts(p)
    res["artifact_videos"] = arts
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--offset-sec", type=float, default=0.0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    res = compare(load_run(args.ref), load_run(args.test), args.offset_sec)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=2)
    print("scores:")
    for m, v in res["scores"].items():
        print(f"  {m:24s} ref={v['ref']}  test={v['test']}  diff={v['diff']}")
    print("video:", res["video"]["ref"], "→", res["video"]["test"])
    d = res["drill_window"]
    print(f"drill window: ref {d['ref']['start_frame']}-{d['ref']['end_frame']} "
          f"test {d['test']['start_frame']}-{d['test']['end_frame']}  "
          f"Δstart={d['start_diff_sec']}s Δend={d['end_diff_sec']}s  ({d['test']['decision_reason']})")
    print(f"  ended on word: ref {d['ref']['end_word']!r}  test {d['test']['end_word']!r}  "
          f"Δword end={d['end_word_diff_sec']}s  (segments: ref {d['ref']['end_phrase']!r}, "
          f"test {d['test']['end_phrase']!r})")
    for key in ("timeline", "flags"):
        r = res[key]
        print(f"{key}: matched={r['matched']} only_ref={r['only_ref']} only_test={r['only_test']} "
              f"max|Δt|={r['max_abs_time_diff_sec']}s content_mismatches={len(r['content_mismatches'])}")
        for item_id, diff in r["content_mismatches"].items():
            print(f"    {item_id}: " + "; ".join(f"{k} {v['ref']!r}→{v['test']!r}" for k, v in diff.items()))
    tr = res.get("transcript")
    if tr:
        print(f"transcript (to {tr['span_end_sec']}s): ref {tr['ref_words']} words, test {tr['test_words']}, "
              f"matched {tr['matched_words']}, WER {tr['word_error_rate']}, identical={tr['identical']}, "
              f"matched-word max|Δt|={tr['matched_max_abs_dt_sec']}s mean Δt={tr['matched_mean_dt_sec']}s")
        for e in tr["edits"]:
            print(f"    {e['time_sec']:7.2f}s {e['op']:7s} {e['ref']!r} → {e['test']!r}")
    if "tracker" in res:
        print("tracker:", res["tracker"])
    print("artifact videos (suffix: ref → test):")
    for suf, v in res["artifact_videos"].items():
        print(f"  {suf or '(source)':28s} {v.get('ref')} → {v.get('test')}")


if __name__ == "__main__":
    main()
