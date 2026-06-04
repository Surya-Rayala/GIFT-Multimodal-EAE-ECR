"""Drill-window detection from tracker entry + transcription.

Determines a (drill_start_frame, drill_end_frame) pair so downstream
pipeline stages (metrics, artifact videos, viewer timeline) can operate
on only the actual drill segment rather than the full source video.

* Drill start  : the first tracker frame where any track has
                 ``is_entry=True`` or ``birth_location == "entry"``.
* Drill end    : the latest transcript segment (after drill_start) whose
                 normalized text contains every word listed in
                 ``drill_window_required_words`` (default ``"room,clear"``).
                 Words may appear in any order with filler between them.
                 A small mean-alignment-confidence gate filters out
                 obvious cross-room ghosts; if the gate eliminates every
                 candidate we fall back to the latest qualifying segment
                 regardless of score.

Design choices:
* No new ML model. WhisperX already ships word-level timestamps + a
  forced-alignment ``score`` per word; that's all we need.
* No diarization. We can't know which speaker is the leader, so it
  wouldn't actually disambiguate. Instead the word-presence rule plus
  "pick the latest qualifier" handles teammate exclamations.
* If no entry is detected we never even check the audio — without the
  entrance gate there's nothing to clear.
* All frame indices are absolute (1-indexed) and tied to the original
  video timeline. No re-indexing.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


GRACE_TAIL_SEC = 0.5            # extra audio kept after the matched segment
DEFAULT_REQUIRED_WORDS = "room,clear"
DEFAULT_MIN_ALIGN_SCORE = 0.4   # lenient — mostly trust the word-presence rule

# Punctuation stripping: keep apostrophes inside words (e.g. "don't") but
# drop everything else. WhisperX includes punctuation glued to tokens.
_PUNCT_RE = re.compile(r"[^a-z0-9'\s]")


@dataclass
class DrillWindow:
    start_frame: int
    end_frame: int
    end_uncertain: bool
    decision_reason: str
    matched_segment: Optional[Dict[str, Any]] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    required_words: List[str] = field(default_factory=list)
    start_time_sec: Optional[float] = None
    end_time_sec: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_required_words(raw: Optional[str]) -> List[str]:
    if not raw or not isinstance(raw, str):
        raw = DEFAULT_REQUIRED_WORDS
    return [w.strip().lower() for w in raw.split(",") if w.strip()]


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    return cleaned.split()


def _mean_align_score(words: Optional[List[Dict[str, Any]]]) -> Optional[float]:
    if not words:
        return None
    scores = [w.get("score") for w in words if isinstance(w.get("score"), (int, float))]
    if not scores:
        return None
    return float(sum(scores) / len(scores))


def find_drill_start_frame(tracker_output: List[Dict[str, Any]]) -> Optional[int]:
    """Earliest frame any track is flagged ``is_entry`` or born ``entry``.

    Walks ``tracker_output`` (frame-by-frame list-of-dicts produced by the
    pose+tracker loop) and returns the first ``frame`` index whose objects
    contain a track confirmed in the entry region. Returns ``None`` if the
    tracker never confirmed an entry.
    """
    for entry in tracker_output:
        for obj in entry.get("objects", []) or []:
            if obj.get("is_entry") or obj.get("birth_location") == "entry":
                return int(entry["frame"])
    return None


def _load_transcription(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        logger.warning("Failed to load transcription at %s", path, exc_info=True)
        return None


def _candidate_summary(seg: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {
        "id": seg.get("id"),
        "start_sec": seg.get("start"),
        "end_sec": seg.get("end"),
        "text": seg.get("text"),
        "mean_align_score": _mean_align_score(seg.get("words")),
        "matched_words": [w for w in required if w in set(_tokenize(seg.get("text", "")))],
    }


def compute_drill_window(
    *,
    transcription_path: Optional[str],
    drill_start_frame: Optional[int],
    total_frames: int,
    frame_rate: float,
    config: Optional[Dict[str, Any]] = None,
) -> DrillWindow:
    """Decide ``(drill_start, drill_end)`` for the current session.

    See module docstring for the rule cascade. Always returns a valid
    :class:`DrillWindow` — failure modes set ``end_uncertain=True`` and
    record a human-readable ``decision_reason`` rather than raising.
    """
    config = config or {}
    fps = float(frame_rate or config.get("frame_rate") or 30.0)
    total_frames = max(1, int(total_frames))

    required = _parse_required_words(config.get("drill_window_required_words"))
    min_score = float(config.get("drill_window_min_align_score", DEFAULT_MIN_ALIGN_SCORE))
    grace_tail = float(config.get("drill_window_grace_tail_sec", GRACE_TAIL_SEC))

    # ---- No-entry early-out ------------------------------------------------
    if drill_start_frame is None:
        return DrillWindow(
            start_frame=1,
            end_frame=total_frames,
            end_uncertain=True,
            decision_reason="no_entry_detected",
            required_words=required,
            start_time_sec=0.0,
            end_time_sec=total_frames / fps if fps > 0 else None,
        )

    drill_start_frame = max(1, int(drill_start_frame))
    drill_start_sec = drill_start_frame / fps if fps > 0 else 0.0

    # ---- Pull the transcription -------------------------------------------
    transcription = _load_transcription(transcription_path) if transcription_path else None
    segments = (transcription or {}).get("segments", []) or []

    if not segments:
        return DrillWindow(
            start_frame=drill_start_frame,
            end_frame=total_frames,
            end_uncertain=True,
            decision_reason="no_transcription_available",
            required_words=required,
            start_time_sec=drill_start_sec,
            end_time_sec=total_frames / fps if fps > 0 else None,
        )

    # ---- Build candidate list ---------------------------------------------
    required_set = set(required)
    candidates: List[Dict[str, Any]] = []
    for seg in segments:
        seg_start = seg.get("start")
        if not isinstance(seg_start, (int, float)):
            continue
        if seg_start < drill_start_sec:
            continue
        tokens = set(_tokenize(seg.get("text", "")))
        if required_set.issubset(tokens):
            candidates.append(seg)

    candidate_summaries = [_candidate_summary(s, required) for s in candidates]

    if not candidates:
        return DrillWindow(
            start_frame=drill_start_frame,
            end_frame=total_frames,
            end_uncertain=True,
            decision_reason="no_clearance_phrase_found",
            candidates=candidate_summaries,
            required_words=required,
            start_time_sec=drill_start_sec,
            end_time_sec=total_frames / fps if fps > 0 else None,
        )

    # ---- Apply soft cross-room ghost gate ---------------------------------
    scored = [(c, _mean_align_score(c.get("words"))) for c in candidates]
    passing = [(c, s) for (c, s) in scored if s is None or s >= min_score]

    if passing:
        chosen = passing[-1][0]      # latest passing
        reason = "selected_latest_passing_score_gate"
    else:
        chosen = candidates[-1]      # latest period
        reason = "selected_latest_below_score_gate"

    # ---- Compute end frame -------------------------------------------------
    seg_end_sec = chosen.get("end")
    if not isinstance(seg_end_sec, (int, float)):
        seg_end_sec = chosen.get("start", drill_start_sec)
    end_sec = float(seg_end_sec) + max(0.0, grace_tail)
    end_frame = int(round(end_sec * fps))
    end_frame = max(drill_start_frame, min(total_frames, end_frame))

    return DrillWindow(
        start_frame=drill_start_frame,
        end_frame=end_frame,
        end_uncertain=False,
        decision_reason=reason,
        matched_segment={
            "id": chosen.get("id"),
            "start_sec": chosen.get("start"),
            "end_sec": chosen.get("end"),
            "text": chosen.get("text"),
            "mean_align_score": _mean_align_score(chosen.get("words")),
        },
        candidates=candidate_summaries,
        required_words=required,
        start_time_sec=drill_start_sec,
        end_time_sec=end_frame / fps if fps > 0 else None,
    )


def save_drill_window_sidecar(
    window: DrillWindow,
    *,
    output_directory: str,
    video_basename: str,
) -> str:
    """Write ``{basename}_DrillWindow.json`` next to other artifacts."""
    path = os.path.join(output_directory, f"{video_basename}_DrillWindow.json")
    try:
        with open(path, "w") as f:
            json.dump(window.to_dict(), f, indent=4)
    except Exception:
        logger.warning("Failed to write DrillWindow sidecar at %s", path, exc_info=True)
    return path
