"""Drill-window detection from tracker entry + transcription.

Determines a (drill_start_frame, drill_end_frame) pair so downstream
pipeline stages (metrics, artifact videos, viewer timeline) can operate
on only the actual drill segment rather than the full source video.

* Drill start  : the first tracker frame where any track has
                 ``is_entry=True`` or ``birth_location == "entry"``.
* Drill end    : the clearance call: a transcript segment in which every
                 word listed in ``drill_window_required_words`` (default
                 ``"room,clear"``) is spoken after the drill start, in any
                 order, with filler between them. The window ends at the
                 word that completes the call (plus a grace tail), taken
                 from the ASR's word timestamps — not at the segment's end.
                 Segment boundaries follow the ASR's punctuation, which
                 changes with tiny audio differences (it can merge the call
                 with seconds of later chatter); word times do not.
                 A small mean-alignment-confidence gate filters out
                 obvious cross-room ghosts; if the gate eliminates every
                 candidate we fall back to the latest qualifying segment
                 regardless of score.

Design choices:
* No new ML model. The ASR already ships segment/word-level timestamps
  (and, when available, a per-word ``score``); that's all we need.
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
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


GRACE_TAIL_SEC = 0.5            # extra time kept after the clearance call
DEFAULT_REQUIRED_WORDS = "room,clear"
DEFAULT_MIN_ALIGN_SCORE = 0.4   # lenient — mostly trust the word-presence rule
# Streaming confirmation: the same call in two consecutive passes may move by
# an ASR step or two as the right-context grows.
CONFIRM_TOLERANCE_SEC = 0.5

# Punctuation stripping: keep apostrophes inside words (e.g. "don't") but
# drop everything else. ASR output includes punctuation glued to tokens.
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
    # The word that completed the clearance call: {"word", "start_sec",
    # "end_sec"}; ``word`` is None when the segment had no word timing.
    end_anchor: Optional[Dict[str, Any]] = None

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


def _match_tokens(text: str) -> Set[str]:
    """Tokens for required-word matching: a possessive/contracted form also
    counts as its base word ("room's clear" → room)."""
    toks = _tokenize(text)
    return set(toks) | {t[:-2] for t in toks if t.endswith("'s") and len(t) > 2}


def _clearance_call(
    seg: Dict[str, Any], drill_start_sec: float, required_set: Set[str]
) -> Optional[Dict[str, Any]]:
    """Where in ``seg`` the clearance call is completed, or None.

    Walks the segment's words spoken at/after the drill start and returns the
    word at which every required word has been heard. The window is anchored
    on that word's timestamps: the ASR's word times are stable, while its
    segment boundaries follow its punctuation and can swallow seconds of
    unrelated speech. Without usable word timing the whole segment stands in
    for the call (the segment must then start at/after the drill start).
    """
    words = [w for w in seg.get("words") or []
             if isinstance(w.get("start"), (int, float)) and isinstance(w.get("end"), (int, float))]
    seen: Set[str] = set()
    for w in words:
        if float(w["start"]) < drill_start_sec:
            continue
        seen |= _match_tokens(w.get("word", "")) & required_set
        if required_set <= seen:
            return {"word": w.get("word"), "start_sec": float(w["start"]), "end_sec": float(w["end"])}
    seg_start, seg_end = seg.get("start"), seg.get("end")
    if not isinstance(seg_start, (int, float)) or seg_start < drill_start_sec:
        return None
    if not required_set <= _match_tokens(seg.get("text", "")):
        return None
    end = float(seg_end) if isinstance(seg_end, (int, float)) else float(seg_start)
    return {"word": None, "start_sec": float(seg_start), "end_sec": end}


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


def _candidate_summary(cand: Tuple[Dict[str, Any], Dict[str, Any]], required: List[str]) -> Dict[str, Any]:
    seg, call = cand
    return {
        "id": seg.get("id"),
        "start_sec": seg.get("start"),
        "end_sec": seg.get("end"),
        "text": seg.get("text"),
        "mean_align_score": _mean_align_score(seg.get("words")),
        "matched_words": [w for w in required if w in _match_tokens(seg.get("text", ""))],
        "call_end_sec": call["end_sec"],
    }


# ---------------------------------------------------------------------------
# Shared scan/select core — used by both the batch ``compute_drill_window``
# and the streaming ``DrillEndDetector`` so there is exactly one rule
# implementation. A "candidate" is a segment in which every required word is
# spoken at/after the drill start, paired with where that call completes.
# ---------------------------------------------------------------------------
def _qualifying_candidates(
    segments: List[Dict[str, Any]],
    drill_start_sec: float,
    required_set: Set[str],
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    out: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for seg in segments:
        call = _clearance_call(seg, drill_start_sec, required_set)
        if call is not None:
            out.append((seg, call))
    out.sort(key=lambda c: c[1]["end_sec"])
    return out


def _select_candidate(
    candidates: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    min_score: float,
    *,
    prefer: str = "latest",
) -> tuple:
    """Apply the soft mean-align-score gate and choose one candidate.

    ``prefer="latest"`` (batch) keeps the last qualifier; ``prefer="first"``
    (streaming) keeps the earliest — the drill ends the first time the room
    is called clear, not the last teammate echo. Returns ``(chosen, reason)``.
    The gate is lenient: a ``None`` mean score (no per-word scores — e.g. the
    Parakeet path) still passes, so word-presence remains the primary rule.
    """
    scored = [(c, _mean_align_score(c[0].get("words"))) for c in candidates]
    passing = [c for (c, s) in scored if s is None or s >= min_score]
    if passing:
        chosen = passing[0] if prefer == "first" else passing[-1]
        reason = "selected_%s_passing_score_gate" % prefer
    else:
        chosen = candidates[0] if prefer == "first" else candidates[-1]
        reason = "selected_%s_below_score_gate" % prefer
    return chosen, reason


def _window_from_candidate(
    chosen: Tuple[Dict[str, Any], Dict[str, Any]],
    *,
    reason: str,
    drill_start_frame: int,
    drill_start_sec: float,
    fps: float,
    grace_tail: float,
    total_frames: int,
    candidate_summaries: List[Dict[str, Any]],
    required: List[str],
) -> DrillWindow:
    seg, call = chosen
    end_sec = call["end_sec"] + max(0.0, grace_tail)
    end_frame = int(round(end_sec * fps))
    end_frame = max(drill_start_frame, min(total_frames, end_frame))
    return DrillWindow(
        start_frame=drill_start_frame,
        end_frame=end_frame,
        end_uncertain=False,
        decision_reason=reason,
        matched_segment={
            "id": seg.get("id"),
            "start_sec": seg.get("start"),
            "end_sec": seg.get("end"),
            "text": seg.get("text"),
            "mean_align_score": _mean_align_score(seg.get("words")),
        },
        candidates=candidate_summaries,
        required_words=required,
        start_time_sec=drill_start_sec,
        end_time_sec=end_frame / fps if fps > 0 else None,
        end_anchor=dict(call),
    )


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
    # Time convention: starts use (frame-1)/fps — 1-indexed frame N begins at
    # (N-1)/fps — matching the metrics and the deferred-transcription audio
    # slice offset; ends keep frame/fps (end of frame).
    drill_start_sec = (drill_start_frame - 1) / fps if fps > 0 else 0.0

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

    # ---- Build candidate list (shared scan core) --------------------------
    candidates = _qualifying_candidates(segments, drill_start_sec, set(required))
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

    # ---- Soft ghost gate + latest-passing selection (batch semantics) -----
    chosen, reason = _select_candidate(candidates, min_score, prefer="latest")
    return _window_from_candidate(
        chosen,
        reason=reason,
        drill_start_frame=drill_start_frame,
        drill_start_sec=drill_start_sec,
        fps=fps,
        grace_tail=grace_tail,
        total_frames=total_frames,
        candidate_summaries=candidate_summaries,
        required=required,
    )


class DrillEndDetector:
    """Stateful, incremental drill-end detector for streaming transcription.

    Fed the current-pass transcript segments via :meth:`update` as audio
    arrives. Fires on the **first** qualifying segment (contains every
    required word, starts at/after the drill start) — but only after
    **LocalAgreement-2** confirmation: the qualifying segment must appear in
    two consecutive passes before it is trusted, so a one-off mis-hearing of
    "clear" cannot end the drill early. Shares the exact scan/select/build
    core with :func:`compute_drill_window` (word-presence rule, score gate,
    grace tail, frame math) so streaming and batch never diverge.

    If the stream ends without a confirmed fire, :meth:`finalize` applies the
    batch "latest-passing" rule over everything accumulated (or reports
    ``end_uncertain`` when nothing qualified) — matching the legacy fallback.
    """

    def __init__(
        self,
        *,
        drill_start_frame: int,
        fps: float,
        total_frames: int,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        config = config or {}
        self.fps = float(fps or config.get("frame_rate") or 30.0)
        self.drill_start_frame = max(1, int(drill_start_frame))
        self.drill_start_sec = (self.drill_start_frame - 1) / self.fps if self.fps > 0 else 0.0
        self.total_frames = max(1, int(total_frames))
        self.required = _parse_required_words(config.get("drill_window_required_words"))
        self.required_set = set(self.required)
        self.min_score = float(config.get("drill_window_min_align_score", DEFAULT_MIN_ALIGN_SCORE))
        self.grace_tail = float(config.get("drill_window_grace_tail_sec", GRACE_TAIL_SEC))
        # Latest full-transcript pass (each pass supersedes the previous —
        # the worker re-transcribes the growing buffer), plus the call times
        # seen in the previous pass (for the two-consecutive-pass
        # confirmation).
        self._segments: List[Dict[str, Any]] = []
        self._prev_calls: List[float] = []

    @property
    def has_pending(self) -> bool:
        """A qualifier was seen last pass and awaits its second confirmation.

        The worker uses this to read only a short confirmation chunk (instead
        of a full chunk) once a tentative "room clear" appears, minimizing how
        far past the true end we transcribe while still getting the extra
        right-context LocalAgreement needs.
        """
        return bool(self._prev_calls)

    def update(self, segments: List[Dict[str, Any]]) -> Optional[DrillWindow]:
        """Feed the current pass's full segment list; return a window if fired.

        A call is confirmed when the previous pass heard it too, at the same
        time within ``CONFIRM_TOLERANCE_SEC``. Calls are matched by when the
        call was completed, not by segment: between passes the ASR may
        re-punctuate and merge or split the surrounding segments.
        """
        self._segments = list(segments or [])
        candidates = _qualifying_candidates(
            self._segments, self.drill_start_sec, self.required_set
        )
        prev, self._prev_calls = self._prev_calls, [c[1]["end_sec"] for c in candidates]
        confirmed = [c for c in candidates
                     if any(abs(c[1]["end_sec"] - p) <= CONFIRM_TOLERANCE_SEC for p in prev)]
        if not confirmed:
            return None
        # Earliest confirmed qualifier that also passes the (lenient) gate.
        chosen, reason = _select_candidate(confirmed, self.min_score, prefer="first")
        return self._build(chosen, reason.replace("selected_", "streaming_confirmed_"))

    def finalize(self) -> DrillWindow:
        """Stream ended without a confirmed fire — batch fallback over all."""
        candidates = _qualifying_candidates(
            self._segments, self.drill_start_sec, self.required_set
        )
        if not candidates:
            return DrillWindow(
                start_frame=self.drill_start_frame,
                end_frame=self.total_frames,
                end_uncertain=True,
                decision_reason="no_clearance_phrase_found",
                candidates=[_candidate_summary(s, self.required) for s in candidates],
                required_words=self.required,
                start_time_sec=self.drill_start_sec,
                end_time_sec=self.total_frames / self.fps if self.fps > 0 else None,
            )
        chosen, reason = _select_candidate(candidates, self.min_score, prefer="latest")
        return self._build(chosen, reason)

    def _build(self, chosen: Tuple[Dict[str, Any], Dict[str, Any]], reason: str) -> DrillWindow:
        candidates = _qualifying_candidates(
            self._segments, self.drill_start_sec, self.required_set
        )
        return _window_from_candidate(
            chosen,
            reason=reason,
            drill_start_frame=self.drill_start_frame,
            drill_start_sec=self.drill_start_sec,
            fps=self.fps,
            grace_tail=self.grace_tail,
            total_frames=self.total_frames,
            candidate_summaries=[_candidate_summary(s, self.required) for s in candidates],
            required=self.required,
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
