"""Live-stream ingest: MPEG-TS over UDP (and any other ffmpeg-readable URL).

GIFT delivers drill video to the EAE as H.264 + AAC in an MPEG-TS stream over
UDP (default ``udp://127.0.0.1:1234?pkt_size=1316``; GoPro feeds use other
addresses). A stream differs from a file in every way the pipeline cares
about: no frame count, no end, no seeking, timestamps that start anywhere,
packets that can be lost, late, or bursty, frame rates that only the
timestamps reveal, and audio + video interleaved on a single socket that only
one reader can own. This module turns such a stream into exactly what the
static-video pipeline expects — a constant-frame-rate sequence of frames
(frame N ↔ t = (N-1)/fps) and an audio track on the same clock — while the
stream is still arriving.

Architecture::

    UDP / RTSP / SRT … (video + audio multiplexed)
          │
    StreamRecorder ── ffmpeg subprocess, isolated from Python load ──────────
          │  receives in real time (large UDP FIFO, overrun-tolerant), fixes
          │  timestamp wraps + discontinuities, stream-copies everything into a
          │  growing, lossless ``{base}_StreamCapture.ts``
          ▼
    capture file on disk (the buffer between real-time reception and slower
          │                processing; also the raw archive of the session)
          ├──► _AudioFollower thread (PyAV) — decodes audio at the live edge,
          │      resamples, places samples on the video clock (gap → silence,
          │      overlap → trimmed) → AudioTimeline (16 kHz ASR view + archive)
          └──► StreamVideoSource (PyAV, vision-loop thread) — decodes video at
                 the vision loop's pace, measures the real frame rate, and maps
                 timestamps onto a CFR grid (missing → duplicate, extra → drop)
                 → frames for the engine, also written to ``{base}.mp4``

Both followers read the same bytes and apply deterministic, timestamp-only
rules, so video frame N and audio time (N-1)/fps always refer to the same
instant (the *epoch* is the first decoded video frame). After processing,
:meth:`StreamIngest.finalize` muxes the archive audio into ``{base}.mp4`` —
a normal CFR video that every post-processing pass, the overlay renderers and
the Analysis Viewer consume exactly like a static input — and writes
``{base}_StreamReport.json`` with what was received and how it was repaired.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import signal
import subprocess
import threading
import time
import weakref
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .audio import _find_binary
from .media_source import redact_url

logger = logging.getLogger(__name__)

ASR_SAMPLE_RATE = 16000
# Placement tolerance for audio chunks: a chunk whose timestamp lands within
# this many seconds of the current end of the timeline is appended as-is
# (resampler delay / timestamp rounding), beyond it the gap is filled with
# silence or the overlap trimmed.
AUDIO_PLACEMENT_TOL_SEC = 0.004
# Video frames decoded per probe at most (the probe also stops after
# ``fps_probe_sec`` of media time).
FPS_PROBE_MAX_FRAMES = 240
# Common camera / broadcast frame rates — a measured rate within
# ``FPS_SNAP_REL_TOL`` of one of these snaps to the exact value.
STANDARD_FRAME_RATES: Tuple[Fraction, ...] = (
    Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001),
    Fraction(30), Fraction(48), Fraction(50), Fraction(60000, 1001),
    Fraction(60), Fraction(72), Fraction(90), Fraction(100),
    Fraction(120000, 1001), Fraction(120), Fraction(144), Fraction(240),
)
FPS_SNAP_REL_TOL = 0.006
# Metadata rate accepted when it agrees with the measured one this closely.
FPS_METADATA_REL_TOL = 0.03

# PyAV demux options for the capture followers.
_DEMUX_OPTIONS = {"probesize": "10000000", "analyzeduration": "3000000"}


class StreamError(RuntimeError):
    """The stream could not be received (no data, recorder failure, …)."""


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def _as_float(value: Any, default: Optional[float]) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        f = float(value)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


@dataclass
class StreamSettings:
    """Stream-handling knobs (config keys in parentheses; all optional)."""

    connect_timeout_sec: float = 120.0  # (stream_connect_timeout_sec) wait for first data
    idle_timeout_sec: float = 15.0      # (stream_idle_timeout_sec) no data this long → stream ended
    max_duration_sec: float = 3600.0    # (stream_max_duration_sec) cap on processed media time
    max_after_entry_sec: float = 600.0  # (stream_max_after_entry_sec) cap after first entry w/o drill end
    frame_rate: Optional[float] = None  # (stream_frame_rate) force the frame rate
    fps_probe_sec: float = 2.0          # (stream_fps_probe_sec) media time used to measure fps
    # Forward timestamp jumps: one shorter than the idle timeout can be a real
    # outage (packets stopped, then resumed) and is kept as elapsed time; a
    # longer one cannot be — the recorder would have ended the stream — so it
    # is a sender/encoder timestamp reset and is re-joined. Both thresholds
    # therefore default to the idle timeout (+1 s slack). Backward jumps are
    # always re-joined.
    max_gap_fill_sec: float = 16.0      # (stream_max_gap_fill_sec) longest gap filled with repeats
    frame_size: Optional[Tuple[int, int]] = None  # (calibration_frame_size) [W, H] frames are scaled to
    udp_fifo_size: int = 50_000         # (stream_udp_fifo_size) ffmpeg UDP FIFO, 188-byte units (~9.4 MB)
    udp_buffer_size: int = 4 * 1024 * 1024  # (stream_udp_buffer_size) socket receive buffer, bytes
    discontinuity_sec: float = 16.0     # (stream_discontinuity_sec) forward jumps beyond this are re-joined
    input_options: List[str] = field(default_factory=list)  # (stream_input_options) extra ffmpeg input args
    keep_capture: bool = True           # (stream_keep_capture) keep {base}_StreamCapture.ts

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]]) -> "StreamSettings":
        c = config or {}
        d = cls()
        frame_size = None
        raw_size = c.get("calibration_frame_size")
        if isinstance(raw_size, (list, tuple)) and len(raw_size) == 2:
            try:
                w, h = int(raw_size[0]), int(raw_size[1])
                if w > 0 and h > 0:
                    frame_size = (w, h)
            except (TypeError, ValueError):
                frame_size = None
        opts = c.get("stream_input_options") or []
        if isinstance(opts, str):
            opts = opts.split()
        fr = _as_float(c.get("stream_frame_rate"), None)
        idle = max(1.0, _as_float(c.get("stream_idle_timeout_sec"), d.idle_timeout_sec))
        longest_outage = idle + 1.0   # see max_gap_fill_sec / discontinuity_sec
        return cls(
            connect_timeout_sec=max(1.0, _as_float(c.get("stream_connect_timeout_sec"), d.connect_timeout_sec)),
            idle_timeout_sec=idle,
            max_duration_sec=max(1.0, _as_float(c.get("stream_max_duration_sec"), d.max_duration_sec)),
            max_after_entry_sec=max(1.0, _as_float(c.get("stream_max_after_entry_sec"), d.max_after_entry_sec)),
            frame_rate=fr if fr and 1.0 <= fr <= 1000.0 else None,
            fps_probe_sec=min(10.0, max(0.25, _as_float(c.get("stream_fps_probe_sec"), d.fps_probe_sec))),
            max_gap_fill_sec=max(0.0, _as_float(c.get("stream_max_gap_fill_sec"), longest_outage)),
            frame_size=frame_size,
            udp_fifo_size=int(_as_float(c.get("stream_udp_fifo_size"), d.udp_fifo_size)),
            udp_buffer_size=int(_as_float(c.get("stream_udp_buffer_size"), d.udp_buffer_size)),
            discontinuity_sec=max(0.5, _as_float(c.get("stream_discontinuity_sec"), longest_outage)),
            input_options=[str(o) for o in opts],
            keep_capture=bool(c.get("stream_keep_capture", d.keep_capture)),
        )


# ---------------------------------------------------------------------------
# Frame-rate decision (pure)
# ---------------------------------------------------------------------------


def snap_frame_rate(fps: float) -> float:
    """Snap to a standard rate when within :data:`FPS_SNAP_REL_TOL`."""
    if not fps or fps <= 0:
        return fps
    best = min(STANDARD_FRAME_RATES, key=lambda r: abs(float(r) - fps))
    if abs(float(best) - fps) / float(best) <= FPS_SNAP_REL_TOL:
        return float(best)
    return round(float(fps), 3)


def measure_frame_rate(frame_times: Sequence[Optional[float]]) -> Optional[float]:
    """Frame rate from presentation timestamps.

    The median spacing gives a drop-immune first estimate, which assigns each
    frame its slot index (a dropped run shows up as a skipped index); a
    least-squares fit of time against slot index then averages out timestamp
    jitter — precise enough to tell 29.97 from 30 fps (0.1 % apart) over the
    probe window. Needs at least 6 timestamps.
    """
    ts = np.asarray(sorted(float(t) for t in frame_times if t is not None), dtype=np.float64)
    if ts.size < 6:
        return None
    d = np.diff(ts)
    d = d[d > 1e-6]
    if d.size < 5:
        return None
    med = float(np.median(d))
    if med <= 0:
        return None
    k = np.round((ts - ts[0]) / med)
    keep = np.concatenate([[True], np.diff(k) > 0])  # one timestamp per slot
    k, ts = k[keep], ts[keep]
    if k.size < 6 or k[-1] <= 0:
        return 1.0 / med
    period = float(np.polyfit(k, ts, 1)[0])
    return 1.0 / period if period > 0 else 1.0 / med


def decide_frame_rate(
    *,
    override: Optional[float] = None,
    metadata_rates: Sequence[Optional[float]] = (),
    frame_times: Sequence[Optional[float]] = (),
    fallback: float = 30.0,
) -> Tuple[float, str, Dict[str, Any]]:
    """Choose the stream's nominal frame rate → ``(fps, source, details)``.

    Priority: explicit override → container/codec metadata when it agrees
    with the measured timestamp spacing (±3 %) → the measured spacing (the
    timestamps are the ground truth when metadata is missing or wrong, e.g.
    a camera that re-negotiated its rate) → metadata alone → ``fallback``.
    """
    meta = None
    for r in metadata_rates:
        try:
            rf = float(r) if r is not None else None
        except (TypeError, ValueError, ZeroDivisionError):
            rf = None
        if rf and 1.0 <= rf <= 1000.0:
            meta = rf
            break
    measured = measure_frame_rate(frame_times)
    details: Dict[str, Any] = {
        "metadata_fps": round(meta, 4) if meta else None,
        "measured_fps": round(measured, 4) if measured else None,
    }
    if override:
        return float(override), "config_override", details
    if meta and measured:
        if abs(meta - measured) / measured <= FPS_METADATA_REL_TOL:
            return snap_frame_rate(meta), "stream_metadata", details
        details["note"] = "metadata disagrees with timestamps; using measured rate"
        return snap_frame_rate(measured), "measured_timestamps", details
    if meta:
        return snap_frame_rate(meta), "stream_metadata", details
    if measured:
        return snap_frame_rate(measured), "measured_timestamps", details
    return float(fallback or 30.0), "fallback", details


# ---------------------------------------------------------------------------
# CFR normalizer (pure)
# ---------------------------------------------------------------------------


class CfrNormalizer:
    """Map timestamped frames onto a constant-frame-rate grid.

    Slot ``k`` (0-based) covers time ``k / fps`` on the epoch-relative
    timeline; a frame at time ``t`` belongs to slot ``round(t * fps)``.

    * frame lands in the next slot           → emitted
    * frame skips slots (loss / VFR / stall) → previous frame repeated for the
      skipped slots (``duplicate=True``), then the frame — so frame N stays at
      t = (N-1)/fps no matter what arrives
    * frame lands in an already-filled slot  → dropped (higher instantaneous
      rate than nominal); it still becomes the image used for later repeats
    * a frame that jumps more than ``confirm_jump_sec`` ahead, or more than
      ``max_backstep_sec`` back, is only *held*: a jump is real only if the
      following frames continue from it (one frame for a forward jump — an
      outage; five for a backward one, which the recorder normally re-joins
      already). If the stream instead continues on the old timeline, the held
      frames were stray timestamps — e.g. a stale frame a decoder releases
      after an outage — and are dropped (``outliers``), never allowed to
      shift the timeline.
    * a confirmed forward jump up to ``max_gap_fill_sec`` is real elapsed time
      (filled with repeats); a longer one, or a confirmed backward one, is a
      timestamp discontinuity: the timeline is re-anchored so the frames
      continue at the next slot (``reanchors``)
    """

    CONFIRM_FORWARD = 1
    CONFIRM_BACKWARD = 4
    MAX_CANDIDATE_RESTARTS = 10

    def __init__(
        self,
        fps: float,
        *,
        max_gap_fill_sec: float = 10.0,
        max_backstep_sec: float = 0.5,
        confirm_jump_sec: float = 0.5,
    ) -> None:
        self.fps = float(fps)
        self.max_gap_slots = int(round(max(0.0, max_gap_fill_sec) * self.fps))
        self.max_back_slots = max(1, int(round(max_backstep_sec * self.fps)))
        self.confirm_slots = max(2, int(round(confirm_jump_sec * self.fps)))
        # Frames of one candidate jump must follow each other this closely.
        self.cont_tol = max(2, int(round(0.25 * self.fps)))
        self.offset = 0.0          # seconds added to incoming times (re-anchoring)
        self.next_slot = 0
        self.last_item: Any = None
        self._run: List[Tuple[float, int, Any]] = []  # held frames of a candidate jump
        self._restarts = 0
        self.emitted = 0
        self.duplicated = 0
        self.dropped = 0
        self.outliers = 0
        self.gaps: List[Dict[str, Any]] = []
        self.reanchors: List[Dict[str, Any]] = []

    def _slot(self, tt: float) -> int:
        # Round half up (not Python's half-to-even) so frames exactly between
        # two slots — e.g. 60 fps into a 30 fps grid — map consistently.
        return int(np.floor(tt * self.fps + 0.5 + 1e-9))

    def _is_jump(self, slot: int) -> bool:
        return (slot - self.next_slot > self.confirm_slots
                or slot < self.next_slot - self.max_back_slots)

    def push(self, t: float, item: Any) -> List[Tuple[int, Any, bool, float]]:
        """Feed one frame → list of ``(slot, item, duplicate, src_time)``."""
        out: List[Tuple[int, Any, bool, float]] = []
        tt = float(t) + self.offset
        if self.last_item is None:
            # First frame defines slot 0 whatever its timestamp.
            self.offset -= tt
            self._place(0.0, 0, item, out)
            return out
        slot = self._slot(tt)
        if self._run:
            last_slot = self._run[-1][1]
            if abs(slot - (last_slot + 1)) <= self.cont_tol:
                self._run.append((tt, slot, item))
                need = self.CONFIRM_BACKWARD if self._run[0][1] < self.next_slot else self.CONFIRM_FORWARD
                if len(self._run) > need:
                    run, self._run, self._restarts = self._run, [], 0
                    self._commit(run, out)
                return out
            # The candidate did not continue: it was a stray timestamp.
            self.outliers += len(self._run)
            self.dropped += len(self._run)
            self._run = []
            if self._is_jump(slot):
                self._restarts += 1
                if self._restarts > self.MAX_CANDIDATE_RESTARTS:
                    # Timestamps everywhere: trust the newest rather than
                    # dropping indefinitely.
                    self._restarts = 0
                    self._commit([(tt, slot, item)], out)
                    return out
                self._run = [(tt, slot, item)]
                return out
            self._restarts = 0
        elif self._is_jump(slot):
            self._run = [(tt, slot, item)]
            return out
        self._place(tt, slot, item, out)
        return out

    def _commit(self, run: List[Tuple[float, int, Any]], out: List[Tuple[int, Any, bool, float]]) -> None:
        first_tt, first_slot, _ = run[0]
        gap = first_slot - self.next_slot
        if not (0 < gap <= self.max_gap_slots):
            shift = self.next_slot / self.fps - first_tt
            self.reanchors.append({"at_frame": self.next_slot + 1, "jump_sec": round(-shift, 3)})
            self.offset += shift
            run = [(tt + shift, self._slot(tt + shift), it) for tt, _, it in run]
        for tt, slot, it in run:
            self._place(tt, slot, it, out)

    def _place(self, tt: float, slot: int, item: Any, out: List[Tuple[int, Any, bool, float]]) -> None:
        if slot < self.next_slot:
            self.dropped += 1
            self.last_item = item
            return
        gap = slot - self.next_slot
        if gap > 0:
            self.gaps.append({
                "start_frame": self.next_slot + 1,
                "frames": gap,
                "sec": round(gap / self.fps, 3),
            })
            for _ in range(gap):
                out.append((self.next_slot, self.last_item, True, tt))
                self.next_slot += 1
                self.duplicated += 1
        out.append((self.next_slot, item, False, tt))
        self.next_slot += 1
        self.emitted += 1
        self.last_item = item

    def finish(self) -> None:
        """End of stream: an unconfirmed candidate jump is discarded."""
        if self._run:
            self.outliers += len(self._run)
            self.dropped += len(self._run)
            self._run = []


# ---------------------------------------------------------------------------
# Audio timeline
# ---------------------------------------------------------------------------


def placement(cur_len: int, t_sec: float, rate: int, n: int, tol: int) -> Tuple[int, int]:
    """Where ``n`` samples starting at ``t_sec`` go on a view of ``cur_len``.

    Returns ``(pad, skip)``: ``pad`` zeros to insert first (a gap — lost
    packets or audio starting after the epoch), ``skip`` leading samples of
    the chunk to drop (overlap — duplicate data or audio before the epoch).
    Offsets within ``tol`` samples are treated as contiguous — that absorbs
    timestamp jitter between consecutive frames. The first samples of an
    empty view are anchored exactly: they define the A/V alignment.
    """
    target = int(round(float(t_sec) * rate))
    if cur_len == 0:
        return max(0, target), min(int(n), max(0, -target))
    if target > cur_len + tol:
        return target - cur_len, 0
    if target < cur_len - tol:
        return 0, min(int(n), cur_len - target)
    return 0, 0


class _Int16Buffer:
    """Append-only growable int16 buffer of shape (n, channels)."""

    def __init__(self, channels: int = 1, capacity: int = 1 << 16) -> None:
        self.channels = int(channels)
        self._data = np.zeros((capacity, self.channels), dtype=np.int16)
        self.length = 0

    def _reserve(self, extra: int) -> None:
        need = self.length + extra
        if need <= self._data.shape[0]:
            return
        cap = max(need, self._data.shape[0] * 2)
        grown = np.zeros((cap, self.channels), dtype=np.int16)
        grown[: self.length] = self._data[: self.length]
        self._data = grown

    def append_zeros(self, n: int) -> None:
        if n > 0:
            self._reserve(n)
            self._data[self.length: self.length + n] = 0
            self.length += n

    def append(self, samples: np.ndarray) -> None:
        arr = np.asarray(samples, dtype=np.int16).reshape(-1, self.channels)
        if arr.shape[0]:
            self._reserve(arr.shape[0])
            self._data[self.length: self.length + arr.shape[0]] = arr
            self.length += arr.shape[0]

    def slice(self, a: int, b: int) -> np.ndarray:
        a, b = max(0, a), min(self.length, b)
        return self._data[a:b].copy() if b > a else np.zeros((0, self.channels), np.int16)


class AudioTimeline:
    """Stream audio on the video clock (t = 0 ↔ first video frame).

    Two views are kept from the same placement rule: a 16 kHz mono view in
    memory for the ASR (read through :class:`TimelineAudioChunkSource`) and a
    native-rate archive view written incrementally to a WAV for the final
    ``{base}.mp4``. Audio decoded before the epoch is known waits in a
    pending list.
    """

    def __init__(self, archive_wav_path: Optional[str] = None) -> None:
        self._cond = threading.Condition()
        self._asr = _Int16Buffer(1)
        self._archive_path = archive_wav_path
        self._archive_writer = None
        self._archive_rate: Optional[int] = None
        self._archive_channels: Optional[int] = None
        self._archive_len = 0
        self._pending: List[Tuple[float, np.ndarray, Optional[np.ndarray]]] = []
        self.epoch: Optional[float] = None
        self.has_audio: Optional[bool] = None
        self.finished = False
        self.codec: Optional[str] = None
        self.source_rate: Optional[int] = None
        self.source_channels: Optional[int] = None
        self.first_audio_offset_sec: Optional[float] = None
        self.gaps = 0
        self.gap_sec = 0.0
        self.overlaps = 0
        self.overlap_sec = 0.0

    # -- producer side ---------------------------------------------------
    def configure_archive(self, rate: int, channels: int) -> None:
        with self._cond:
            if self._archive_rate is None:
                self._archive_rate, self._archive_channels = int(rate), int(channels)

    def set_epoch(self, epoch_sec: float) -> None:
        with self._cond:
            if self.epoch is not None:
                return
            self.epoch = float(epoch_sec)
            pending, self._pending = self._pending, []
            for item in pending:
                self._place_locked(*item)
            self._cond.notify_all()

    def add(self, raw_t: float, asr: np.ndarray, archive: Optional[np.ndarray],
            asr_t: Optional[float] = None) -> None:
        """Place one decoded audio frame.

        ``raw_t`` is the frame's stream time (it positions the native-rate
        ``archive`` samples). ``asr_t`` is the stream time of the first 16 kHz
        sample in ``asr``: a resampler holds back a few samples, so its output
        lags its input. Placing that output at ``raw_t`` would shift the whole
        ASR view by the resampler delay (~1 ms at 48k→16k) once the leading
        pre-epoch audio is trimmed. ``asr_t`` defaults to ``raw_t``.
        """
        with self._cond:
            self.has_audio = True
            item = (float(raw_t), float(raw_t if asr_t is None else asr_t), asr, archive)
            if self.epoch is None:
                self._pending.append(item)
                return
            self._place_locked(*item)
            self._cond.notify_all()

    def _place_locked(self, raw_t: float, asr_t: float, asr: np.ndarray,
                      archive: Optional[np.ndarray]) -> None:
        epoch = float(self.epoch or 0.0)
        t = raw_t - epoch
        if self.first_audio_offset_sec is None:
            self.first_audio_offset_sec = round(t, 4)
        # ASR view (16 kHz mono), positioned by the resampler's own output time.
        asr = np.asarray(asr, dtype=np.int16).reshape(-1)
        tol = int(AUDIO_PLACEMENT_TOL_SEC * ASR_SAMPLE_RATE)
        pad, skip = placement(self._asr.length, asr_t - epoch, ASR_SAMPLE_RATE, asr.size, tol)
        if pad and self._asr.length:
            self.gaps += 1
            self.gap_sec += pad / ASR_SAMPLE_RATE
        if skip and self._asr.length:
            self.overlaps += 1
            self.overlap_sec += skip / ASR_SAMPLE_RATE
        self._asr.append_zeros(pad)
        self._asr.append(asr[skip:])
        # Archive view (native rate/channels), streamed to disk.
        if archive is None or self._archive_rate is None or not self._archive_path:
            return
        ch = int(self._archive_channels or 1)
        arch = np.asarray(archive, dtype=np.int16).reshape(-1, ch)
        tol_a = int(AUDIO_PLACEMENT_TOL_SEC * self._archive_rate)
        pad_a, skip_a = placement(self._archive_len, t, self._archive_rate, arch.shape[0], tol_a)
        writer = self._archive_writer_locked()
        if writer is None:
            return
        if pad_a:
            writer.write(np.zeros((pad_a, ch), dtype=np.int16))
            self._archive_len += pad_a
        body = arch[skip_a:]
        if body.shape[0]:
            writer.write(body)
            self._archive_len += body.shape[0]

    def _archive_writer_locked(self):
        if self._archive_writer is None and self._archive_path and self._archive_rate:
            try:
                import soundfile as sf
                self._archive_writer = sf.SoundFile(
                    self._archive_path, "w", samplerate=int(self._archive_rate),
                    channels=int(self._archive_channels or 1), subtype="PCM_16",
                )
            except Exception:
                logger.warning("Could not open stream audio archive %s", self._archive_path, exc_info=True)
                self._archive_path = None
        return self._archive_writer

    def mark_no_audio(self) -> None:
        with self._cond:
            if self.has_audio is None:
                self.has_audio = False
            self._cond.notify_all()

    def finish(self) -> None:
        with self._cond:
            if self.has_audio is None:
                self.has_audio = False
            self.finished = True
            if self._archive_writer is not None:
                try:
                    self._archive_writer.close()
                except Exception:
                    pass
                self._archive_writer = None
            self._cond.notify_all()

    # -- consumer side -----------------------------------------------------
    @property
    def duration_sec(self) -> float:
        with self._cond:
            return self._asr.length / ASR_SAMPLE_RATE

    @property
    def archive_info(self) -> Optional[Dict[str, Any]]:
        with self._cond:
            if not self._archive_path or not self._archive_rate or not self._archive_len:
                return None
            return {
                "path": self._archive_path,
                "rate": int(self._archive_rate),
                "channels": int(self._archive_channels or 1),
                "duration_sec": self._archive_len / float(self._archive_rate),
            }

    def read_asr(self, start_sec: float, dur_sec: float, stop_event: Optional[threading.Event] = None
                 ) -> Optional[np.ndarray]:
        """Samples in ``[start, start+dur)`` as float32, blocking until they
        have arrived; ``None`` once the stream ended before ``start``.
        """
        s = int(round(max(0.0, float(start_sec)) * ASR_SAMPLE_RATE))
        e = int(round(max(0.0, float(start_sec) + float(dur_sec)) * ASR_SAMPLE_RATE))
        with self._cond:
            while not self.finished and self.has_audio is not False and self._asr.length < e:
                if stop_event is not None and stop_event.is_set():
                    break
                self._cond.wait(timeout=0.25)
            if s >= self._asr.length:
                return None
            seg = self._asr.slice(s, e).reshape(-1)
        if seg.size < int(0.05 * ASR_SAMPLE_RATE):
            return None
        return seg.astype(np.float32) / 32768.0

    def stats(self) -> Dict[str, Any]:
        with self._cond:
            return {
                "present": bool(self.has_audio),
                "codec": self.codec,
                "source_sample_rate": self.source_rate,
                "source_channels": self.source_channels,
                "first_audio_offset_sec": self.first_audio_offset_sec,
                "received_sec": round(self._asr.length / ASR_SAMPLE_RATE, 3),
                "gaps_filled": self.gaps,
                "gap_fill_sec": round(self.gap_sec, 3),
                "overlaps_trimmed": self.overlaps,
                "overlap_trim_sec": round(self.overlap_sec, 3),
            }


class TimelineAudioChunkSource:
    """:class:`AudioChunkSource`-compatible reader over an :class:`AudioTimeline`.

    ``read`` blocks until the requested window has arrived (live stream) and
    returns ``None`` once the stream ended — the drill-end locator's
    end-of-stream signal.
    """

    sample_rate = ASR_SAMPLE_RATE
    audio_duration_sec: Optional[float] = None
    failed_at_sec: Optional[float] = None

    def __init__(self, timeline: AudioTimeline, stop_event: Optional[threading.Event] = None) -> None:
        self.timeline = timeline
        self.stop_event = stop_event

    def read(self, start_sec: float, dur_sec: float) -> Optional[np.ndarray]:
        if dur_sec <= 0:
            return None
        return self.timeline.read_asr(start_sec, dur_sec, self.stop_event)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Recorder (ffmpeg subprocess)
# ---------------------------------------------------------------------------

_LIVE_RECORDERS: "weakref.WeakSet[StreamRecorder]" = weakref.WeakSet()


def _kill_live_recorders() -> None:  # pragma: no cover - interpreter shutdown
    for rec in list(_LIVE_RECORDERS):
        try:
            rec.kill()
        except Exception:
            pass


atexit.register(_kill_live_recorders)


def _udp_url_with_defaults(url: str, settings: StreamSettings) -> str:
    """Add robust-reception options to a ``udp://`` URL (user values win)."""
    if not url.lower().startswith("udp:"):
        return url
    base, _, query = url.partition("?")
    present = {p.split("=", 1)[0].lower() for p in query.split("&") if p}
    extra = []
    if "fifo_size" not in present:
        extra.append(f"fifo_size={int(settings.udp_fifo_size)}")
    if "overrun_nonfatal" not in present:
        extra.append("overrun_nonfatal=1")
    if "buffer_size" not in present:
        extra.append(f"buffer_size={int(settings.udp_buffer_size)}")
    if "timeout" not in present:
        # ffmpeg ends the input after this much silence — the clean,
        # signal-free end-of-stream path for UDP (microseconds).
        extra.append(f"timeout={int(settings.idle_timeout_sec * 1e6)}")
    if not extra:
        return url
    return base + "?" + "&".join([q for q in query.split("&") if q] + extra)


class StreamRecorder:
    """Receive a stream with ffmpeg into a growing MPEG-TS capture file.

    A supervisor thread owns the ffmpeg process: it restarts it while no
    data has arrived yet (until ``connect_timeout_sec``), watches the capture
    file grow (arrival bookkeeping + idle detection), and stops it gracefully
    on request (``q``, then SIGINT/CTRL_BREAK twice, then kill).
    ``finished`` is set only after ffmpeg has exited, so readers that see it
    know every byte is on disk.
    """

    POLL_SEC = 0.05

    def __init__(self, url: str, capture_path: str, log_path: str, settings: StreamSettings) -> None:
        self.url = url
        self.capture_path = capture_path
        self.log_path = log_path
        self.settings = settings
        self.ffmpeg = _find_binary("ffmpeg")
        self.finished = threading.Event()
        self.data_started = threading.Event()
        self._stop_requested = threading.Event()
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self.size = 0
        self.started_wall: Optional[float] = None
        self.first_data_wall: Optional[float] = None
        self.last_growth_wall: Optional[float] = None
        self.ended_wall: Optional[float] = None
        self.spawns = 0
        self.stop_reason: Optional[str] = None
        self.exit_codes: List[Optional[int]] = []
        self.error: Optional[str] = None
        # Set when the source's audio codec cannot be stream-copied into
        # MPEG-TS (e.g. PCM/G.711 from some RTSP cameras) → re-encode to AAC.
        self.audio_transcode = False

    # -- command ------------------------------------------------------------
    def command(self) -> List[str]:
        s = self.settings
        audio_codec = ["-c:a", "aac", "-b:a", "160k"] if self.audio_transcode else []
        return [
            self.ffmpeg or "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "level+info",
            # Timestamp jumps beyond this are re-joined by ffmpeg (encoder
            # restarts, wraps); smaller ones are kept as real gaps.
            "-dts_delta_threshold", f"{s.discontinuity_sec:g}",
            *s.input_options,
            "-i", _udp_url_with_defaults(self.url, s),
            "-map", "0:v:0", "-map", "0:a:0?", "-dn", "-sn",
            "-c", "copy", *audio_codec,
            # Write packets as they arrive: no interleaving queue (one stalled
            # stream must not hold back the other) and flush per packet so the
            # followers see data promptly.
            "-max_interleave_delta", "0", "-flush_packets", "1",
            # Backstop for an orphaned recorder (EAE killed while a camera
            # keeps streaming): stop after the processing cap's worth of media.
            "-t", f"{s.max_duration_sec + s.connect_timeout_sec + 120:.0f}",
            "-f", "mpegts", "-y", self.capture_path,
        ]

    # -- lifecycle -------------------------------------------------------------
    def start(self) -> None:
        if self.ffmpeg is None:
            raise StreamError("ffmpeg is required to receive streams but was not found.")
        os.makedirs(os.path.dirname(os.path.abspath(self.capture_path)) or ".", exist_ok=True)
        self.started_wall = time.monotonic()
        _LIVE_RECORDERS.add(self)
        self._thread = threading.Thread(target=self._supervise, name="stream-recorder", daemon=True)
        self._thread.start()

    def _spawn(self) -> subprocess.Popen:
        kwargs: Dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        log = open(self.log_path, "ab")
        try:
            log.write(("\n# %s spawn %d: %s\n" % (
                time.strftime("%Y-%m-%d %H:%M:%S"), self.spawns + 1,
                " ".join(redact_url(a) or a for a in self.command()),
            )).encode("utf-8", "replace"))
            log.flush()
            proc = subprocess.Popen(
                self.command(), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=log, **kwargs,
            )
        finally:
            log.close()
        self.spawns += 1
        return proc

    def _supervise(self) -> None:
        s = self.settings
        try:
            while True:
                with self._lock:
                    if self._stop_requested.is_set():
                        break
                    proc = self._proc = self._spawn()
                while proc.poll() is None:
                    self._poll_size()
                    now = time.monotonic()
                    if self._stop_requested.is_set():
                        self._graceful_stop(proc)
                        break
                    if not self.data_started.is_set():
                        if now - (self.started_wall or now) > s.connect_timeout_sec:
                            self.stop_reason = self.stop_reason or "connect_timeout"
                            self._graceful_stop(proc)
                            break
                    elif now - (self.last_growth_wall or now) > s.idle_timeout_sec + 3.0:
                        # Protocol-independent idle stop (UDP normally ends
                        # itself via its socket timeout first).
                        self.stop_reason = self.stop_reason or "stream_idle"
                        self._graceful_stop(proc)
                        break
                    time.sleep(self.POLL_SEC)
                self.exit_codes.append(proc.poll())
                self._poll_size()
                if self.data_started.is_set() or self._stop_requested.is_set():
                    break
                if not self.audio_transcode and self._log_mentions(
                    "not currently supported in container", "could not write header",
                    "codec not supported",
                ):
                    self.audio_transcode = True
                    logger.warning("Stream audio cannot be copied into MPEG-TS; re-encoding it to AAC.")
                    continue
                if time.monotonic() - (self.started_wall or 0.0) > s.connect_timeout_sec:
                    self.stop_reason = self.stop_reason or "connect_timeout"
                    break
                time.sleep(0.1)  # no data yet — listen again (keep the unbound gap short)
            if self.data_started.is_set() and self.stop_reason is None:
                self.stop_reason = "stream_ended"
        except Exception as exc:  # pragma: no cover - defensive
            self.error = f"{type(exc).__name__}: {exc}"
            logger.exception("Stream recorder supervisor failed")
        finally:
            self._poll_size()
            self.ended_wall = time.monotonic()
            self.finished.set()

    def _poll_size(self) -> None:
        try:
            size = os.path.getsize(self.capture_path)
        except OSError:
            return
        if size > self.size:
            now = time.monotonic()
            self.size = size
            self.last_growth_wall = now
            if self.first_data_wall is None:
                self.first_data_wall = now
                self.data_started.set()

    def _graceful_stop(self, proc: subprocess.Popen) -> None:
        """``q`` → interrupt ×2 → terminate → kill (each step bounded)."""
        steps: List[Tuple[str, float]] = [("q", 1.5), ("int", 1.0), ("int", 2.0), ("term", 2.0), ("kill", 2.0)]
        for step, wait in steps:
            if proc.poll() is not None:
                return
            try:
                if step == "q":
                    if proc.stdin:
                        proc.stdin.write(b"q")
                        proc.stdin.flush()
                elif step == "int":
                    if os.name == "nt":
                        proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                    else:
                        proc.send_signal(signal.SIGINT)
                elif step == "term":
                    proc.terminate()
                else:
                    proc.kill()
            except (OSError, ValueError):
                pass
            try:
                proc.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                continue

    def request_stop(self, reason: str) -> None:
        if not self._stop_requested.is_set():
            self.stop_reason = self.stop_reason or reason
        self._stop_requested.set()

    def wait_finished(self, timeout: Optional[float] = None) -> bool:
        return self.finished.wait(timeout)

    def kill(self) -> None:
        self._stop_requested.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass

    def wait_for_data(self, stop_event: Optional[threading.Event] = None) -> bool:
        """Block until the first bytes arrive (True) or the recorder gave up."""
        while not self.data_started.is_set():
            if self.finished.is_set() or (stop_event is not None and stop_event.is_set()):
                return self.data_started.is_set()
            self.data_started.wait(0.1)
        return True

    def _log_mentions(self, *needles: str) -> bool:
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read().lower()
        except OSError:
            return False
        return any(n in text for n in needles)

    def log_tail(self, n: int = 12) -> str:
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as fh:
                lines = [ln.rstrip() for ln in fh.readlines() if ln.strip()]
            return "\n".join(lines[-n:])
        except OSError:
            return ""

    def log_summary(self) -> Dict[str, int]:
        """Counts of reception problems ffmpeg reported while recording."""
        keys = {
            "continuity_errors": "continuity check failed",
            "corrupt_packets": "corrupt",
            "timestamp_discontinuities": "timestamp discontinuity",
            "udp_fifo_overruns": "circular buffer overrun",
            "non_monotonic_dts": "non monotonically increasing dts",
        }
        counts = {k: 0 for k in keys}
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    low = line.lower()
                    for k, needle in keys.items():
                        if needle in low:
                            counts[k] += 1
        except OSError:
            pass
        return counts


class _FollowReader:
    """Read-only, non-seekable file object that follows a growing file.

    ``read`` returns as soon as any bytes are available, waits while the
    recorder is still running, and reports EOF (``b""``) only after the
    recorder has exited and every byte has been consumed — so PyAV demuxes a
    live capture exactly as it would the finished file.
    """

    def __init__(self, path: str, recorder: StreamRecorder, stop_event: threading.Event, poll: float = 0.01) -> None:
        self.path = path
        self.recorder = recorder
        self.stop_event = stop_event
        self.poll = poll
        self._fh = None
        self.bytes_read = 0

    def _ensure_open(self) -> bool:
        if self._fh is None:
            try:
                self._fh = open(self.path, "rb")
            except OSError:
                return False
        return True

    def read(self, n: int = -1) -> bytes:
        n = 1 << 20 if n is None or n < 0 else int(n)
        while True:
            if self._ensure_open():
                data = self._fh.read(n)
                if data:
                    self.bytes_read += len(data)
                    return data
            if self.stop_event.is_set():
                return b""
            if self.recorder.finished.is_set():
                if self._ensure_open():
                    data = self._fh.read(n)  # bytes flushed right before exit
                    self.bytes_read += len(data)
                    return data
                return b""
            time.sleep(self.poll)

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None


def _open_follower(recorder: StreamRecorder, stop_event: threading.Event):
    import av  # PyAV (bundled with the env's ffmpeg)

    reader = _FollowReader(recorder.capture_path, recorder, stop_event)
    container = av.open(reader, mode="r", format="mpegts", options=dict(_DEMUX_OPTIONS))
    return container, reader


# ---------------------------------------------------------------------------
# Audio follower
# ---------------------------------------------------------------------------


def _frames_time(frames: Sequence[Any], fallback: float) -> float:
    """Stream time of the first sample in a resampler's output ``frames``.

    The resampler stamps its output with the input clock minus the samples it
    still holds, which is the correct position; ``fallback`` covers outputs
    without a timestamp.
    """
    for o in frames:
        return float(o.time) if o.time is not None else fallback
    return fallback


def _samples(frames: Sequence[Any]) -> np.ndarray:
    parts = [o.to_ndarray().reshape(-1) for o in frames]
    return np.concatenate(parts) if parts else np.zeros(0, np.int16)


class _AudioPlacer:
    """Resample decoded audio frames into both timeline views and place them.

    The ASR view goes through a 48k→16k resampler that holds back a few
    samples, so each output batch is positioned by the resampler's own output
    timestamp, never by the input frame's. That is what keeps the ASR view
    sample-exact on the video clock.
    """

    def __init__(self, timeline: AudioTimeline, time_base: Any = None) -> None:
        import av

        self.timeline = timeline
        self.time_base = time_base
        self._rs_asr = av.AudioResampler(format="s16", layout="mono", rate=ASR_SAMPLE_RATE)
        self._rs_arch = None
        self.next_t: Optional[float] = None      # stream time right after the last frame
        self._asr_clock: Optional[float] = None  # stream time of the next 16 kHz sample

    def feed(self, fr: Any) -> None:
        if fr.time is None:
            # Several AAC frames share one PES packet and only the first
            # carries a PTS (the follower's non-seekable input leaves the rest
            # unset). Stamp the frame as contiguous so the resampler can time
            # its output.
            if self.next_t is None:
                return
            if fr.time_base is None and self.time_base is not None:
                fr.time_base = self.time_base
            if fr.time_base is not None:
                fr.pts = int(round(self.next_t / float(fr.time_base)))
        t = float(fr.time) if fr.time is not None else float(self.next_t)
        self.next_t = t + fr.samples / float(fr.sample_rate or 1)
        if self._rs_arch is None:
            import av

            layout = fr.layout.name if fr.layout.nb_channels <= 2 else "stereo"
            self._rs_arch = av.AudioResampler(format="s16", layout=layout, rate=fr.sample_rate)
            self.timeline.source_rate = int(fr.sample_rate)
            self.timeline.source_channels = int(fr.layout.nb_channels)
            self.timeline.configure_archive(int(fr.sample_rate), 1 if layout == "mono" else 2)
        asr_out = self._rs_asr.resample(fr)
        arch = _samples(self._rs_arch.resample(fr))
        asr = _samples(asr_out)
        self._place(t, asr, arch if arch.size else None, asr_out)

    def flush(self) -> None:
        """End of stream: place the samples the resampler still holds."""
        tail = self._rs_asr.resample(None)
        if tail and self.next_t is not None:
            self._place(self.next_t, _samples(tail), None, tail)

    def _place(self, t: float, asr: np.ndarray, arch: Optional[np.ndarray], asr_out: Sequence[Any]) -> None:
        asr_t = _frames_time(asr_out, t if self._asr_clock is None else self._asr_clock)
        self._asr_clock = asr_t + asr.size / float(ASR_SAMPLE_RATE)
        self.timeline.add(t, asr, arch, asr_t=asr_t)


class _AudioFollower(threading.Thread):
    """Decode the capture's audio at the live edge onto the AudioTimeline."""

    def __init__(self, recorder: StreamRecorder, timeline: AudioTimeline, stop_event: threading.Event) -> None:
        super().__init__(name="stream-audio", daemon=True)
        self.recorder = recorder
        self.timeline = timeline
        self.stop_event = stop_event
        self.decode_errors = 0
        self.error: Optional[str] = None

    def run(self) -> None:
        container = reader = None
        try:
            if not self.recorder.wait_for_data(self.stop_event):
                return
            import av

            container, reader = _open_follower(self.recorder, self.stop_event)
            if not container.streams.audio:
                self.timeline.mark_no_audio()
                return
            stream = container.streams.audio[0]
            self.timeline.codec = stream.codec_context.name
            placer = _AudioPlacer(self.timeline, stream.time_base)
            for packet in container.demux(stream):
                if self.stop_event.is_set():
                    break
                try:
                    frames = packet.decode()
                except av.error.FFmpegError:
                    self.decode_errors += 1
                    continue
                for fr in frames:
                    placer.feed(fr)
            else:
                placer.flush()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            logger.warning("Stream audio follower stopped: %s", self.error, exc_info=True)
        finally:
            self.timeline.finish()
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass
            if reader is not None:
                reader.close()


# ---------------------------------------------------------------------------
# Video source
# ---------------------------------------------------------------------------


@dataclass
class FrameMeta:
    index: int                      # 1-based output frame number
    time_sec: float                 # (index-1)/fps on the normalized timeline
    duplicate: bool                 # repeat of the previous frame (gap fill)
    source_time_sec: Optional[float]  # epoch-relative time of the decoded frame used


class _FrameWriter(threading.Thread):
    """Background writer of the processed frame sequence ({base}.mp4 video)."""

    def __init__(self, path: str, fps: float, size: Tuple[int, int]) -> None:
        super().__init__(name="stream-writer", daemon=True)
        import cv2

        self.path = path
        self.q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=8)
        self.written = 0
        self.error: Optional[str] = None
        self._writer = None
        for fourcc in ("avc1", "mp4v"):
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), float(fps), tuple(size))
            if w.isOpened():
                self._writer = w
                self.fourcc = fourcc
                break
            w.release()
        if self._writer is None:
            self.error = "no usable MP4 encoder (avc1/mp4v)"
        self.start()

    def write(self, frame: np.ndarray) -> None:
        if self._writer is not None and self.error is None:
            self.q.put(frame)

    def run(self) -> None:
        while True:
            frame = self.q.get()
            if frame is None:
                break
            try:
                self._writer.write(frame)
                self.written += 1
            except Exception as exc:  # pragma: no cover - defensive
                self.error = f"{type(exc).__name__}: {exc}"
        if self._writer is not None:
            self._writer.release()

    def close(self) -> None:
        if self.is_alive():
            self.q.put(None)
            self.join()


class StreamVideoSource:
    """Decode the capture's video at the consumer's pace → CFR frames."""

    def __init__(
        self,
        recorder: StreamRecorder,
        timeline: AudioTimeline,
        settings: StreamSettings,
        stop_event: threading.Event,
        *,
        video_out_path: Optional[str],
        fallback_fps: float = 30.0,
    ) -> None:
        self.recorder = recorder
        self.timeline = timeline
        self.settings = settings
        self.stop_event = stop_event
        self.video_out_path = video_out_path
        self.fallback_fps = fallback_fps
        self.container = None
        self.reader = None
        self.stream = None
        self.fps: Optional[float] = None
        self.fps_source: Optional[str] = None
        self.fps_details: Dict[str, Any] = {}
        self.source_size: Optional[Tuple[int, int]] = None
        self.frame_size: Optional[Tuple[int, int]] = None
        self.codec: Optional[str] = None
        self.epoch: Optional[float] = None
        self.decoded = 0
        self.decode_errors = 0
        self.resized = 0
        self.size_changes: List[Dict[str, Any]] = []
        self.normalizer: Optional[CfrNormalizer] = None
        self.writer: Optional[_FrameWriter] = None
        self._decoded_iter: Optional[Iterator[Tuple[Optional[float], np.ndarray]]] = None
        self._probe_buffer: List[Tuple[Optional[float], np.ndarray]] = []
        self.delivered = 0
        self.eof = False

    def _decode_iter(self) -> Iterator[Tuple[Optional[float], np.ndarray]]:
        import av

        next_t: Optional[float] = None
        for packet in self.container.demux(self.stream):
            if self.stop_event.is_set():
                return
            try:
                frames = packet.decode()
            except av.error.FFmpegError:
                self.decode_errors += 1
                continue
            for fr in frames:
                t = float(fr.time) if fr.time is not None else next_t
                step = 1.0 / (self.fps or self.fallback_fps or 30.0)
                next_t = (t + step) if t is not None else None
                img = fr.to_ndarray(format="bgr24")
                self.decoded += 1
                yield t, img
        self.eof = True

    def probe(self) -> Dict[str, Any]:
        """Wait for the stream, decode the probe window, fix fps/size/epoch."""
        if not self.recorder.wait_for_data(self.stop_event):
            tail = self.recorder.log_tail()
            raise StreamError(
                f"No stream data received from {redact_url(self.recorder.url)} within "
                f"{self.settings.connect_timeout_sec:.0f}s"
                + (f" (recorder: {self.recorder.stop_reason})" if self.recorder.stop_reason else "")
                + (f"\n{tail}" if tail else "")
            )
        try:
            self.container, self.reader = _open_follower(self.recorder, self.stop_event)
        except Exception as exc:
            raise StreamError(
                f"Could not demux the stream ({type(exc).__name__}: {exc}).\n{self.recorder.log_tail()}"
            ) from exc
        if not self.container.streams.video:
            raise StreamError("The stream carries no video track.")
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.codec = self.stream.codec_context.name
        self._decoded_iter = self._decode_iter()
        first_t: Optional[float] = None
        for t, img in self._decoded_iter:
            self._probe_buffer.append((t, img))
            if first_t is None and t is not None:
                first_t = t
            if len(self._probe_buffer) >= FPS_PROBE_MAX_FRAMES:
                break
            if first_t is not None and t is not None and t - first_t >= self.settings.fps_probe_sec:
                break
        if not self._probe_buffer:
            raise StreamError("The stream ended before any video frame could be decoded.\n"
                              + self.recorder.log_tail())

        meta_rates: List[Optional[float]] = []
        for attr in ("average_rate", "guessed_rate", "base_rate"):
            try:
                meta_rates.append(float(getattr(self.stream, attr)) if getattr(self.stream, attr) else None)
            except (TypeError, ValueError, ZeroDivisionError, AttributeError):
                meta_rates.append(None)
        try:
            ctx_rate = self.stream.codec_context.framerate
            meta_rates.append(float(ctx_rate) if ctx_rate else None)
        except (TypeError, ValueError, ZeroDivisionError, AttributeError):
            pass
        self.fps, self.fps_source, self.fps_details = decide_frame_rate(
            override=self.settings.frame_rate,
            metadata_rates=meta_rates,
            frame_times=[t for t, _ in self._probe_buffer],
            fallback=self.fallback_fps,
        )
        h, w = self._probe_buffer[0][1].shape[:2]
        self.source_size = (w, h)
        self.frame_size = self.settings.frame_size or (w, h)
        if self.settings.frame_size and self.settings.frame_size != (w, h):
            src_ar, dst_ar = w / float(h), self.frame_size[0] / float(self.frame_size[1])
            if abs(src_ar - dst_ar) / dst_ar > 0.01:
                logger.warning(
                    "Stream frames are %dx%d but calibration_frame_size is %dx%d (different "
                    "aspect ratio) — frames are stretched to the calibration size.",
                    w, h, self.frame_size[0], self.frame_size[1],
                )
        # The epoch is the first decoded frame (frame 1, t = 0); audio is
        # placed relative to it.
        first_t = self._probe_buffer[0][0]
        if first_t is None:
            first_t = next((t for t, _ in self._probe_buffer if t is not None), 0.0)
        self.epoch = float(first_t)
        self.timeline.set_epoch(self.epoch)
        self.normalizer = CfrNormalizer(self.fps, max_gap_fill_sec=self.settings.max_gap_fill_sec)
        if self.video_out_path:
            self.writer = _FrameWriter(self.video_out_path, self.fps, self.frame_size)
            if self.writer.error:
                logger.warning("Stream video archive disabled: %s", self.writer.error)
        return self.info()

    def info(self) -> Dict[str, Any]:
        return {
            "codec": self.codec,
            "fps": self.fps,
            "fps_source": self.fps_source,
            **self.fps_details,
            "source_size": list(self.source_size) if self.source_size else None,
            "frame_size": list(self.frame_size) if self.frame_size else None,
        }

    def _fit(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if (w, h) == tuple(self.frame_size):
            return img
        import cv2

        if (w, h) != tuple(self.source_size or (w, h)):
            if not self.size_changes or self.size_changes[-1]["size"] != [w, h]:
                self.size_changes.append({"at_decoded_frame": self.decoded, "size": [w, h]})
        self.resized += 1
        return cv2.resize(img, tuple(self.frame_size), interpolation=cv2.INTER_LINEAR)

    def frames(self) -> Iterator[Tuple[int, np.ndarray, FrameMeta]]:
        """Yield ``(frame_number, bgr_frame, FrameMeta)`` on the CFR grid."""
        if self.normalizer is None:
            self.probe()
        assert self.normalizer is not None and self._decoded_iter is not None

        def _source() -> Iterator[Tuple[Optional[float], np.ndarray]]:
            buffered, self._probe_buffer = self._probe_buffer, []
            yield from buffered
            yield from self._decoded_iter

        fps = float(self.fps or 30.0)
        last_t: Optional[float] = None
        for t, img in _source():
            if t is None:
                t = (last_t + 1.0 / fps) if last_t is not None else self.epoch or 0.0
            last_t = t
            img = self._fit(img)
            for slot, item, dup, src_t in self.normalizer.push(t - float(self.epoch or 0.0), img):
                self.delivered += 1
                if self.writer is not None:
                    self.writer.write(item)
                yield slot + 1, item, FrameMeta(slot + 1, slot / fps, dup, round(src_t, 6))
        self.normalizer.finish()

    def close(self) -> None:
        if self.container is not None:
            try:
                self.container.close()
            except Exception:
                pass
            self.container = None
        if self.reader is not None:
            self.reader.close()
            self.reader = None

    def stats(self) -> Dict[str, Any]:
        n = self.normalizer
        gaps = list(n.gaps) if n else []
        return {
            **self.info(),
            "decoded_frames": self.decoded,
            "decode_errors": self.decode_errors,
            "frames_delivered": self.delivered,
            "frames_duplicated": n.duplicated if n else 0,
            "frames_dropped": n.dropped if n else 0,
            "timestamp_outliers": n.outliers if n else 0,
            "gaps": gaps[:200],
            "gap_count": len(gaps),
            "longest_gap_sec": max((g["sec"] for g in gaps), default=0.0),
            "reanchors": list(n.reanchors) if n else [],
            "frames_resized": self.resized,
            "size_changes": self.size_changes[:50],
        }


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class StreamIngest:
    """One stream session: recorder + audio follower + video source.

    Lifecycle: ``start()`` (as early as possible — before models load) →
    ``probe()`` (blocks until the stream arrives; fixes fps/size/epoch) →
    iterate ``frames()`` in the vision loop, read audio via
    ``audio_source()`` → ``request_stop(reason)`` → ``finalize(n)``. ``close()``
    is idempotent and safe in ``finally`` blocks.
    """

    def __init__(self, url: str, output_dir: str, basename: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.url = url
        self.output_dir = os.path.abspath(output_dir)
        self.basename = basename
        self.config = config or {}
        self.settings = StreamSettings.from_config(self.config)
        os.makedirs(self.output_dir, exist_ok=True)
        self.capture_path = os.path.join(self.output_dir, f"{basename}_StreamCapture.ts")
        self.log_path = os.path.join(self.output_dir, f"{basename}_StreamRecorder.log")
        self.report_path = os.path.join(self.output_dir, f"{basename}_StreamReport.json")
        self.video_path = os.path.join(self.output_dir, f"{basename}.mp4")
        self._video_tmp = os.path.join(self.output_dir, f"{basename}_stream_video.tmp.mp4")
        self._audio_tmp = os.path.join(self.output_dir, f"{basename}_stream_audio.tmp.wav")
        # A previous session's files in the same folder must go before the
        # recorder starts: a leftover capture would look like data already
        # arriving (and be read as this stream), and its size would mask the
        # new capture's growth from the idle check.
        for stale in (self.capture_path, self.report_path, self.video_path,
                      self._video_tmp, self._audio_tmp):
            try:
                os.remove(stale)
            except OSError:
                pass
        try:
            # A fresh log per session (the recorder appends across respawns).
            open(self.log_path, "wb").close()
        except OSError:
            pass
        self.stop_event = threading.Event()
        self.timeline = AudioTimeline(self._audio_tmp)
        self.recorder = StreamRecorder(url, self.capture_path, self.log_path, self.settings)
        self.audio_follower = _AudioFollower(self.recorder, self.timeline, self.stop_event)
        self.video = StreamVideoSource(
            self.recorder, self.timeline, self.settings, self.stop_event,
            video_out_path=self._video_tmp,
            fallback_fps=float(self.config.get("frame_rate") or 30.0),
        )
        self.started = False
        self.probe_info: Optional[Dict[str, Any]] = None
        self.processing_stop_reason: Optional[str] = None
        self.started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self._closed = False

    def start(self) -> "StreamIngest":
        if not self.started:
            self.recorder.start()
            self.audio_follower.start()
            self.started = True
            logger.info("Receiving %s → %s", redact_url(self.url), self.capture_path)
        return self

    def probe(self) -> Dict[str, Any]:
        self.start()
        if self.probe_info is None:
            self.probe_info = self.video.probe()
            logger.info(
                "Stream video: %s %sx%s @ %.3f fps (%s); audio follower running.",
                self.video.codec, *(self.video.source_size or (0, 0)), self.video.fps, self.video.fps_source,
            )
        return self.probe_info

    @property
    def fps(self) -> float:
        return float(self.video.fps or 0.0)

    def frames(self) -> Iterator[Tuple[int, np.ndarray, FrameMeta]]:
        return self.video.frames()

    def audio_source(self) -> TimelineAudioChunkSource:
        return TimelineAudioChunkSource(self.timeline, self.stop_event)

    def request_stop(self, reason: str) -> None:
        """Stop receiving (processing is done or a cap was hit)."""
        if self.processing_stop_reason is None:
            self.processing_stop_reason = reason
        self.recorder.request_stop(reason)

    def _shutdown(self, timeout: float = 20.0) -> None:
        if not self.started:
            self.timeline.finish()
            self.video.close()
            return
        self.recorder.request_stop(self.processing_stop_reason or "closed")
        self.recorder.wait_finished(timeout)
        # Followers drain to the (now final) end of the capture by themselves;
        # the stop event only guards against a wedged reader.
        self.audio_follower.join(timeout)
        if self.audio_follower.is_alive():
            self.stop_event.set()
            self.audio_follower.join(5.0)
        self.timeline.finish()
        self.video.close()

    def finalize(self, processed_frames: int) -> Dict[str, Any]:
        """Stop receiving, write ``{base}.mp4`` + the stream report.

        Returns the report dict (``video_path`` is the playable CFR source
        for post-processing, or None when it could not be written).
        """
        self._shutdown()
        fps = self.fps or float(self.config.get("frame_rate") or 30.0)
        duration = processed_frames / fps if fps > 0 else 0.0
        video_ok = False
        if self.video.writer is not None:
            self.video.writer.close()
            video_ok = (self.video.writer.error is None and os.path.isfile(self._video_tmp)
                        and os.path.getsize(self._video_tmp) > 0)
        mux = None
        if video_ok:
            mux = self._mux_final(duration)
        for tmp in (self._video_tmp, self._audio_tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        if not self.settings.keep_capture:
            try:
                os.remove(self.capture_path)
            except OSError:
                pass
        report = self.report(processed_frames, mux_info=mux)
        try:
            with open(self.report_path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)
        except OSError:
            logger.warning("Could not write %s", self.report_path, exc_info=True)
        self._closed = True
        return report

    def _mux_final(self, duration: float) -> Dict[str, Any]:
        """Video-only tmp + archive WAV → ``{base}.mp4`` (AAC, same clock)."""
        ffmpeg = _find_binary("ffmpeg")
        arch = self.timeline.archive_info
        info: Dict[str, Any] = {"audio": bool(arch)}
        if ffmpeg and arch:
            cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-i", self._video_tmp, "-i", arch["path"],
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                # Pad audio that ended early so both tracks span the video.
                "-af", "apad", "-t", f"{duration:.6f}",
                "-movflags", "+faststart", self.video_path,
            ]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if res.returncode == 0 and os.path.isfile(self.video_path):
                    return info
                info["mux_error"] = (res.stderr or "")[-400:]
            except (OSError, subprocess.TimeoutExpired) as exc:
                info["mux_error"] = str(exc)
            logger.warning("Audio mux of the stream recording failed; keeping video only.")
        info["audio"] = False
        try:
            os.replace(self._video_tmp, self.video_path)
        except OSError:
            info["video_error"] = "could not move the video recording into place"
        return info

    def report(self, processed_frames: int, mux_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        rec = self.recorder
        fps = self.fps or 0.0
        base = rec.started_wall or 0.0
        return {
            "schema_version": "1.0",
            "source_url": redact_url(self.url),
            "started_at": self.started_at,
            "wait_for_first_data_sec": (round(rec.first_data_wall - base, 3)
                                        if rec.first_data_wall else None),
            "receive_wall_sec": (round((rec.ended_wall or time.monotonic()) - (rec.first_data_wall or base), 3)
                                 if rec.first_data_wall else None),
            "stop_reason": self.processing_stop_reason or rec.stop_reason,
            "recorder_stop_reason": rec.stop_reason,
            "recorder_spawns": rec.spawns,
            "recorder_exit_codes": rec.exit_codes,
            "recorder_issues": rec.log_summary(),
            "bytes_received": rec.size,
            "frame_rate": fps,
            "processed_frames": int(processed_frames),
            "processed_duration_sec": round(processed_frames / fps, 3) if fps else None,
            "video": self.video.stats(),
            "audio": {**self.timeline.stats(), "decode_errors": self.audio_follower.decode_errors,
                      "follower_error": self.audio_follower.error},
            "artifacts": {
                "capture": self.capture_path if os.path.isfile(self.capture_path) else None,
                "video": self.video_path if os.path.isfile(self.video_path) else None,
                "recorder_log": self.log_path if os.path.isfile(self.log_path) else None,
            },
            "mux": mux_info,
            "settings": {
                "connect_timeout_sec": self.settings.connect_timeout_sec,
                "idle_timeout_sec": self.settings.idle_timeout_sec,
                "max_duration_sec": self.settings.max_duration_sec,
                "max_after_entry_sec": self.settings.max_after_entry_sec,
                "max_gap_fill_sec": self.settings.max_gap_fill_sec,
                "discontinuity_sec": self.settings.discontinuity_sec,
                "frame_rate_override": self.settings.frame_rate,
                "calibration_frame_size": list(self.settings.frame_size) if self.settings.frame_size else None,
            },
        }

    def close(self) -> None:
        """Idempotent hard cleanup (errors / aborted runs)."""
        if self._closed:
            return
        self._closed = True
        try:
            self.request_stop(self.processing_stop_reason or "aborted")
            self._shutdown(timeout=10.0)
        finally:
            if self.video.writer is not None:
                self.video.writer.close()
            self.recorder.kill()
