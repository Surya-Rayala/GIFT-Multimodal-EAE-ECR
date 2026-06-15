"""STAY_ALONG_WALL — wall-distance discipline metric.

Scoring rule (per entrant, per frame in the active drill window):

    L            = robust per-entrant shoulder-to-elbow length (in MAP units),
                   estimated from pose keypoints converted through the engine's
                   PixelMapper.
    dist(person) = boundary.exterior.distance(foot_point_in_map)
    label        = too_close   if dist <  WALL_TOO_CLOSE_FRACTION * L
                 = too_far     if dist >  WALL_TOO_FAR_FRACTION   * L
                 = safe        otherwise
    score(track) = safe_count / observed_count
    final_score  = mean over entrants

Current band: ``[0, 1.0 × L]`` is safe. The "too close" gate is disabled
(``WALL_TOO_CLOSE_FRACTION = 0``) — hugging the wall is acceptable; only
the depth gate matters. Distances greater than one shoulder-to-elbow
length from the nearest wall are flagged as ``too_far``.

Each contiguous too_close / too_far run with duration ≥ MIN_EXCURSION_SEC is
emitted as an excursion flag (start_frame, end_frame, type) into
``ctx.metric_flags`` so the analysis layer can surface it as a flag and a
timeline bar without the metric needing to know the analysis schema.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from .metric import AbstractMetric
from ._shared import load_inroom_ids, pick_latest, select_entry_tracks, team_size


# ---------------------------------------------------------------------------
# Developer constants — band shape and excursion-flag policy.
# Per project convention these are NOT exposed in JSON. Adjust here.
# ---------------------------------------------------------------------------
WALL_TOO_CLOSE_FRACTION       = 0.0   # × L: closer than this → too_close.
                                      # 0 disables the gate (no frame can have
                                      # negative wall distance) — only the depth
                                      # gate matters.
WALL_TOO_FAR_FRACTION         = 1.0   # × L: farther than this → too_far
MIN_EXCURSION_SEC             = 0.5   # excursion runs shorter than this
                                      # are NOT flagged (score still penalised)
MIN_VALID_L_SAMPLES           = 5     # below this many per-frame L samples
                                      # → fall back to team median, then constant
MAD_OUTLIER_K                 = 3.0   # median ± k·MAD outlier band
FALLBACK_SHOULDER_LENGTH_MAP  = 25.0  # last-resort L (map units) when no pose data
                                      # exists for any entrant in the team

# Halpe26 keypoint indices.
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW     = 7
KP_RIGHT_ELBOW    = 8


# ---------------------------------------------------------------------------
# Pure helpers (kept module-level so expertCompare can reuse them on cached
# tracker output without touching the metric instance).
# ---------------------------------------------------------------------------


def _pixel_to_map_xy(pixel_mapper, pt) -> Optional[Tuple[float, float]]:
    """Resolve a (x,y) pixel point to map coords. Returns None on any failure."""
    if pixel_mapper is None or pt is None:
        return None
    try:
        xy = pixel_mapper.pixel_to_map(pt)
    except Exception:
        return None
    arr = np.asarray(xy, dtype=float).reshape(-1)
    if arr.size < 2 or not np.isfinite(arr[:2]).all():
        return None
    return float(arr[0]), float(arr[1])


_KP_LEFT_ANKLE  = 15
_KP_RIGHT_ANKLE = 16

# Pixel offset used to evaluate the local pixel→map scale by finite difference.
# Big enough to dominate floating-point noise in pixel_to_map; small enough that
# the homography is well-approximated as locally affine over the offset.
_LOCAL_SCALE_PROBE_DELTA_PX = 6.0


def _local_pixel_to_map_scale(
    pixel_mapper, anchor_xy: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    """Local pixel→map scale (k_x, k_y) at a floor-plane image location.

    Probes the homography by finite differences around ``anchor_xy``: how
    many map units does a unit pixel step cover at this location? The
    returned ``k_x`` and ``k_y`` are linear scale factors in map-units /
    pixel along the image x and y axes respectively. Use them to scale a
    pixel-space displacement vector to its equivalent length on the floor
    *as if the displacement were lying on the floor at this anchor*. That
    sidesteps the homography's "above-floor points project to the wrong
    spot" problem for above-floor keypoints (shoulder, elbow), because we
    measure the arm purely in pixel space and only convert by a factor
    evaluated at a true on-floor reference.

    Returns ``None`` if the homography cannot be evaluated at the anchor
    or any probe (out of frame, NaN, etc.).
    """
    delta = _LOCAL_SCALE_PROBE_DELTA_PX
    p0 = _pixel_to_map_xy(pixel_mapper, (anchor_xy[0], anchor_xy[1]))
    p_x = _pixel_to_map_xy(pixel_mapper, (anchor_xy[0] + delta, anchor_xy[1]))
    p_y = _pixel_to_map_xy(pixel_mapper, (anchor_xy[0], anchor_xy[1] + delta))
    if p0 is None or p_x is None or p_y is None:
        return None
    k_x = float(np.hypot(p_x[0] - p0[0], p_x[1] - p0[1])) / delta
    k_y = float(np.hypot(p_y[0] - p0[0], p_y[1] - p0[1])) / delta
    if not (np.isfinite(k_x) and np.isfinite(k_y)) or k_x <= 0.0 or k_y <= 0.0:
        return None
    return (k_x, k_y)


def _floor_anchor_pixel(
    keypoints,
    scores,
    bbox,
    *,
    pose_conf_floor: float,
) -> Optional[Tuple[float, float]]:
    """Pick the most-on-floor reference point in pixel space for this frame.

    Preference order:
      1. Ankle midpoint  — both ankles are physically ~5 cm off the floor,
         so their pixel positions hit the floor-plane homography most
         accurately. Requires both ankle keypoints above the confidence
         floor.
      2. Bbox bottom-center — always available as long as the detector
         produced a bbox; corresponds to where the body meets the floor
         in the image. Slightly noisier than ankles when the detector
         crops feet, but never absent.

    Returns ``None`` only when neither option is usable.
    """
    try:
        la_score = float(scores[_KP_LEFT_ANKLE])
        ra_score = float(scores[_KP_RIGHT_ANKLE])
    except (TypeError, ValueError, IndexError):
        la_score = ra_score = 0.0

    if la_score >= pose_conf_floor and ra_score >= pose_conf_floor:
        try:
            la = keypoints[_KP_LEFT_ANKLE]
            ra = keypoints[_KP_RIGHT_ANKLE]
            return (
                (float(la[0]) + float(ra[0])) * 0.5,
                (float(la[1]) + float(ra[1])) * 0.5,
            )
        except (TypeError, IndexError):
            pass

    if bbox is not None:
        try:
            x1, _y1, x2, y2 = (
                float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]),
            )
            return ((x1 + x2) * 0.5, y2)
        except (TypeError, IndexError, ValueError):
            return None
    return None


def _per_frame_shoulder_elbow_length(
    keypoints,
    scores,
    pixel_mapper,
    *,
    pose_conf_floor: float,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[float]:
    """One frame's shoulder→elbow length, scaled to MAP units via the local
    pixel→map factor at the person's floor anchor.

    Algorithm:
      1. For each shoulder/elbow side that's above the confidence floor,
         compute the per-arm pixel displacement ``(dx, dy)``. We average
         all valid sides at the end.
      2. Pick a floor-anchor pixel for this frame (ankles preferred; bbox
         bottom-center fallback).
      3. Probe the homography at that anchor to get ``(k_x, k_y)`` —
         map-units per pixel along x and y at the person's floor location.
      4. ``arm_length_map = sqrt((dx · k_x)² + (dy · k_y)²)``. This is the
         arm's pixel length scaled by the local floor-plane scale, which
         is what we want: a length in map units equivalent to the arm if
         it were lying on the floor at the person's location.

    Returns ``None`` when no usable side, no floor anchor, or the
    homography refuses to evaluate the local scale.
    """
    if keypoints is None or scores is None:
        return None
    try:
        if len(keypoints) <= KP_RIGHT_ELBOW or len(scores) <= KP_RIGHT_ELBOW:
            return None
    except TypeError:
        return None

    pixel_arms: List[Tuple[float, float]] = []
    for sh_idx, el_idx in (
        (KP_LEFT_SHOULDER, KP_LEFT_ELBOW),
        (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW),
    ):
        try:
            sh_score = float(scores[sh_idx])
            el_score = float(scores[el_idx])
        except (TypeError, ValueError):
            continue
        if sh_score < pose_conf_floor or el_score < pose_conf_floor:
            continue
        try:
            sh = keypoints[sh_idx]
            el = keypoints[el_idx]
            dx = float(el[0]) - float(sh[0])
            dy = float(el[1]) - float(sh[1])
        except (TypeError, IndexError, ValueError):
            continue
        if not (np.isfinite(dx) and np.isfinite(dy)):
            continue
        pixel_arms.append((dx, dy))

    if not pixel_arms:
        return None

    anchor = _floor_anchor_pixel(
        keypoints, scores, bbox, pose_conf_floor=pose_conf_floor,
    )
    if anchor is None:
        return None
    scale = _local_pixel_to_map_scale(pixel_mapper, anchor)
    if scale is None:
        return None
    k_x, k_y = scale

    lengths: List[float] = []
    for dx, dy in pixel_arms:
        L = float(np.hypot(dx * k_x, dy * k_y))
        if np.isfinite(L) and L > 0.0:
            lengths.append(L)
    if not lengths:
        return None
    return float(np.mean(lengths))


def _robust_aggregate_L(samples: List[float]) -> Optional[float]:
    """MAD-trimmed median of per-frame map-space lengths.

    Drops non-finite values, removes points outside ``median ± MAD_OUTLIER_K
    × MAD``, then returns the median of survivors. Returns ``None`` when
    fewer than ``MIN_VALID_L_SAMPLES`` survive.
    """
    arr = np.asarray([s for s in samples if s is not None and np.isfinite(s)], dtype=float)
    if arr.size < MIN_VALID_L_SAMPLES:
        return None
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad <= 0.0:
        return med
    band = MAD_OUTLIER_K * mad
    keep = arr[np.abs(arr - med) <= band]
    if keep.size < MIN_VALID_L_SAMPLES:
        return med
    return float(np.median(keep))


def _resolve_track_L(
    track_id: int,
    keypoint_details: Dict[Tuple[int, int], Any],
    pixel_mapper,
    *,
    pose_conf_floor: float,
    frame_lo: int,
    frame_hi: int,
    bbox_details: Optional[Dict[Tuple[int, int], Any]] = None,
) -> Optional[float]:
    """Compute one entrant's robust L (map units) from per-frame keypoints.

    ``frame_lo`` / ``frame_hi`` are 1-indexed inclusive. ``bbox_details``
    is optional but improves L accuracy: it lets the per-frame estimator
    fall back to the bbox bottom-center as a floor anchor when ankles
    aren't confidently visible. When omitted the estimator only succeeds
    on frames with confident ankles.

    Returns ``None`` when too few valid samples — the caller then falls
    back to the team median or the module-level constant.
    """
    samples: List[float] = []
    for fidx in range(int(frame_lo), int(frame_hi) + 1):
        kp_tuple = keypoint_details.get((fidx, int(track_id)))
        if not kp_tuple:
            continue
        kps, scores = kp_tuple
        bbox = (
            bbox_details.get((fidx, int(track_id)))
            if bbox_details is not None
            else None
        )
        L = _per_frame_shoulder_elbow_length(
            kps, scores, pixel_mapper,
            pose_conf_floor=pose_conf_floor,
            bbox=bbox,
        )
        if L is not None:
            samples.append(L)
    return _robust_aggregate_L(samples)


def _classify_frame(dist: float, L: float) -> str:
    """Bucket a distance-to-wall against a per-person band."""
    if dist < WALL_TOO_CLOSE_FRACTION * L:
        return "too_close"
    if dist > WALL_TOO_FAR_FRACTION * L:
        return "too_far"
    return "safe"


def _runs_of_label(labels: List[Optional[str]]) -> List[Tuple[str, int, int]]:
    """Run-length-encode a label list. ``None`` (unobserved) frames split runs.

    Returns ``[(label, start_idx, end_idx_inclusive), ...]`` where indices
    are 0-based positions in the input list. ``None`` itself is never
    returned as a label — those positions are just gaps.
    """
    runs: List[Tuple[str, int, int]] = []
    n = len(labels)
    i = 0
    while i < n:
        cur = labels[i]
        if cur is None:
            i += 1
            continue
        j = i + 1
        while j < n and labels[j] == cur:
            j += 1
        runs.append((cur, i, j - 1))
        i = j
    return runs


def _frame_to_time_sec(frame_1based: Optional[int], fps: float) -> Optional[float]:
    if frame_1based is None or fps <= 0:
        return None
    return round((int(frame_1based) - 1) / float(fps), 3)


# ---------------------------------------------------------------------------
# Metric class
# ---------------------------------------------------------------------------


class MoveAlongWall_Metric(AbstractMetric):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.metricName = "STAY_ALONG_WALL"

        boundary = config.get("Boundary")
        if boundary is None:
            raise ValueError("STAY_ALONG_WALL requires config['Boundary'].")
        self.boundary_region: Polygon = (
            boundary if isinstance(boundary, Polygon) else Polygon(boundary)
        )
        # Cache the exterior LinearRing once. ``Polygon.exterior`` is a
        # property; pulling it out of the per-frame distance loop saves
        # one attribute access per (track × frame) — small individually
        # but avoids unnecessary work on long drills.
        self._boundary_line = self.boundary_region.exterior

        # Pose-confidence floor: reuse the global setting (same one the
        # tracker already applies upstream). No new config key.
        self.pose_conf_floor = float(config.get("pose_conf_threshold", 0.3) or 0.3)

        self.map = config.get("Map Image", None)

        # Cap the metric + comparison to the expected team size, even if the
        # tracker happened to identify more entrants. The live analysis maps
        # / videos still render every trajectory upstream — this only limits
        # what the metric scores, flags, and the compare visualization see.
        self.num_tracks = team_size(config)

        # Populated by ``process`` and consumed by ``getFinalScore`` /
        # ``analysis.py`` via ``ctx.metric_flags`` (the side channel).
        self.scores_by_id: List[float] = []
        self._per_entrant_summary: List[Dict[str, Any]] = []
        self._excursion_records: List[Dict[str, Any]] = []

    # -- main run ------------------------------------------------------

    def process(self, ctx):
        self.scores_by_id = []
        self._per_entrant_summary = []
        self._excursion_records = []

        pixel_mapper = getattr(ctx, "pixel_mapper", None)
        keypoint_details: Dict[Tuple[int, int], Any] = (
            getattr(ctx, "keypoint_details", {}) or {}
        )
        bbox_details: Dict[Tuple[int, int], Any] = (
            getattr(ctx, "bbox_details", {}) or {}
        )

        # Per-entrant POD-capture cutoff (entrant's window ends when they reach
        # their assigned POD — same as before).
        capture_map: Dict[int, int] = {}
        for _, info in (getattr(ctx, "pod_capture", {}) or {}).items():
            assigned_id = info.get("assigned_id")
            capture_frame = info.get("capture_frame")
            if assigned_id is not None and capture_frame is not None:
                try:
                    capture_map[int(assigned_id)] = int(capture_frame)
                except Exception:
                    continue

        inroom_ids = set(getattr(ctx, "inroom_ids", []) or [])
        drill_start = int(getattr(ctx, "drill_start_frame", 1) or 1)
        drill_end = getattr(ctx, "drill_end_frame", None)
        fps = float(self.config.get("frame_rate", 30) or 30)

        # Cap to the first ``num_tracks`` entrants by entry order — same
        # rule the rest of the entry-team metrics use. Anyone beyond that
        # (e.g. a 5th tracked person when team_size is 4) is excluded from
        # the score, the per-frame violation labelling, and the emitted
        # flags. The live tracking videos still show the full trajectory
        # upstream; only this metric's view is narrowed.
        selected = select_entry_tracks(
            ctx.tracks_by_id or {},
            inroom_ids=list(inroom_ids),
            num_tracks=self.num_tracks,
        )
        selected_tracks: Dict[int, Any] = {
            int(tid): traj for tid, traj in selected
        }

        # Pass 1: per-track scoring window + per-track L
        per_track_window: Dict[int, Tuple[int, int]] = {}
        per_track_L: Dict[int, Optional[float]] = {}
        for track_id, positions in selected_tracks.items():
            tid = int(track_id)

            first = _first_valid_index(positions)
            last = _last_valid_index(positions)
            if first is None or last is None:
                continue

            entry_frame = max(first + 1, drill_start)
            end_frame = min(capture_map.get(tid, last + 1), last + 1)
            if drill_end is not None:
                end_frame = min(end_frame, int(drill_end))
            end_frame = max(entry_frame, end_frame)
            per_track_window[tid] = (entry_frame, end_frame)

            per_track_L[tid] = _resolve_track_L(
                tid,
                keypoint_details,
                pixel_mapper,
                pose_conf_floor=self.pose_conf_floor,
                frame_lo=entry_frame,
                frame_hi=end_frame,
                bbox_details=bbox_details,
            )

        # Team median fallback for entrants with too few keypoint samples.
        valid_team_L = [v for v in per_track_L.values() if v is not None]
        team_median_L = float(np.median(valid_team_L)) if valid_team_L else None

        def _resolve_L(tid: int) -> float:
            v = per_track_L.get(tid)
            if v is not None:
                return v
            if team_median_L is not None:
                return team_median_L
            return FALLBACK_SHOULDER_LENGTH_MAP

        min_excursion_frames = max(1, int(round(MIN_EXCURSION_SEC * fps)))

        # Pass 2: per-frame band classification + flag emission
        boundary_line = self._boundary_line  # local rebinding, hot-loop access
        for tid, (entry_frame, end_frame) in per_track_window.items():
            positions = selected_tracks[tid]
            n_pos = len(positions)
            L = _resolve_L(tid)
            close_thresh = WALL_TOO_CLOSE_FRACTION * L
            far_thresh = WALL_TOO_FAR_FRACTION * L

            window_len = end_frame - entry_frame + 1
            labels: List[Optional[str]] = [None] * window_len
            seen = 0
            safe = 0
            close_frames = 0
            far_frames = 0

            for i in range(window_len):
                fidx_zero = entry_frame + i - 1   # 0-indexed into positions
                if fidx_zero < 0 or fidx_zero >= n_pos:
                    continue
                pt = positions[fidx_zero]
                if pt is None:
                    continue
                seen += 1
                dist = float(boundary_line.distance(Point(float(pt[0]), float(pt[1]))))
                if dist < close_thresh:
                    labels[i] = "too_close"
                    close_frames += 1
                elif dist > far_thresh:
                    labels[i] = "too_far"
                    far_frames += 1
                else:
                    labels[i] = "safe"
                    safe += 1

            score = float(safe) / float(seen) if seen > 0 else 0.0
            self.scores_by_id.append(score)
            self._per_entrant_summary.append({
                "track_id": tid,
                "entry_frame": int(entry_frame),
                "end_frame": int(end_frame),
                "L_map": float(L),
                "L_source": (
                    "track_keypoints" if per_track_L.get(tid) is not None
                    else ("team_median" if team_median_L is not None else "fallback_constant")
                ),
                "score": float(score),
                "observed_frames": int(seen),
                "too_close_frames": int(close_frames),
                "too_far_frames": int(far_frames),
                "too_close_time_sec": round(close_frames / fps, 3) if fps > 0 else 0.0,
                "too_far_time_sec": round(far_frames / fps, 3) if fps > 0 else 0.0,
            })

            # Run-length encode the labels and emit flags for sustained
            # too_close / too_far excursions.
            for n_excursion, (label, lo, hi) in enumerate(_runs_of_label(labels), start=1):
                if label not in ("too_close", "too_far"):
                    continue
                run_len = hi - lo + 1
                if run_len < min_excursion_frames:
                    continue

                start_frame = entry_frame + lo
                end_frame_run = entry_frame + hi
                duration_sec = round(run_len / fps, 3) if fps > 0 else 0.0
                self._excursion_records.append({
                    "track_id": int(tid),
                    "label": label,
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame_run),
                    "start_time_sec": _frame_to_time_sec(start_frame, fps),
                    "end_time_sec": _frame_to_time_sec(end_frame_run, fps),
                    "duration_sec": duration_sec,
                    "L_map": float(L),
                    "n": int(n_excursion),
                })

        # Push excursions into the ctx side-channel so analysis.py can
        # surface them without the metric needing to know the schema.
        if hasattr(ctx, "metric_flags"):
            ctx.metric_flags.extend(
                self._build_flag_payloads(fps=fps)
            )

    def _build_flag_payloads(self, *, fps: float) -> List[Dict[str, Any]]:
        """Translate excursion records into the analysis flag schema."""
        out: List[Dict[str, Any]] = []
        for rec in self._excursion_records:
            label = rec["label"]
            tid = rec["track_id"]
            n = rec["n"]
            kind = "wall_too_close" if label == "too_close" else "wall_too_far"
            human = "too close to wall" if label == "too_close" else "too far from wall"
            ratio = (
                rec["duration_sec"]
            )
            out.append({
                "flag_id": f"{kind}_{tid}_{n}",
                "type": kind,
                "severity": "warning",
                "metric_id": "move_along_wall",
                "track_id": int(tid),
                "frame": int(rec["start_frame"]),
                "time_sec": rec["start_time_sec"],
                "start_frame": int(rec["start_frame"]),
                "end_frame": int(rec["end_frame"]),
                "start_time_sec": rec["start_time_sec"],
                "end_time_sec": rec["end_time_sec"],
                "duration_sec": ratio,
                "title": f"Track {tid} {human}",
                "message": (
                    f"Was {human} for {ratio:.2f}s "
                    f"(L≈{rec['L_map']:.1f} map px, "
                    f"band {WALL_TOO_CLOSE_FRACTION:.1f}–{WALL_TOO_FAR_FRACTION:.1f}×L)."
                ),
                # Excursion record kept for analysis-side timeline-item building.
                "_wall_excursion_record": rec,
            })
        return out

    def getFinalScore(self) -> float:
        if not self.scores_by_id:
            return 0.0
        return round(float(np.mean(self.scores_by_id)), 2)

    # ------------------------------------------------------------------
    # Offline trainee-vs-expert comparison (works against cached tracker
    # outputs + position caches; no live engine required).
    # ------------------------------------------------------------------

    def expertCompare(
        self,
        session_folder: str,
        expert_folder: str,
        map_image=None,
        _config: Optional[Dict[str, Any]] = None,
        **_kwargs,
    ) -> Dict[str, Any]:
        fps = float(self.config.get("frame_rate", 30) or 30)
        if fps <= 0:
            fps = 30.0

        expert_img_path = os.path.join(session_folder, "STAY_ALONG_WALL_Reference.png")
        trainee_img_path = os.path.join(session_folder, "STAY_ALONG_WALL_Trainee.png")
        txt_path = os.path.join(session_folder, "STAY_ALONG_WALL_Comparison.txt")
        os.makedirs(session_folder, exist_ok=True)

        def _coerce_map_image(img_or_path):
            if img_or_path is None:
                return None
            if isinstance(img_or_path, str):
                return cv2.imread(img_or_path) if os.path.exists(img_or_path) else None
            return img_or_path

        if map_image is None:
            map_image = self.map
        map_image = _coerce_map_image(map_image)

        if map_image is None:
            err_text = "There was an error while processing this comparison. Missing map image."
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(err_text)
            return {
                "Name": "STAY_ALONG_WALL",
                "Type": "SideBySide",
                "ReferenceImageLocation": expert_img_path,
                "TraineeImageLocation": trainee_img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        # Construct a local PixelMapper from the same point-mapping file the
        # live run used. ``load_config`` already resolved the path to absolute.
        pixel_mapper = self._build_local_pixel_mapper()
        if pixel_mapper is None:
            err_text = (
                "There was an error while processing this comparison. "
                "Could not load PixelMapper from config['point_mapping_path']."
            )
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(err_text)
            return {
                "Name": "STAY_ALONG_WALL",
                "Type": "SideBySide",
                "ReferenceImageLocation": expert_img_path,
                "TraineeImageLocation": trainee_img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        try:
            expert_tracks = _load_position_cache(expert_folder)
            trainee_tracks = _load_position_cache(session_folder)
        except Exception:
            err_text = "There was an error while processing this comparison. Missing or invalid PositionCache."
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(err_text)
            return {
                "Name": "STAY_ALONG_WALL",
                "Type": "SideBySide",
                "ReferenceImageLocation": expert_img_path,
                "TraineeImageLocation": trainee_img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        if not expert_tracks or not trainee_tracks:
            err_text = "There was an error while processing this comparison. No valid tracks found."
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(err_text)
            return {
                "Name": "STAY_ALONG_WALL",
                "Type": "SideBySide",
                "ReferenceImageLocation": expert_img_path,
                "TraineeImageLocation": trainee_img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        expert_kp = _load_keypoint_details(expert_folder)
        trainee_kp = _load_keypoint_details(session_folder)
        expert_bbox = _load_bbox_details(expert_folder)
        trainee_bbox = _load_bbox_details(session_folder)

        expert_capture = _load_capture_frames(expert_folder)
        trainee_capture = _load_capture_frames(session_folder)

        expert_entry = _entry_map(expert_tracks)
        trainee_entry = _entry_map(trainee_tracks)

        # Each run's video may have been recorded at a different fps. Read
        # the live-recorded fps from each folder's RunMetadata sidecar
        # separately so the seconds-based outputs (excursion durations,
        # too-close / too-far time totals, min-excursion frame thresholds)
        # match what the live engine computed for that run.
        from ..utils.run_metadata import resolve_fps_from_metadata
        trainee_fps = resolve_fps_from_metadata(session_folder, fallback=fps) or fps
        reference_fps = resolve_fps_from_metadata(expert_folder, fallback=fps) or fps

        # Match the live engine's window: it scores this metric over
        # [frame 1 .. drill_end]. Each run's drill end lives in its own
        # DrillWindow.json sidecar.
        reference_drill_end = _load_drill_end(expert_folder)
        trainee_drill_end = _load_drill_end(session_folder)

        expert_infos = self._prep_infos(
            expert_tracks, expert_capture, expert_entry,
            keypoint_details=expert_kp, bbox_details=expert_bbox,
            pixel_mapper=pixel_mapper, fps=reference_fps,
            drill_end=reference_drill_end,
        )
        trainee_infos = self._prep_infos(
            trainee_tracks, trainee_capture, trainee_entry,
            keypoint_details=trainee_kp, bbox_details=trainee_bbox,
            pixel_mapper=pixel_mapper, fps=trainee_fps,
            drill_end=trainee_drill_end,
        )

        # Pull the wall-violation flag windows + per-track total durations
        # the live engine wrote into each run's Analysis.json. No
        # reclassification — we just trust what was saved.
        reference_windows, reference_seconds = _load_wall_flags(expert_folder)
        trainee_windows, trainee_seconds = _load_wall_flags(session_folder)

        self.__generateExpertCompareGraphic(
            output_folder=session_folder,
            map_view=map_image,
            infos=expert_infos,
            flag_windows_by_track=reference_windows,
            outside_seconds_by_track=reference_seconds,
            out_name="STAY_ALONG_WALL_Reference.png",
            title="Reference",
            fps=reference_fps,
        )
        self.__generateExpertCompareGraphic(
            output_folder=session_folder,
            map_view=map_image,
            infos=trainee_infos,
            flag_windows_by_track=trainee_windows,
            outside_seconds_by_track=trainee_seconds,
            out_name="STAY_ALONG_WALL_Trainee.png",
            title="Trainee",
            fps=trainee_fps,
        )

        text, saved_text = _format_comparison_report(expert_infos, trainee_infos)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(saved_text)

        return {
            "Name": "STAY_ALONG_WALL",
            "Type": "SideBySide",
            "ReferenceImageLocation": expert_img_path,
            "TraineeImageLocation": trainee_img_path,
            "TxtLocation": txt_path,
            "Text": text,
        }

    def _build_local_pixel_mapper(self):
        """Build a PixelMapper from this metric's config. None on failure."""
        path = self.config.get("point_mapping_path")
        if not isinstance(path, str) or not path:
            return None
        try:
            from libs.Track.processing.utils import load_pixel_mapper
            return load_pixel_mapper(
                path,
                ransac_reproj_threshold=float(self.config.get("homography_ransac_thresh", 3.0)),
                confidence=float(self.config.get("homography_confidence", 0.999)),
                max_iters=int(self.config.get("homography_max_iters", 2000)),
            )
        except Exception:
            return None

    def _prep_infos(
        self,
        track_dicts: List[Dict[str, Any]],
        capture_map: Dict[int, int],
        entry_map: Dict[int, int],
        *,
        keypoint_details: Dict[Tuple[int, int], Any],
        bbox_details: Optional[Dict[Tuple[int, int], Any]],
        pixel_mapper,
        fps: float,
        drill_end: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Compute per-entrant L, score, excursions for offline comparison.

        ``drill_end`` clamps each entrant's scoring window the same way the
        live engine does (it runs this metric with ``drill_start=1`` but caps
        the end at the detected drill end). Passing it keeps the recomputed
        per-entrant scores / time totals identical to the run's saved score;
        omitting it would count post-drill wandering the live run excluded.
        """
        # Match the live ``process()`` rule: keep only the first
        # ``num_tracks`` entrants by entry order. Anyone beyond that (e.g.
        # a 5th tracked person when team_size is 4) is excluded from the
        # comparison view and from the per-side score so the offline path
        # stays in lock-step with what the metric scored live.
        tracks_by_id = {int(t["id"]): t["traj"] for t in track_dicts}
        capped = select_entry_tracks(
            tracks_by_id,
            num_tracks=self.num_tracks,
        )
        capped_ids = {int(tid) for tid, _ in capped}
        track_dicts = [t for t in track_dicts if int(t["id"]) in capped_ids]

        per_track_L: Dict[int, Optional[float]] = {}
        windows: Dict[int, Tuple[int, int]] = {}

        for track in track_dicts:
            tid = int(track["id"])
            traj = track["traj"]
            first = _first_valid_index(traj)
            last = _last_valid_index(traj)
            if first is None or last is None:
                continue
            entry_frame = first + 1
            last_seen = last + 1
            end_frame = min(int(capture_map.get(tid, last_seen)), last_seen)
            if drill_end is not None:
                end_frame = min(end_frame, int(drill_end))
            end_frame = max(entry_frame, end_frame)
            windows[tid] = (entry_frame, end_frame)

            per_track_L[tid] = _resolve_track_L(
                tid,
                keypoint_details,
                pixel_mapper,
                pose_conf_floor=self.pose_conf_floor,
                frame_lo=entry_frame,
                frame_hi=end_frame,
                bbox_details=bbox_details,
            )

        team_valid = [v for v in per_track_L.values() if v is not None]
        team_median = float(np.median(team_valid)) if team_valid else None
        boundary_line = self._boundary_line  # hot-loop alias

        infos: List[Dict[str, Any]] = []
        for track in track_dicts:
            tid = int(track["id"])
            if tid not in windows:
                continue
            entry_frame, end_frame = windows[tid]
            traj = track["traj"]
            n_pos = len(traj)

            L = per_track_L.get(tid)
            if L is None:
                L = team_median if team_median is not None else FALLBACK_SHOULDER_LENGTH_MAP
            L_source = (
                "track_keypoints" if per_track_L.get(tid) is not None
                else ("team_median" if team_median is not None else "fallback_constant")
            )

            close_thresh = WALL_TOO_CLOSE_FRACTION * L
            far_thresh = WALL_TOO_FAR_FRACTION * L

            window_len = end_frame - entry_frame + 1
            labels: List[Optional[str]] = [None] * window_len
            seen = 0
            safe = 0
            close_frames = 0
            far_frames = 0
            for i in range(window_len):
                fidx_zero = entry_frame + i - 1
                if fidx_zero < 0 or fidx_zero >= n_pos:
                    continue
                pt = traj[fidx_zero]
                if pt is None:
                    continue
                seen += 1
                dist = float(boundary_line.distance(Point(float(pt[0]), float(pt[1]))))
                if dist < close_thresh:
                    labels[i] = "too_close"
                    close_frames += 1
                elif dist > far_thresh:
                    labels[i] = "too_far"
                    far_frames += 1
                else:
                    labels[i] = "safe"
                    safe += 1

            # Per-frame violation classification is still done here so the
            # score and the too_close_time_sec / too_far_time_sec totals
            # remain identical to what the live engine produced. We no
            # longer expose the run-length segments because the comparison
            # graphic reads the flag windows directly from the saved
            # Analysis.json via ``_load_wall_flags`` — same data source the
            # timeline uses, so the two views can't drift apart.
            score = float(safe) / float(seen) if seen > 0 else 0.0
            infos.append({
                "track_id": tid,
                "entry_number": int(entry_map.get(tid, 0)) or None,
                "traj": traj,
                "entry_frame": entry_frame,
                "end_frame": end_frame,
                "L_map": float(L),
                "L_source": L_source,
                "score": float(score),
                "observed_frames": int(seen),
                "too_close_time_sec": round(close_frames / fps, 3) if fps > 0 else 0.0,
                "too_far_time_sec": round(far_frames / fps, 3) if fps > 0 else 0.0,
                "reached_pod": tid in capture_map,
            })

        infos.sort(key=lambda d: int(d["entry_number"] or 10**9))
        return infos

    @staticmethod
    def __generateExpertCompareGraphic(
        output_folder: str,
        map_view,
        infos: List[Dict[str, Any]],
        flag_windows_by_track: Dict[int, List[Tuple[int, int]]],
        outside_seconds_by_track: Dict[int, float],
        out_name: str,
        title: str,
        fps: float = 30.0,
    ) -> None:
        os.makedirs(output_folder, exist_ok=True)
        img = map_view.copy()
        h, w = img.shape[:2]
        original_w = w

        predefined_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128),
        ]

        def _color_for_id(track_id: int):
            return predefined_colors[int(track_id) % len(predefined_colors)]

        # No room-boundary overlay drawn here. The source map already shows
        # the room geometry, and the live tracking videos (annotate_map_video)
        # don't add one — matching their behaviour keeps the visual identical
        # so users don't see "extra" boundary lines unique to the comparison.

        # Two-pass render — highlighter metaphor.
        # Pass 1 lays down a soft semi-transparent BAND in the entrant's
        # colour along the frames inside each flagged-excursion window
        # (``flag_windows_by_track``). Pass 2 draws every entrant's full
        # trajectory as a thin smooth line on top of the highlighter band,
        # so the line stays clean while the band reads as "this stretch is
        # highlighted." The flag windows come straight from the run's
        # Analysis.json so the highlight matches the saved flags exactly.

        # ---- Pass 1: highlighter band (drawn first, sits BENEATH the line)
        #
        # We composite the band on a single working copy so a per-pixel
        # alpha blend can soften it against the map. Drawing each segment
        # at full opacity onto ``overlay`` and then alpha-blending with
        # ``img`` gives a uniform translucency without per-segment alpha
        # arithmetic.
        BAND_THICK = 12       # generous so it's visible behind the thin line
        BAND_ALPHA = 0.35     # soft — the map detail still shows through
        overlay = img.copy()
        any_band = False
        for info in infos:
            tid = int(info["track_id"])
            traj = info["traj"]
            track_color = _color_for_id(tid)
            for s_frame, e_frame in flag_windows_by_track.get(tid, []):
                prev_pt: Optional[Tuple[int, int]] = None
                for fidx in range(s_frame, e_frame + 1):
                    pt_xy = traj[fidx - 1] if 0 <= (fidx - 1) < len(traj) else None
                    if pt_xy is None:
                        prev_pt = None
                        continue
                    pt = (int(round(float(pt_xy[0]))), int(round(float(pt_xy[1]))))
                    if prev_pt is not None:
                        cv2.line(overlay, prev_pt, pt, track_color, BAND_THICK, cv2.LINE_AA)
                        any_band = True
                    prev_pt = pt
        if any_band:
            cv2.addWeighted(overlay, BAND_ALPHA, img, 1.0 - BAND_ALPHA, 0, dst=img)

        # ---- Pass 2: full trajectory line in track colour, drawn on top.
        #     Stays thin and smooth so the highlighter beneath does the
        #     "this stretch is flagged" work without disturbing the path.
        LINE_THICK = 2
        for info in infos:
            tid = int(info["track_id"])
            traj = info["traj"]
            entry_frame = int(info["entry_frame"])
            end_frame = int(info["end_frame"])
            track_color = _color_for_id(tid)

            prev_pt = None
            for fidx in range(entry_frame, end_frame + 1):
                pt_xy = traj[fidx - 1] if 0 <= (fidx - 1) < len(traj) else None
                if pt_xy is None:
                    prev_pt = None
                    continue
                pt = (int(round(float(pt_xy[0]))), int(round(float(pt_xy[1]))))
                if prev_pt is not None:
                    cv2.line(img, prev_pt, pt, track_color, LINE_THICK, cv2.LINE_AA)
                prev_pt = pt

        # Legend panel — text rendered via the shared TrueType helper so
        # glyphs read as crisp typed text instead of OpenCV's wavy stroke
        # fonts. Geometry (rectangles, colour swatches, highlighter swatch)
        # still uses cv2 — only the strings go through PIL.
        from .utils import draw_text, text_size

        TITLE_SIZE = 17
        BODY_SIZE = 14
        pad = 14
        sw = 14
        gap = 10
        line_h = 24

        # Per-entrant off-wall RATIO shown in the legend: the flagged
        # wall-excursion time (summed straight from the wall flags the live
        # engine saved in Analysis.json, via ``outside_seconds_by_track``)
        # divided by the entrant's total evaluated window — (end_frame -
        # entry_frame + 1) / fps, the same window the metric scored over.
        # Expressed as a percentage so it's comparable across entrants with
        # different window lengths. An entrant with no flags shows 0%
        # (sub-MIN_EXCURSION blips aren't surfaced as flags, so the legend
        # stays in step with the timeline / highlighter).
        NOTE_SIZE = 12
        note_lines = [
            "% off-wall = flagged off-wall time",
            "÷ time evaluated for that entrant",
        ]

        items: List[Tuple[int, int, float]] = []  # (entry_number, track_id, off_wall_pct)
        for info in infos:
            entry_number = info.get("entry_number")
            track_id = info.get("track_id")
            if entry_number is None or track_id is None:
                continue
            sec = float(outside_seconds_by_track.get(int(track_id), 0.0))
            window_frames = int(info.get("end_frame", 0)) - int(info.get("entry_frame", 0)) + 1
            considered_sec = (window_frames / fps) if (fps > 0 and window_frames > 0) else 0.0
            pct = 100.0 * min(1.0, max(0.0, sec / considered_sec)) if considered_sec > 0 else 0.0
            items.append((int(entry_number), int(track_id), pct))
        items.sort(key=lambda x: x[0])

        def _entrant_label(entry: int, pct: float) -> str:
            return f"Entrant #{entry} · {pct:.0f}% off-wall"

        # Width = widest TrueType-rendered string (title in title size,
        # per-entrant + highlight rows in body size, note in note size).
        max_w = text_size(title, size_px=TITLE_SIZE)[0]
        for s in ["Highlighter = flagged wall excursion"] + [
            _entrant_label(entry, pct) for entry, _, pct in items
        ]:
            max_w = max(max_w, text_size(s, size_px=BODY_SIZE)[0])
        for s in note_lines:
            max_w = max(max_w, text_size(s, size_px=NOTE_SIZE)[0])

        # title + highlight + per-entrant rows + a small gap + note lines
        n_lines = 2 + len(items) + len(note_lines)
        panel_w = pad * 2 + sw + gap + max_w
        panel_h = pad * 2 + line_h * max(1, n_lines)

        extra_right = panel_w + pad * 2
        bg = tuple(int(x) for x in img[0, 0].tolist())
        canvas = np.full((h, original_w + extra_right, 3), bg, dtype=np.uint8)
        canvas[:, :original_w] = img
        img = canvas

        x0 = original_w + pad
        y0 = pad
        cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), -1)
        cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (220, 220, 220), 2)

        # Title row.
        y = y0 + pad
        draw_text(img, title, (x0 + pad, y), size_px=TITLE_SIZE, fill_bgr=(255, 255, 255))
        y += line_h

        # Highlight legend row: a wide soft band (mimicking the highlighter
        # behind the line) crossed by a thin neutral grey line on top.
        # Colour-agnostic — represents the styling, not a specific entrant.
        seg_y = y + (line_h // 2)
        seg_x0 = x0 + pad
        seg_x1 = seg_x0 + sw
        swatch_band_color = (180, 180, 180)
        swatch_line_color = (220, 220, 220)
        legend_overlay = img.copy()
        cv2.line(legend_overlay, (seg_x0, seg_y), (seg_x1, seg_y), swatch_band_color, 11, cv2.LINE_AA)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.line(mask, (seg_x0, seg_y), (seg_x1, seg_y), 255, 11, cv2.LINE_AA)
        m3 = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
        blended = (img.astype(np.float32) * (1.0 - 0.45 * m3)
                   + legend_overlay.astype(np.float32) * (0.45 * m3))
        img[:] = blended.astype(np.uint8)
        cv2.line(img, (seg_x0, seg_y), (seg_x1, seg_y), swatch_line_color, 2, cv2.LINE_AA)
        draw_text(
            img, "Highlighter = flagged wall excursion",
            (x0 + pad + sw + gap, y), size_px=BODY_SIZE, fill_bgr=(255, 255, 255),
        )
        y += line_h

        for entry_number, track_id, off_wall_pct in items:
            color = _color_for_id(track_id)
            cv2.rectangle(img, (x0 + pad, y + 4), (x0 + pad + sw, y + 4 + sw), color, -1)
            cv2.rectangle(img, (x0 + pad, y + 4), (x0 + pad + sw, y + 4 + sw), (255, 255, 255), 1)
            draw_text(
                img,
                _entrant_label(entry_number, off_wall_pct),
                (x0 + pad + sw + gap, y),
                size_px=BODY_SIZE,
                fill_bgr=(255, 255, 255),
            )
            y += line_h

        # Explanatory note at the bottom of the legend (dimmer + smaller so it
        # reads as a caption, not another entrant row).
        for note in note_lines:
            draw_text(
                img,
                note,
                (x0 + pad, y),
                size_px=NOTE_SIZE,
                fill_bgr=(180, 180, 180),
            )
            y += int(round(NOTE_SIZE * 1.5))

        cv2.imwrite(os.path.join(output_folder, out_name), img)


# ---------------------------------------------------------------------------
# Module-level helpers used by both the metric and expertCompare.
# ---------------------------------------------------------------------------


def _first_valid_index(traj: List[Optional[Tuple[float, float]]]) -> Optional[int]:
    for i, value in enumerate(traj):
        if value is not None:
            return i
    return None


def _last_valid_index(traj: List[Optional[Tuple[float, float]]]) -> Optional[int]:
    for i in range(len(traj) - 1, -1, -1):
        if traj[i] is not None:
            return i
    return None


def _load_position_cache(folder: str) -> List[Dict[str, Any]]:
    """Read ``*_PositionCache.txt`` from ``folder``. Excludes inroom IDs."""
    cache_path = pick_latest(folder, "*_PositionCache.txt")
    if cache_path is None:
        raise FileNotFoundError(f"No PositionCache found in {folder}")

    df = pd.read_csv(cache_path)
    cols = {c.strip().lower(): c for c in df.columns}
    frame_col = cols.get("frame")
    id_col = cols.get("id")
    x_col = cols.get("mapx")
    y_col = cols.get("mapy")
    if frame_col is None or id_col is None or x_col is None or y_col is None:
        raise ValueError(f"Unexpected PositionCache format: {cache_path}")

    df = df[[frame_col, id_col, x_col, y_col]].dropna()
    if df.empty:
        return []

    df[frame_col] = df[frame_col].astype(int)
    df[id_col] = df[id_col].astype(int)
    df[x_col] = df[x_col].astype(float)
    df[y_col] = df[y_col].astype(float)

    inroom_ids = load_inroom_ids(folder)
    df = df[~df[id_col].isin(list(inroom_ids))].copy()
    if df.empty:
        return []

    max_frame = int(df[frame_col].max())
    tracks: Dict[int, List[Optional[Tuple[float, float]]]] = {}
    for tid, g in df.groupby(id_col):
        tid_i = int(tid)
        traj: List[Optional[Tuple[float, float]]] = [None] * max_frame
        for _, row in g.iterrows():
            fidx = int(row[frame_col])
            if 1 <= fidx <= max_frame:
                traj[fidx - 1] = (float(row[x_col]), float(row[y_col]))
        tracks[tid_i] = traj
    return [{"id": int(tid), "traj": traj} for tid, traj in tracks.items()]


def _load_keypoint_details(folder: str) -> Dict[Tuple[int, int], Tuple[List[Any], List[Any]]]:
    """Load per-frame, per-track keypoints from ``*_TrackerOutput.json``.

    Returns a dict with the same shape as ``MetricContext.keypoint_details``:
    ``{(frame_1based, track_id): (keypoints_list, scores_list)}``. Returns an
    empty dict on missing/malformed input — callers fall back to the team
    median or constant L.
    """
    import json

    path = pick_latest(folder, "*_TrackerOutput.json")
    out: Dict[Tuple[int, int], Tuple[List[Any], List[Any]]] = {}
    if path is None:
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return out
    if not isinstance(data, list):
        return out

    for frame_entry in data:
        try:
            fidx = int(frame_entry.get("frame", -1))
        except (TypeError, ValueError):
            continue
        if fidx < 0:
            continue
        for obj in frame_entry.get("objects", []) or []:
            tid = obj.get("id")
            if tid is None:
                continue
            kps = obj.get("keypoints")
            scores = obj.get("keypoint_scores")
            if kps is None or scores is None:
                continue
            try:
                out[(fidx, int(tid))] = (kps, scores)
            except (TypeError, ValueError):
                continue
    return out


def _load_bbox_details(folder: str) -> Dict[Tuple[int, int], Tuple[float, float, float, float]]:
    """Load per-frame, per-track bounding boxes from ``*_TrackerOutput.json``.

    Bbox format mirrors ``MetricContext.bbox_details``:
    ``{(frame_1based, track_id): (x1, y1, x2, y2)}``. Returns an empty dict
    when the cache is missing or malformed — the L estimator then falls
    back to the ankle-only path for floor-anchor selection.
    """
    import json

    path = pick_latest(folder, "*_TrackerOutput.json")
    out: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}
    if path is None:
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return out
    if not isinstance(data, list):
        return out

    for frame_entry in data:
        try:
            fidx = int(frame_entry.get("frame", -1))
        except (TypeError, ValueError):
            continue
        if fidx < 0:
            continue
        for obj in frame_entry.get("objects", []) or []:
            tid = obj.get("id")
            bbox = obj.get("bbox")
            if tid is None or not bbox or len(bbox) < 4:
                continue
            try:
                out[(fidx, int(tid))] = (
                    float(bbox[0]), float(bbox[1]),
                    float(bbox[2]), float(bbox[3]),
                )
            except (TypeError, ValueError):
                continue
    return out


def _load_capture_frames(folder: str) -> Dict[int, int]:
    cache_path = pick_latest(folder, "*_PodCache.txt")
    if cache_path is None:
        return {}
    try:
        df = pd.read_csv(cache_path)
    except Exception:
        return {}
    if df is None or df.empty:
        return {}
    cols = {c.strip().lower(): c for c in df.columns}
    assigned_col = cols.get("assigned_id")
    capture_frame_col = cols.get("capture_frame")
    if assigned_col is None or capture_frame_col is None:
        return {}
    capture_map: Dict[int, int] = {}
    for _, row in df[[assigned_col, capture_frame_col]].dropna().iterrows():
        try:
            capture_map[int(row[assigned_col])] = int(row[capture_frame_col])
        except Exception:
            continue
    return capture_map


def _track_id_from_flag(flag: Dict[str, Any]) -> Optional[int]:
    """Extract the track_id for a wall flag.

    Prefers an explicit ``track_id`` field. Falls back to parsing the
    ``flag_id`` string, which is emitted by ``_build_flag_payloads`` as
    ``f"{type}_{tid}_{n}"`` — this keeps the helper compatible with run
    folders produced before ``track_id`` was added to the v2 flag schema.
    """
    tid = flag.get("track_id")
    if tid is not None:
        try:
            return int(tid)
        except (TypeError, ValueError):
            pass

    flag_id = flag.get("flag_id")
    ftype = flag.get("type")
    if isinstance(flag_id, str) and isinstance(ftype, str) and flag_id.startswith(ftype + "_"):
        rest = flag_id[len(ftype) + 1:]  # drop "<type>_" prefix
        # remaining form is "<tid>_<n>" — pull the int before the last "_"
        underscore = rest.rfind("_")
        candidate = rest[:underscore] if underscore > 0 else rest
        try:
            return int(candidate)
        except (TypeError, ValueError):
            return None
    return None


def _load_drill_end(folder: str) -> Optional[int]:
    """Read ``end_frame`` from a run's ``*_DrillWindow.json`` sidecar.

    Returns None when the sidecar is missing/unreadable so the caller falls
    back to the full track length (matching pre-drill-window runs).
    """
    import json

    path = pick_latest(folder, "*_DrillWindow.json")
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        end = data.get("end_frame") if isinstance(data, dict) else None
        return int(end) if end is not None else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _load_wall_flags(folder: str) -> Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, float]]:
    """Read wall-violation flag windows + per-track total durations.

    Returns ``(windows_by_track, seconds_by_track)`` where:

    - ``windows_by_track`` = ``{track_id: [(start_frame, end_frame), ...]}``
      Used by ``__generateExpertCompareGraphic`` to highlight the
      trajectory stretch inside each flag.

    - ``seconds_by_track`` = ``{track_id: total_duration_sec}``
      Sum of ``duration_sec`` over every wall_too_close / wall_too_far
      flag for the track. Surfaced in the legend so the user sees exactly
      the time the live engine committed to the timeline — no
      reclassification, no fps multiplication, no rounding drift.

    Both maps cover ``wall_too_close`` and ``wall_too_far`` flags
    indistinguishably (the visualization deliberately doesn't separate
    them — see the original simplification request).
    """
    import json

    path = pick_latest(folder, "*_Analysis.json")
    windows: Dict[int, List[Tuple[int, int]]] = {}
    seconds: Dict[int, float] = {}
    if path is None:
        return windows, seconds
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return windows, seconds
    if not isinstance(data, dict):
        return windows, seconds

    for flag in data.get("flags") or []:
        if not isinstance(flag, dict):
            continue
        if flag.get("type") not in ("wall_too_close", "wall_too_far"):
            continue
        tid = _track_id_from_flag(flag)
        if tid is None:
            continue
        try:
            s = int(flag["start_frame"])
            e = int(flag["end_frame"])
        except (KeyError, TypeError, ValueError):
            continue
        if e < s:
            s, e = e, s
        windows.setdefault(tid, []).append((s, e))

        # Prefer the saved ``duration_sec`` field; fall back to deriving it
        # from ``start_time_sec`` / ``end_time_sec`` if duration is missing
        # (older flag emissions). Both came from the live engine's fps so
        # no reclassification is happening here either way.
        d = flag.get("duration_sec")
        if d is None:
            s_t = flag.get("start_time_sec")
            e_t = flag.get("end_time_sec")
            if s_t is not None and e_t is not None:
                try:
                    d = float(e_t) - float(s_t)
                except (TypeError, ValueError):
                    d = None
        try:
            d_f = float(d) if d is not None else 0.0
        except (TypeError, ValueError):
            d_f = 0.0
        if d_f > 0:
            seconds[tid] = seconds.get(tid, 0.0) + d_f

    # Sort per-track windows by start_frame so downstream rendering is
    # deterministic.
    for tid in windows:
        windows[tid].sort()
    return windows, seconds


def _entry_map(track_dicts: List[Dict[str, Any]]) -> Dict[int, int]:
    starts = []
    for track in track_dicts:
        tid = int(track["id"])
        first = _first_valid_index(track["traj"])
        if first is not None:
            starts.append((tid, int(first)))
    starts.sort(key=lambda x: x[1])
    return {tid: i + 1 for i, (tid, _) in enumerate(starts)}


def _format_comparison_report(
    expert_infos: List[Dict[str, Any]],
    trainee_infos: List[Dict[str, Any]],
) -> Tuple[str, str]:
    """Produce the (text_returned_to_caller, text_saved_to_disk) pair."""
    expert_scores = [float(i.get("score", 0.0)) for i in expert_infos]
    trainee_scores = [float(i.get("score", 0.0)) for i in trainee_infos]
    expert_close = [float(i.get("too_close_time_sec", 0.0)) for i in expert_infos]
    trainee_close = [float(i.get("too_close_time_sec", 0.0)) for i in trainee_infos]
    expert_far = [float(i.get("too_far_time_sec", 0.0)) for i in expert_infos]
    trainee_far = [float(i.get("too_far_time_sec", 0.0)) for i in trainee_infos]

    e_mean = float(np.mean(expert_scores)) if expert_scores else 0.0
    t_mean = float(np.mean(trainee_scores)) if trainee_scores else 0.0
    e_close_mean = float(np.mean(expert_close)) if expert_close else 0.0
    t_close_mean = float(np.mean(trainee_close)) if trainee_close else 0.0
    e_far_mean = float(np.mean(expert_far)) if expert_far else 0.0
    t_far_mean = float(np.mean(trainee_far)) if trainee_far else 0.0

    score_part = (
        f"Trainee score {t_mean:.2f} vs Reference {e_mean:.2f} "
        f"(Δ {(t_mean - e_mean):+.2f})."
    )
    time_part = (
        f"Trainee avg {t_close_mean:.2f}s too close + {t_far_mean:.2f}s too far; "
        f"Reference avg {e_close_mean:.2f}s too close + {e_far_mean:.2f}s too far."
    )

    header = (
        "Entry #, Trainee ID, T Score, T Too-Close (s), T Too-Far (s), T L_map (px), "
        "Reference ID, R Score, R Too-Close (s), R Too-Far (s), R L_map (px), "
        "Score Δ (T-R), Performance"
    )

    eps_score = 0.01
    eps_time = 0.10
    rows: List[str] = []
    max_n = max(len(expert_infos), len(trainee_infos))
    for i in range(max_n):
        ei = expert_infos[i] if i < len(expert_infos) else None
        ti = trainee_infos[i] if i < len(trainee_infos) else None

        e_id = ei.get("track_id") if ei else None
        t_id = ti.get("track_id") if ti else None
        e_sc = ei.get("score") if ei else None
        t_sc = ti.get("score") if ti else None
        e_cl = ei.get("too_close_time_sec") if ei else None
        t_cl = ti.get("too_close_time_sec") if ti else None
        e_fr = ei.get("too_far_time_sec") if ei else None
        t_fr = ti.get("too_far_time_sec") if ti else None
        e_L = ei.get("L_map") if ei else None
        t_L = ti.get("L_map") if ti else None

        if e_sc is not None and t_sc is not None:
            score_diff = float(t_sc) - float(e_sc)
            score_diff_str = f"{score_diff:+.2f}"
        else:
            score_diff = None
            score_diff_str = "N/A"

        # Performance verdict combines score delta and total-violation-time.
        if score_diff is None:
            performance = "N/A"
        else:
            t_violation = (float(t_cl or 0.0) + float(t_fr or 0.0))
            e_violation = (float(e_cl or 0.0) + float(e_fr or 0.0))
            t_diff = t_violation - e_violation
            if abs(score_diff) <= eps_score:
                if abs(t_diff) <= eps_time:
                    performance = "SIMILAR"
                elif t_diff < 0:
                    performance = "BETTER"
                else:
                    performance = "WORSE"
            elif score_diff > eps_score:
                performance = "BETTER"
            else:
                performance = "WORSE"

        rows.append(
            f"{i + 1}, "
            f"{t_id if t_id is not None else 'N/A'}, "
            f"{('N/A' if t_sc is None else f'{float(t_sc):.2f}')}, "
            f"{('N/A' if t_cl is None else f'{float(t_cl):.2f}')}, "
            f"{('N/A' if t_fr is None else f'{float(t_fr):.2f}')}, "
            f"{('N/A' if t_L is None else f'{float(t_L):.1f}')}, "
            f"{e_id if e_id is not None else 'N/A'}, "
            f"{('N/A' if e_sc is None else f'{float(e_sc):.2f}')}, "
            f"{('N/A' if e_cl is None else f'{float(e_cl):.2f}')}, "
            f"{('N/A' if e_fr is None else f'{float(e_fr):.2f}')}, "
            f"{('N/A' if e_L is None else f'{float(e_L):.1f}')}, "
            f"{score_diff_str}, {performance}"
        )

    details_csv = header + "\n" + "\n".join(rows)
    text = score_part + "\n" + time_part + "\n\n" + details_csv

    pretty_headers = [h.strip() for h in header.split(",")]
    pretty_rows: List[List[str]] = []
    for row in rows:
        parts = [p.strip() for p in row.split(",")]
        while len(parts) < len(pretty_headers):
            parts.append("N/A")
        pretty_rows.append(parts[: len(pretty_headers)])

    if pretty_rows:
        widths = [
            max([len(h)] + [len(r[j]) for r in pretty_rows])
            for j, h in enumerate(pretty_headers)
        ]
        sep = " | "
        lines = [sep.join(h.ljust(widths[i]) for i, h in enumerate(pretty_headers))]
        lines.append(sep.join("-" * w for w in widths))
        for r in pretty_rows:
            lines.append(sep.join(r[i].ljust(widths[i]) for i in range(len(pretty_headers))))
        details_pretty = "\n".join(lines)
    else:
        details_pretty = "(no rows)"

    saved_text = score_part + "\n" + time_part + "\n\n" + details_pretty + "\n"
    return text, saved_text
