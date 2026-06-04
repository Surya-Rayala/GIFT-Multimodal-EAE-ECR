"""Shared helpers used by multiple metrics in src/metrics/.

Each helper here was previously duplicated verbatim (or near-verbatim) across
several metric modules. Behavior matches the pre-existing copies; the guards
inside ``exponential_time_penalty`` are the union of the three call sites so
that all current call paths behave identically to before.
"""

import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np


def pick_latest(folder: str, pattern: Union[str, Iterable[str]]) -> Optional[str]:
    """Return the most-recently-modified file in ``folder`` matching ``pattern``.

    ``pattern`` accepts either a single glob string (the common case) or an
    iterable of glob strings (the ``teammate_coverage`` fallback form). Returns
    ``None`` when nothing matches.
    """
    if isinstance(pattern, str):
        matches = glob.glob(os.path.join(folder, pattern))
    else:
        matches: List[str] = []
        for pat in pattern:
            matches.extend(glob.glob(os.path.join(folder, pat)))
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def load_inroom_ids(folder: str) -> Set[int]:
    """Extract in-room track IDs from the latest ``*_TrackerOutput.json`` in ``folder``.

    A track counts as in-room when its object dict has ``identity_role == "inroom"``
    or ``is_inroom`` truthy. Returns an empty set on any I/O or parse failure.
    """
    tracker_path = pick_latest(folder, "*_TrackerOutput.json")
    if tracker_path is None:
        return set()

    try:
        with open(tracker_path, "r") as f:
            tracker_output = json.load(f)
    except Exception:
        return set()

    inroom: Set[int] = set()
    for frame_entry in tracker_output:
        for obj in frame_entry.get("objects", []):
            tid = obj.get("id")
            if tid is None:
                continue
            if obj.get("identity_role") == "inroom" or obj.get("is_inroom", False):
                inroom.add(int(tid))
    return inroom


def exponential_time_penalty(overrun: float, limit: float) -> float:
    """Score an overrun time against a limit.

    Returns 0.0 when ``limit <= 0`` or ``overrun >= limit``. Otherwise returns
    ``exp(-overrun / (limit - overrun))`` — a smooth decay that approaches 0
    as overrun approaches limit.
    """
    if limit <= 0:
        return 0.0
    if overrun >= limit:
        return 0.0
    return float(np.exp(-overrun / (limit - overrun)))


def gaze_cone_triangle(origin, direction, half_angle_deg, length: float = 10000.0):
    """Build a 2D isoceles triangle representing a gaze cone.

    Apex is at ``origin``; the two other vertices are at distance ``length``,
    rotated ``±half_angle_deg`` from the normalized ``direction`` vector.
    Returns a ``(3, 2)`` float32 array. When ``direction`` is zero-length,
    returns a zero triangle.
    """
    d = np.asarray(direction, dtype=np.float32)
    n = np.linalg.norm(d)
    if n == 0.0:
        return np.zeros((3, 2), dtype=np.float32)
    d /= n

    ang = np.deg2rad(float(half_angle_deg))
    cos_a, sin_a = float(np.cos(ang)), float(np.sin(ang))
    rot_left = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    rot_right = np.array([[cos_a, sin_a], [-sin_a, cos_a]], dtype=np.float32)

    o = np.asarray(origin, dtype=np.float32)
    left_vec = rot_left @ d * float(length)
    right_vec = rot_right @ d * float(length)
    return np.stack([o, o + left_vec, o + right_vec], axis=0)


def triangle_box_intersect(triangle, box) -> bool:
    """Return True when a 2D triangle overlaps an axis-aligned box.

    ``box`` is ``(x1, y1, x2, y2)``. Uses OpenCV's ``intersectConvexConvex``.
    """
    tri = np.asarray(triangle, dtype=np.float32).reshape(-1, 1, 2)
    x1, y1, x2, y2 = map(float, box)
    rect = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
    ).reshape(-1, 1, 2)
    inter_area, _ = cv2.intersectConvexConvex(tri, rect)
    return inter_area > 0.0


def in_drill_window(frame_idx: int, ctx) -> bool:
    """Return True when ``frame_idx`` (1-indexed) falls in the drill window.

    The window is read from ``ctx.drill_start_frame`` and ``ctx.drill_end_frame``
    on :class:`~src.metrics.context.MetricContext`. ``drill_end_frame=None`` means
    "no upper bound" (today's behavior when drill-window detection is disabled).
    """
    start = getattr(ctx, "drill_start_frame", 1) or 1
    end = getattr(ctx, "drill_end_frame", None)
    if frame_idx < start:
        return False
    if end is not None and frame_idx > end:
        return False
    return True


def _valid_sample_count(traj: Sequence[Any]) -> int:
    return sum(1 for pt in traj if pt is not None)


def _first_valid_index(traj: Sequence[Any]) -> Optional[int]:
    for i, pt in enumerate(traj):
        if pt is not None:
            return i
    return None


def select_entry_tracks(
    tracks_by_id: Any,
    *,
    inroom_ids: Optional[Sequence[int]] = None,
    num_tracks: int,
) -> List[Tuple[int, Sequence[Any]]]:
    """Single source of truth for the team-entry track-selection rule.

    Filters in-room IDs, keeps the top ``num_tracks`` trajectories by valid
    sample count (ties broken by first-valid frame, then track id), and
    returns them re-sorted by entry order. Used by every team-entry metric
    and the POD-by-entry assigner so the same N trajectories drive every
    score.

    Returns ``[(track_id, traj), ...]``. Pass ``num_tracks=0`` (or any
    non-positive) to skip the top-N truncation.
    """
    inroom_set = {int(x) for x in (inroom_ids or [])}
    friend = [
        (int(tid), traj)
        for tid, traj in (tracks_by_id or {}).items()
        if int(tid) not in inroom_set
    ]

    def _length_key(item: Tuple[int, Sequence[Any]]):
        tid, traj = item
        first = _first_valid_index(traj)
        return (
            -_valid_sample_count(traj),
            first if first is not None else 10**12,
            tid,
        )

    friend.sort(key=_length_key)
    if int(num_tracks) > 0:
        friend = friend[: int(num_tracks)]

    def _entry_key(item: Tuple[int, Sequence[Any]]):
        tid, traj = item
        first = _first_valid_index(traj)
        return (first if first is not None else 10**12, tid)

    friend.sort(key=_entry_key)
    return friend


def team_size(config) -> int:
    """Resolve the expected number of friendly entrants for entry-team metrics.

    Reads ``config["team_size"]`` when present; otherwise falls back to the
    number of POD points (preserves behavior for older saved configs that
    pre-date the dedicated ``team_size`` key).
    """
    if isinstance(config, dict) and config.get("team_size") is not None:
        try:
            n = int(config["team_size"])
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass
    pods = config.get("POD", []) if isinstance(config, dict) else []
    try:
        return int(len(pods))
    except TypeError:
        return 0


# ---------------------------------------------------------------------------
# Entry-direction primitives shared by entrance_vectors, analysis, and the
# POD-by-entry assignment helper. The contract here is the single source of
# truth for "what counts as the entry direction" across the codebase.
# ---------------------------------------------------------------------------


CORNER_VERTEX_TOL_FRACTION = 0.5  # × min door side length: how close a boundary corner must be to the door polygon to count as a corner door


@dataclass(frozen=True)
class DoorAxes:
    """Per-door geometry derived from the room boundary + door polygon.

    Attributes
    ----------
    polygon : list of (x, y)
        Vertices of the door polygon, in source order.
    centroid : (float, float)
        Centroid of the door polygon. Used as the door reference point.
    p_a, p_b : (float, float)
        Two unit vectors representing the doctrinally-valid entry paths.
        For STRAIGHT doors these are the two opposite tangent directions
        along the host wall (``+T̂`` and ``−T̂``). For CORNER doors these
        are the two host-wall directions pointing away from the shared
        corner vertex (so each path goes deeper into the room along one
        of the two adjacent walls).
    n_in : (float, float)
        Inward normal unit vector (perpendicular to the host wall,
        pointing toward the room interior).
    door_type : "STRAIGHT" or "CORNER"
    """

    polygon: List[Tuple[float, float]]
    centroid: Tuple[float, float]
    p_a: Tuple[float, float]
    p_b: Tuple[float, float]
    n_in: Tuple[float, float]
    door_type: str


def _polygon_centroid(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Mean of polygon vertices. Returns the zero vector for an empty input."""
    arr = np.asarray(points, dtype=float)
    if arr.shape[0] == 0:
        return np.zeros(2, dtype=float)
    return arr.mean(axis=0)


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 0.0:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n <= 1e-9:
        return None
    return v / n


def compute_door_axes(
    boundary_pts: Sequence[Sequence[float]],
    door_polygon_pts: Sequence[Sequence[float]],
) -> Optional[DoorAxes]:
    """Derive the door's axes from the room boundary + the door polygon.

    Returns ``None`` if the geometry is degenerate (fewer than 3 boundary
    points, no door points, or the host wall has zero length).
    """
    boundary = np.asarray(boundary_pts, dtype=float)
    door = np.asarray(door_polygon_pts, dtype=float)
    if boundary.shape[0] < 3 or door.shape[0] < 2:
        return None

    door_centroid = _polygon_centroid(door)
    boundary_centroid = _polygon_centroid(boundary)

    # Distance from door centroid to each boundary edge segment.
    n_b = boundary.shape[0]
    edge_dists: List[Tuple[int, float]] = [
        (i, _segment_distance(door_centroid, boundary[i], boundary[(i + 1) % n_b]))
        for i in range(n_b)
    ]
    edge_dists.sort(key=lambda kv: kv[1])

    host_a_idx = edge_dists[0][0]
    host_b_idx = edge_dists[1][0] if len(edge_dists) > 1 else host_a_idx

    # Door bbox spans, used to scale the corner-vertex tolerance.
    door_min_side = max(1.0, float(min(np.ptp(door[:, 0]), np.ptp(door[:, 1]))))
    corner_tol = CORNER_VERTEX_TOL_FRACTION * door_min_side

    # Two boundary edges share a vertex iff they are adjacent in the polygon.
    shared_vertex_idx: Optional[int] = None
    if (host_a_idx + 1) % n_b == host_b_idx:
        shared_vertex_idx = host_b_idx
    elif (host_b_idx + 1) % n_b == host_a_idx:
        shared_vertex_idx = host_a_idx

    is_corner = False
    if shared_vertex_idx is not None:
        v_corner = boundary[shared_vertex_idx]
        # Distance from the shared corner to the closest door polygon edge.
        n_d = door.shape[0]
        d_corner_to_door = min(
            _segment_distance(v_corner, door[j], door[(j + 1) % n_d])
            for j in range(n_d)
        )
        if d_corner_to_door <= corner_tol:
            is_corner = True

    # Inward normal of the host wall (pointing toward the boundary centroid).
    a = boundary[host_a_idx]
    b = boundary[(host_a_idx + 1) % n_b]
    edge_vec = b - a
    edge_unit = _unit(edge_vec)
    if edge_unit is None:
        return None
    perp = np.array([-edge_unit[1], edge_unit[0]], dtype=float)
    edge_mid = 0.5 * (a + b)
    if float(np.dot(perp, boundary_centroid - edge_mid)) < 0.0:
        perp = -perp
    n_in = perp

    if is_corner and shared_vertex_idx is not None:
        # Host walls = the two edges that meet at the shared corner. Each
        # path direction = the unit vector along that edge pointing AWAY from
        # the corner (deeper into the room along that wall).
        v_corner = boundary[shared_vertex_idx]
        # Identify the two edges meeting at v_corner.
        edge_after = ((shared_vertex_idx) % n_b, (shared_vertex_idx + 1) % n_b)
        edge_before = ((shared_vertex_idx - 1) % n_b, shared_vertex_idx % n_b)
        ea_other = boundary[edge_after[1]]
        eb_other = boundary[edge_before[0]]
        p_a = _unit(ea_other - v_corner)
        p_b = _unit(eb_other - v_corner)
        if p_a is None or p_b is None:
            return None
        door_type = "CORNER"
    else:
        # Straight door: ±T̂ where T̂ is the host wall tangent, oriented so
        # cross(N̂_in, T̂) > 0 (right-hand convention for stable left/right).
        t_hat = edge_unit
        if float(n_in[0] * t_hat[1] - n_in[1] * t_hat[0]) < 0.0:
            t_hat = -t_hat
        p_a = t_hat
        p_b = -t_hat
        door_type = "STRAIGHT"

    return DoorAxes(
        polygon=[(float(p[0]), float(p[1])) for p in door],
        centroid=(float(door_centroid[0]), float(door_centroid[1])),
        p_a=(float(p_a[0]), float(p_a[1])),
        p_b=(float(p_b[0]), float(p_b[1])),
        n_in=(float(n_in[0]), float(n_in[1])),
        door_type=door_type,
    )


def load_door_axes(
    boundary_pts: Sequence[Sequence[float]],
    door_polygons: Sequence[Sequence[Sequence[float]]],
) -> List[DoorAxes]:
    """Compute :class:`DoorAxes` for each door polygon. Skips degenerate ones."""
    out: List[DoorAxes] = []
    for poly in door_polygons or []:
        ax = compute_door_axes(boundary_pts, poly)
        if ax is not None:
            out.append(ax)
    return out


def _point_in_polygon(pt: Tuple[float, float], poly: Sequence[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test. ``poly`` is a list of (x, y) vertices."""
    x, y = float(pt[0]), float(pt[1])
    n = len(poly)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def door_for_entry(
    point: Optional[Sequence[float]],
    doors: Sequence[DoorAxes],
) -> Optional[DoorAxes]:
    """Return the door whose polygon contains ``point``, else the nearest door
    (by centroid distance). Returns ``None`` when ``doors`` is empty or
    ``point`` is missing.
    """
    if point is None or not doors:
        return None
    pt = (float(point[0]), float(point[1]))
    for d in doors:
        if _point_in_polygon(pt, d.polygon):
            return d
    # Fallback: nearest door centroid.
    best: Optional[DoorAxes] = None
    best_d = float("inf")
    for d in doors:
        dx = pt[0] - d.centroid[0]
        dy = pt[1] - d.centroid[1]
        dist = dx * dx + dy * dy
        if dist < best_d:
            best_d = dist
            best = d
    return best


def first_entry_frame(
    traj: Sequence[Optional[Tuple[float, float]]],
    doors: Sequence[DoorAxes],
) -> Optional[int]:
    """First 0-indexed frame whose mapped position lies inside any door
    polygon. Falls back to the first non-None position when no door polygon
    contains any sample (preserves existing behavior on maps without door
    polygons).
    """
    if not traj:
        return None
    if doors:
        for i, p in enumerate(traj):
            if p is None:
                continue
            for d in doors:
                if _point_in_polygon((float(p[0]), float(p[1])), d.polygon):
                    return i
    for i, p in enumerate(traj):
        if p is not None:
            return i
    return None


def fit_entry_velocity(
    traj: Sequence[Optional[Tuple[float, float]]],
    start_idx: int,
    fps: float,
    window_sec: float,
) -> Optional[np.ndarray]:
    """Least-squares 2D velocity (pixels/sec) over ``window_sec`` of valid
    samples starting at ``start_idx``.

    Returns ``None`` when fewer than two valid samples or the time spread
    is zero. Direction is well-defined regardless of whether endpoints are
    noisy — every sample contributes weight ``1/N`` rather than two endpoints
    carrying ``1/2`` each.
    """
    if traj is None or start_idx is None:
        return None
    fps = float(fps or 30.0)
    window_sec = float(window_sec)
    if fps <= 0 or window_sec <= 0:
        return None
    win_frames = max(2, int(round(window_sec * fps)))
    end_idx = min(len(traj), int(start_idx) + win_frames)

    ts: List[float] = []
    xs: List[float] = []
    ys: List[float] = []
    for k in range(int(start_idx), end_idx):
        p = traj[k]
        if p is None:
            continue
        ts.append((k - int(start_idx)) / fps)
        xs.append(float(p[0]))
        ys.append(float(p[1]))

    if len(ts) < 2:
        return None
    t_arr = np.asarray(ts, dtype=float)
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)

    t_mean = float(np.mean(t_arr))
    dt = t_arr - t_mean
    denom = float(np.sum(dt * dt))
    if denom <= 1e-12:
        return None
    v_x = float(np.sum(dt * (x_arr - float(np.mean(x_arr)))) / denom)
    v_y = float(np.sum(dt * (y_arr - float(np.mean(y_arr)))) / denom)
    return np.array([v_x, v_y], dtype=float)


def classify_entry_side(
    movement_vec: Optional[np.ndarray],
    door: Optional[DoorAxes],
) -> Tuple[int, str, float, float]:
    """Classify a movement vector against a door's two reference path axes.

    Always commits to whichever path axis the movement is more aligned
    with — there is no minimum-magnitude or minimum-alignment gate. An
    UNKNOWN return value is reserved for **truly degenerate** inputs:
    missing door geometry, missing movement, or zero-length movement.
    Anything that has direction at all gets a definite side.

    The original gates were dropped because a single UNKNOWN entrant
    produced second-order noise in the alternation pattern of every
    subsequent entrant — silently inverting the score's interpretation
    of "did the team alternate?" Committing every entrant to whichever
    axis is closer keeps that pattern stable.

    Returns
    -------
    (sign, side, proj_a, proj_b)
        ``sign`` is +1 for side A, −1 for side B, 0 for UNKNOWN.
        ``side`` is "A", "B", or "UNKNOWN".
    """
    if door is None or movement_vec is None:
        return 0, "UNKNOWN", 0.0, 0.0
    m = np.asarray(movement_vec, dtype=float)
    norm = float(np.linalg.norm(m))
    if norm <= 1e-9:
        return 0, "UNKNOWN", 0.0, 0.0
    m_hat = m / norm
    p_a = np.asarray(door.p_a, dtype=float)
    p_b = np.asarray(door.p_b, dtype=float)
    proj_a = float(np.dot(m_hat, p_a))
    proj_b = float(np.dot(m_hat, p_b))
    if proj_a >= proj_b:
        return 1, "A", proj_a, proj_b
    return -1, "B", proj_a, proj_b


def load_entry_polygon_points(entry_polys_path: Optional[str]) -> List[List[Tuple[float, float]]]:
    """Read entry polygons file as raw lists of (x, y) tuples.

    Format: comma-separated coordinates per line, ``#`` comments allowed.
    Returns an empty list on missing/unreadable input. This is the
    metric-side counterpart to ``Track.processing.utils.load_entry_polygons``
    (which returns shapely polygons); we keep this lightweight version so
    metrics don't depend on shapely.

    The path is normalized for cross-platform robustness: ``~`` is expanded,
    forward/back slashes are normalized to the host OS, and ``..`` segments
    are collapsed.
    """
    if not entry_polys_path or not isinstance(entry_polys_path, str):
        return []
    p = os.path.normpath(os.path.expanduser(entry_polys_path))
    if not os.path.exists(p):
        return []
    out: List[List[Tuple[float, float]]] = []
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = [t for t in s.replace(",", " ").split() if t]
                try:
                    nums = [float(t) for t in parts]
                except ValueError:
                    continue
                if len(nums) < 6 or len(nums) % 2 != 0:
                    continue
                pts = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
                out.append(pts)
    except OSError:
        return []
    return out
