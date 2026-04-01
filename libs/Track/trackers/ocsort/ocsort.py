# Motion Trackers for Video Analysis -Surya ⎜(Extended from Mikel Broström's work on boxmot(10.0.81))
"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

from ...motion.kalman_filters.xysr_kf import KalmanFilterXYSR
from ...utils.association import associate, linear_assignment
from ...utils.iou import get_asso_func, run_asso_func
from ...utils.pose_association import (
    POSE_ANCHOR_NAMES,
    extract_pose_anchors,
    compute_detection_track_pose_metrics,
)
from ..basetracker import BaseTracker
from ...utils import PerClassDecorator
from ...utils.ops import xyxy2xysr

# ---------------------------------------------------------------------
# Extra ops + geometry
# ---------------------------------------------------------------------
from ...utils.ops import xyxy2xywh, xyxy2tlwh
import shapely.geometry as geo
from shapely.affinity import scale


# ==============================
# Small utilities
# ==============================
def k_previous_obs(observations: Dict[int, np.ndarray], cur_age: int, k: int):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_x_to_bbox(x: np.ndarray, score: Optional[float] = None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


def speed_direction(bbox1: np.ndarray, bbox2: np.ndarray):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm



def normalize(v: np.ndarray):
    norm = np.linalg.norm(v)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm

@dataclass(frozen=True)
class PoseAssociationGateConfig:
    hybrid_threshold: float
    box_evidence_floor: float
    pose_evidence_floor: float


@dataclass(frozen=True)
class PoseAssociationConfig:
    enabled: bool
    weight: float
    min_affinity: float
    detection_mean_conf_threshold: float
    primary: PoseAssociationGateConfig
    rematch: PoseAssociationGateConfig

# ==============================
# Point Kalman Filter (keypoints)
# ==============================
class KalmanFilterPoint:
    """
    Constant velocity KF for a 2D point.
    State: [x, y, vx, vy]
    """
    def __init__(self):
        self.state = np.zeros((4, 1))
        self.P = np.eye(4) * 1000.0
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        self.R = np.eye(2) * 10.0
        self.Q = np.eye(4) * 0.01

    def initiate(self, measurement: np.ndarray):
        measurement = np.asarray(measurement, dtype=float).reshape(2, 1)
        self.state[:2] = measurement
        self.state[2:] = 0.0

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2].flatten()

    def update(self, measurement: np.ndarray):
        z = np.asarray(measurement, dtype=float).reshape(2, 1)
        y = z - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + (K @ y)
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P


# ==============================
# Track object
# ==============================
class KalmanBoxTracker:
    """
    Internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(
        self,
        bbox: np.ndarray,
        cls: int,
        det_ind: int,
        delta_t: int = 3,
        max_obs: int = 50,
        keypoints: Optional[np.ndarray] = None,
        keypoint_confidence_threshold: float = 0.2,
    ):
        self.det_ind = det_ind

        # KF for bbox
        self.kf = KalmanFilterXYSR(dim_x=7, dim_z=4, max_obs=max_obs)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=float,
        )

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = xyxy2xysr(bbox)

        # book-keeping
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.max_obs = max_obs
        self.history = deque([], maxlen=self.max_obs)
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self.conf = float(bbox[-1])
        self.cls = int(cls)

        self.last_observation = np.array([-1, -1, -1, -1, -1], dtype=float)  # placeholder
        self.observations: Dict[int, np.ndarray] = {}
        self.history_observations = deque([], maxlen=self.max_obs)

        self.velocity = None
        self.delta_t = delta_t

        # -------------------- Mapping / keypoints state --------------------
        self.current_map_pos: Optional[np.ndarray] = None
        self.last_map_pos: Optional[np.ndarray] = None
        self.final_id: Optional[int] = None

        self.keypoints = keypoints
        self.keypoint_confidence_threshold = float(keypoint_confidence_threshold)
        self.filtered_keypoints = self.filter_keypoints(keypoints, self.keypoint_confidence_threshold)

        self.pose_anchors = extract_pose_anchors(self.filtered_keypoints, self.keypoint_confidence_threshold)
        self.predicted_pose_anchors = {
            name: dict(info) for name, info in self.pose_anchors.items()
        }
        self.pose_anchor_filters: Dict[str, Optional[KalmanFilterPoint]] = {
            name: None for name in POSE_ANCHOR_NAMES
        }
        self.pose_anchor_missing_counts: Dict[str, int] = {
            name: 0 for name in POSE_ANCHOR_NAMES
        }
        self.pose_anchor_reliability: Dict[str, float] = {
            name: float(self.pose_anchors[name].get("confidence", 0.0)) if self.pose_anchors[name].get("valid", False) else 0.0
            for name in POSE_ANCHOR_NAMES
        }

        self.ankle_based_point: Optional[np.ndarray] = None
        self.keypoint_kalman_filter: Optional[KalmanFilterPoint] = None

        self.prev_top_center: Optional[np.ndarray] = None
        self.missing_keypoints_frames = 0
        self.previous_bbox: Optional[np.ndarray] = None

        # Joint-specific position history for velocity estimation
        self.joint_histories = {
            "lhip": deque([], maxlen=8),
            "rhip": deque([], maxlen=8),
            "lknee": deque([], maxlen=8),
            "rknee": deque([], maxlen=8),
            "bbox": deque([], maxlen=8),
        }

        self.bbox_top_center_history = deque(maxlen=5)

    @staticmethod
    def filter_keypoints(keypoints: Optional[np.ndarray], threshold: float) -> Optional[np.ndarray]:
        """
        Keep x,y as-is; invalidate by zeroing confidence only.
        (Prevents accidental averaging toward (0,0) if someone later forgets confidence checks.)
        """
        if keypoints is None:
            return None
        filtered = np.array(keypoints, copy=True)
        low = filtered[:, 2] < threshold
        filtered[low, 2] = 0.0
        return filtered

    def _init_or_update_pose_anchor_filters(
        self,
        anchors: Dict[str, Dict[str, Any]],
        reappearance_threshold: int = 5,
    ):
        for name in POSE_ANCHOR_NAMES:
            info = anchors.get(name, {})
            valid = bool(info.get("valid", False))
            if valid:
                point = np.asarray(info["point"], dtype=float)
                if self.pose_anchor_missing_counts[name] >= reappearance_threshold or self.pose_anchor_filters[name] is None:
                    self.pose_anchor_filters[name] = KalmanFilterPoint()
                    self.pose_anchor_filters[name].initiate(point)
                else:
                    self.pose_anchor_filters[name].update(point)

                filt = self.pose_anchor_filters[name]
                filtered_point = filt.state[:2].flatten() if filt is not None else point
                self.pose_anchor_missing_counts[name] = 0
                self.pose_anchor_reliability[name] = float(info.get("confidence", 0.0))
                self.pose_anchors[name] = {
                    "point": np.asarray(filtered_point, dtype=float),
                    "confidence": float(info.get("confidence", 0.0)),
                    "support": int(info.get("support", 0)),
                    "valid": True,
                }
            else:
                self.pose_anchor_missing_counts[name] += 1
                decay = 0.85 if self.pose_anchor_missing_counts[name] <= 3 else 0.65
                self.pose_anchor_reliability[name] *= decay
                predicted_info = self.predicted_pose_anchors.get(name, {}) if self.predicted_pose_anchors is not None else {}
                predicted_point = predicted_info.get("point", None)
                if predicted_point is not None:
                    predicted_point = np.asarray(predicted_point, dtype=float)
                self.pose_anchors[name] = {
                    "point": predicted_point,
                    "confidence": float(self.pose_anchor_reliability[name]),
                    "support": int(predicted_info.get("support", 0)),
                    "valid": self.pose_anchor_missing_counts[name] <= reappearance_threshold and predicted_point is not None,
                }

    def _predict_pose_anchors(self, stale_after: int = 8):
        predicted: Dict[str, Dict[str, Any]] = {}
        for name in POSE_ANCHOR_NAMES:
            filt = self.pose_anchor_filters.get(name)
            missing = int(self.pose_anchor_missing_counts.get(name, 0))
            reliability = float(self.pose_anchor_reliability.get(name, 0.0))

            if filt is not None:
                point = filt.predict()
                predicted[name] = {
                    "point": np.asarray(point, dtype=float),
                    "confidence": reliability,
                    "support": int(self.pose_anchors.get(name, {}).get("support", 0)),
                    "valid": missing <= stale_after,
                }
            else:
                prev = self.pose_anchors.get(name, {})
                predicted[name] = {
                    "point": prev.get("point", None),
                    "confidence": reliability,
                    "support": int(prev.get("support", 0)),
                    "valid": bool(prev.get("valid", False)) and missing <= stale_after,
                }

        self.predicted_pose_anchors = predicted

    def update(self, bbox: Optional[np.ndarray], cls: Optional[int], det_ind: Optional[int], keypoints: Optional[np.ndarray] = None):
        """
        Update KF with observed bbox; optionally update keypoints.
        """
        self.det_ind = det_ind

        if bbox is not None:
            self.conf = float(bbox[-1])
            self.cls = int(cls)

            # velocity direction (observation-based)
            if self.last_observation.sum() >= 0:
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                self.velocity = speed_direction(previous_box, bbox)

            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1

            self.kf.update(xyxy2xysr(bbox))

            # keypoints / pose anchors
            if keypoints is not None:
                self.keypoints = keypoints
                self.filtered_keypoints = self.filter_keypoints(keypoints, self.keypoint_confidence_threshold)
            else:
                self.keypoints = None
                self.filtered_keypoints = None

            current_anchors = extract_pose_anchors(self.filtered_keypoints, self.keypoint_confidence_threshold)
            self._init_or_update_pose_anchor_filters(current_anchors)
            self.predicted_pose_anchors = {
                name: dict(info) for name, info in self.pose_anchors.items()
            }
        else:
            # unmatched: still update KF with None per original design
            self.kf.update(bbox)

    def predict(self):
        """
        Advance bbox KF and return predicted bbox.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        self._predict_pose_anchors()
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Current bbox estimate.
        """
        return convert_x_to_bbox(self.kf.x)

    # ------------------------------------------------------------------
    def _continuous_velocity(self, name: str, k: int = 3, current_frame: Optional[int] = None, max_gap: int = 1):
        hist = self.joint_histories.get(name, None)
        if hist is None or len(hist) < k:
            return None

        sub = list(hist)[-k:]
        frames = [f for f, _ in sub]

        if frames != list(range(frames[0], frames[0] + k)):
            return None
        if current_frame is not None and (current_frame - frames[-1] > max_gap):
            return None

        p0 = np.asarray(sub[0][1], dtype=float)
        p1 = np.asarray(sub[-1][1], dtype=float)
        dt = frames[-1] - frames[0]
        if dt == 0:
            return None
        return (p1 - p0) / dt
    # ------------------------------------------------------------------

    def update_map_position(
        self,
        pixel_mapper,
        xywh: np.ndarray,
        frame_idx: int,
        keypoints: Optional[np.ndarray] = None,
        keypoint_indices: Optional[Tuple[int, int]] = None,
        smoothing_factor: float = 0.5,
        bbox_top_center_history_len: int = 5,
        reappearance_threshold: int = 5,
        dynamic_smoothing_min: float = 0.1,
        dynamic_smoothing_max: float = 0.3,
        dynamic_smoothing_thresh: float = 5,
    ):
        """
        Update map position using either bbox-based or keypoint-based (foot/ankle) logic.
        Functionality preserved; made more robust and deterministic.
        """
        # ensure history length matches parameter (backward compatible)
        if self.bbox_top_center_history.maxlen != bbox_top_center_history_len:
            self.bbox_top_center_history = deque(self.bbox_top_center_history, maxlen=bbox_top_center_history_len)

        new_map_pos = None

        if keypoint_indices is None:
            # bbox-based fallback
            self.ankle_based_point = np.array([xywh[0], (xywh[1] + xywh[3] / 2.0)], dtype=float)
            new_map_pos = pixel_mapper.detection_to_map(xywh)

        else:
            # top-center of bbox
            current_top_center = np.array([xywh[0], xywh[1] - (xywh[3] / 2.0)], dtype=float)
            self.joint_histories["bbox"].append((frame_idx, current_top_center))
            self.bbox_top_center_history.append(current_top_center)

            ankle_point = None

            if keypoints is not None:
                # foot-based center (heel/toe indices)
                foot_indices_left = [20, 22, 24]
                foot_indices_right = [21, 23, 25]

                def collect_pts(indices: List[int]):
                    pts = []
                    for i in indices:
                        if i < len(keypoints) and keypoints[i][2] > self.keypoint_confidence_threshold:
                            pts.append(keypoints[i][:2])
                    return pts

                left_pts = collect_pts(foot_indices_left)
                right_pts = collect_pts(foot_indices_right)

                # hip & knee history updates
                hip_knee_map = {"lhip": 11, "rhip": 12, "lknee": 13, "rknee": 14}
                for name, idx in hip_knee_map.items():
                    if idx < len(keypoints) and keypoints[idx][2] > self.keypoint_confidence_threshold:
                        self.joint_histories[name].append((frame_idx, keypoints[idx][:2]))

                if left_pts or right_pts:
                    left_mean = np.mean(left_pts, axis=0) if left_pts else None
                    right_mean = np.mean(right_pts, axis=0) if right_pts else None
                    if left_mean is not None and right_mean is not None:
                        ankle_point = (left_mean + right_mean) / 2.0
                    else:
                        ankle_point = left_mean if left_mean is not None else right_mean
                else:
                    # ankle fallback from indices
                    i1, i2 = keypoint_indices
                    kp1 = keypoints[i1] if i1 < len(keypoints) else None
                    kp2 = keypoints[i2] if i2 < len(keypoints) else None
                    valid = []
                    for kp in (kp1, kp2):
                        if kp is not None and kp[2] > self.keypoint_confidence_threshold:
                            valid.append(kp[:2])
                    if valid:
                        ankle_point = np.mean(valid, axis=0)

            if ankle_point is not None:
                ankle_point = np.asarray(ankle_point, dtype=float)

                # re-init KF if keypoints were missing too long
                if self.missing_keypoints_frames >= reappearance_threshold:
                    if self.keypoint_kalman_filter is None:
                        self.keypoint_kalman_filter = KalmanFilterPoint()
                    self.keypoint_kalman_filter.initiate(ankle_point)

                if self.keypoint_kalman_filter is None:
                    self.keypoint_kalman_filter = KalmanFilterPoint()
                    self.keypoint_kalman_filter.initiate(ankle_point)

                # standard KF cycle for smoothing: predict -> update, then use filtered state
                self.keypoint_kalman_filter.predict()
                self.keypoint_kalman_filter.update(ankle_point)
                filtered_point = self.keypoint_kalman_filter.state[:2].flatten()

                new_map_pos = pixel_mapper.pixel_to_map(filtered_point)
                self.ankle_based_point = filtered_point
                self.missing_keypoints_frames = 0

            else:
                # keypoints missing
                self.missing_keypoints_frames += 1
                if self.keypoint_kalman_filter is not None:
                    # choose velocity from highest-priority continuous joint
                    priority = ["lhip", "rhip", "lknee", "rknee", "bbox"]
                    vel_candidate = None
                    for name in priority:
                        vel_candidate = self._continuous_velocity(name, k=5, current_frame=frame_idx, max_gap=3)
                        if vel_candidate is not None:
                            break

                    if vel_candidate is not None:
                        vx, vy = float(vel_candidate[0]), float(vel_candidate[1])
                        self.keypoint_kalman_filter.state[2:] = np.array([vx, vy], dtype=float).reshape(2, 1)

                    predicted_point = self.keypoint_kalman_filter.predict()
                    if predicted_point is not None:
                        new_map_pos = pixel_mapper.pixel_to_map(predicted_point)
                        self.ankle_based_point = np.asarray(predicted_point, dtype=float)
                # else: remain None

            self.prev_top_center = current_top_center

        self.previous_bbox = xywh

        # dynamic smoothing
        if new_map_pos is not None and self.current_map_pos is not None:
            cur = np.asarray(self.current_map_pos, dtype=float)
            nxt = np.asarray(new_map_pos, dtype=float)
            position_delta = np.linalg.norm(cur - nxt)
            dynamic_smoothing = max(
                dynamic_smoothing_min,
                min(dynamic_smoothing_max, 1.0 - (position_delta / float(dynamic_smoothing_thresh))),
            )
        else:
            dynamic_smoothing = smoothing_factor

        # apply smoothing + store positions as numpy arrays
        if self.current_map_pos is not None:
            self.last_map_pos = np.asarray(self.current_map_pos, dtype=float)
            if new_map_pos is not None:
                nxt = np.asarray(new_map_pos, dtype=float)
                self.current_map_pos = dynamic_smoothing * nxt + (1.0 - dynamic_smoothing) * self.last_map_pos
            else:
                self.current_map_pos = self.last_map_pos
        else:
            self.current_map_pos = None if new_map_pos is None else np.asarray(new_map_pos, dtype=float)

    def calculate_velocity(self):
        vel = np.array([None, None], dtype=object)
        if self.current_map_pos is not None and self.last_map_pos is not None:
            v = np.asarray(self.current_map_pos, dtype=float) - np.asarray(self.last_map_pos, dtype=float)
            v = normalize(v.astype(float))
            vel = v
        return vel


# ==============================
# OC-SORT tracker
# ==============================
class OCSort(BaseTracker):
    """
    OC-SORT Tracker for Video Analysis
    """

    # Special ID reserved for a single pre-existing enemy already inside the room
    ENEMY_FINAL_ID = 99

    def __init__(
        self,
        per_class: bool = False,
        det_thresh: float = 0.2,
        max_age: int = 30,
        min_hits: int = 3,
        asso_threshold: float = 0.3,
        delta_t: int = 3,
        asso_func: str = "iou",
        inertia: float = 0.2,
        use_byte: bool = False,
        pixel_mapper=None,
        limit_entry: bool = False,
        entry_polys=None,
        class_id_to_label: dict = None,
        entry_window_time=None,
        boundary=None,
        boundary_pad_pct: float = 0.0,
        track_enemy: bool = True,
        entry_conf_threshold: Optional[float] = None,
        detection_keypoint_mean_conf_threshold: Optional[float] = None,
        pose_hybrid_enabled: bool = True,
        pose_weight: float = 0.5,
        pose_min_affinity: float = 0.10,
        keypoint_mean_conf_threshold: Optional[float] = None,
        max_obs: int = 50,
    ):
        super().__init__(max_age=max_age, class_id_to_label=class_id_to_label)

        self.per_class = per_class
        self.max_age = max_age
        self.min_hits = min_hits
        self.asso_threshold = asso_threshold
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = get_asso_func(asso_func)
        self.inertia = inertia
        self.use_byte = use_byte
        self.pose_association = self._build_pose_association_config(
            asso_threshold=asso_threshold,
            det_thresh=det_thresh,
            pose_hybrid_enabled=pose_hybrid_enabled,
            pose_weight=pose_weight,
            pose_min_affinity=pose_min_affinity,
            detection_keypoint_mean_conf_threshold=detection_keypoint_mean_conf_threshold,
            legacy_keypoint_mean_conf_threshold=keypoint_mean_conf_threshold,
        )
        self.entry_conf_threshold = float(entry_conf_threshold) if entry_conf_threshold is not None else float(det_thresh)

        self.max_obs = max_obs

        KalmanBoxTracker.count = 0

        self.pixel_mapper = pixel_mapper
        self.limit_entry = limit_entry
        self.entry_polys = entry_polys
        self.next_final_id = 1

        if self.limit_entry:
            if not self.entry_polys:
                raise ValueError("entry_polys cannot be None if limit_entry is True")
            if self.pixel_mapper is None:
                raise ValueError("pixel_mapper cannot be None if limit_entry is True")

        self.entry_window_time = entry_window_time
        self.entry_window_counter = 0
        self.entry_window_active = False

        # boundary padding
        self.boundary = boundary
        if self.boundary is not None:
            self.boundary_padded = (
                scale(self.boundary, xfact=1 + boundary_pad_pct, yfact=1 + boundary_pad_pct, origin="center")
                if boundary_pad_pct
                else self.boundary
            )
        else:
            self.boundary_padded = None

        # enemy state
        self.enemy_active = False
        self.enemy_done = False
        self.enemy_was_in_entry = False
        self.track_enemy = track_enemy

    @staticmethod
    def _build_pose_association_config(
        asso_threshold: float,
        det_thresh: float,
        pose_hybrid_enabled: bool,
        pose_weight: float,
        pose_min_affinity: float,
        detection_keypoint_mean_conf_threshold: Optional[float],
        legacy_keypoint_mean_conf_threshold: Optional[float],
    ) -> PoseAssociationConfig:
        mean_conf_threshold = detection_keypoint_mean_conf_threshold
        if mean_conf_threshold is None:
            mean_conf_threshold = legacy_keypoint_mean_conf_threshold
        if mean_conf_threshold is None:
            mean_conf_threshold = det_thresh

        return PoseAssociationConfig(
            enabled=bool(pose_hybrid_enabled),
            weight=float(pose_weight),
            min_affinity=float(pose_min_affinity),
            detection_mean_conf_threshold=float(mean_conf_threshold),
            primary=PoseAssociationGateConfig(
                hybrid_threshold=float(asso_threshold),
                box_evidence_floor=float(asso_threshold) * 0.5,
                pose_evidence_floor=float(asso_threshold) * 0.5,
            ),
            rematch=PoseAssociationGateConfig(
                hybrid_threshold=float(asso_threshold),
                box_evidence_floor=float(asso_threshold) * 0.4,
                pose_evidence_floor=float(asso_threshold) * 0.4,
            ),
        )

    @staticmethod
    def _split_detections_by_confidence(
        dets: np.ndarray,
        det_thresh: float,
        keypoints: Optional[np.ndarray],
        detection_keypoint_mean_conf_threshold: float,
    ):
        confs = dets[:, 4]

        if keypoints is not None:
            kp_mean_confs = np.asarray(
                [float(np.mean(kp[:, 2])) if kp is not None and len(kp) > 0 else 0.0 for kp in keypoints],
                dtype=float,
            )
        else:
            kp_mean_confs = None

        inds_low = confs > 0.1
        inds_high = confs < det_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        if kp_mean_confs is not None:
            inds_second = np.logical_and(inds_second, kp_mean_confs > detection_keypoint_mean_conf_threshold)

        remain_inds = confs > det_thresh
        if kp_mean_confs is not None:
            remain_inds = np.logical_and(remain_inds, kp_mean_confs > detection_keypoint_mean_conf_threshold)

        dets_first = dets[remain_inds]
        dets_second = dets[inds_second]

        if keypoints is not None:
            keypoints_first = keypoints[remain_inds]
            keypoints_second = keypoints[inds_second]
        else:
            keypoints_first = None
            keypoints_second = None

        return dets_first, dets_second, keypoints_first, keypoints_second

    def _compute_pose_metrics(
        self,
        detections: np.ndarray,
        tracks: List[KalmanBoxTracker],
        keypoints: Optional[np.ndarray],
        keypoint_confidence_threshold: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if keypoints is None or len(tracks) == 0 or len(detections) == 0:
            return None, None

        pose_affinity = np.zeros((len(detections), len(tracks)), dtype=float)
        pose_reliability = np.zeros((len(detections), len(tracks)), dtype=float)
        for det_idx in range(len(detections)):
            sims, reliabilities = compute_detection_track_pose_metrics(
                keypoints[det_idx],
                tracks,
                detections[det_idx, 0:4],
                keypoint_confidence_threshold,
            )
            pose_affinity[det_idx, :] = sims
            pose_reliability[det_idx, :] = reliabilities

        pose_affinity[pose_affinity < self.pose_association.min_affinity] = 0.0
        pose_reliability[pose_affinity <= 0.0] = 0.0
        return pose_affinity, pose_reliability

    @staticmethod
    def _update_track_with_detection(
        track: KalmanBoxTracker,
        detection: np.ndarray,
        keypoints: Optional[np.ndarray],
        keypoint_confidence_threshold: float,
    ):
        track.keypoint_confidence_threshold = float(keypoint_confidence_threshold)
        track.update(detection[:5], detection[5], detection[6], keypoints=keypoints)

    @PerClassDecorator
    def update(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        embs: np.ndarray = None,
        keypoints: Optional[np.ndarray] = None,
        keypoint_confidence_threshold: float = 0.5,
        keypoint_indices: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        assert isinstance(dets, np.ndarray), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert len(dets.shape) == 2, "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert dets.shape[1] == 6, "Unsupported 'dets' 2nd dimension length, valid length is 6"

        self.frame_count += 1

        if self.limit_entry and self.entry_window_active:
            self.entry_window_counter += 1

        h, w = img.shape[0:2]

        # append detection indices
        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])

        dets, dets_second, keypoints_first, keypoints_second = self._split_detections_by_confidence(
            dets,
            self.det_thresh,
            keypoints,
            self.pose_association.detection_mean_conf_threshold,
        )


        # get predicted locations from existing trackers
        trks = np.zeros((len(self.active_tracks), 5), dtype=float)
        to_del = []
        outputs: List[Dict[str, Any]] = []

        for t in range(len(trks)):
            pos = self.active_tracks[t].predict()[0]
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0.0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.active_tracks.pop(t)

        pose_affinity, pose_reliability = self._compute_pose_metrics(
            dets,
            self.active_tracks,
            keypoints_first,
            keypoint_confidence_threshold,
        )

        velocities = np.array(
            [trk.velocity if trk.velocity is not None else np.array((0.0, 0.0)) for trk in self.active_tracks],
            dtype=float,
        )
        last_boxes = np.array([trk.last_observation for trk in self.active_tracks], dtype=float)
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.active_tracks], dtype=float)

        # ---------------- First association ----------------
        matched, unmatched_dets, unmatched_trks = associate(
            dets[:, 0:5],
            trks,
            self.asso_func,
            self.asso_threshold,
            velocities,
            k_observations,
            self.inertia,
            w,
            h,
            pose_affinity=pose_affinity,
            pose_reliability=pose_reliability,
            pose_weight=self.pose_association.weight if self.pose_association.enabled else 0.0,
            hybrid_threshold=self.pose_association.primary.hybrid_threshold,
            box_threshold_floor=self.pose_association.primary.box_evidence_floor,
            pose_threshold_floor=self.pose_association.primary.pose_evidence_floor,
        )

        for m in matched:
            det_idx, trk_idx = m[0], m[1]
            kp = keypoints_first[det_idx] if keypoints_first is not None else None
            self._update_track_with_detection(
                self.active_tracks[trk_idx],
                dets[det_idx],
                kp,
                keypoint_confidence_threshold,
            )

        # ---------------- Second association (BYTE) ----------------
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = np.array(self.asso_func(dets_second, u_trks))
            if iou_left.max() > self.asso_threshold:
                matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.asso_threshold:
                        continue
                    kp = keypoints_second[m[0]] if keypoints_second is not None else None
                    self._update_track_with_detection(
                        self.active_tracks[trk_ind],
                        dets_second[det_ind],
                        kp,
                        keypoint_confidence_threshold,
                    )
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        # ---------------- Rematch unmatched dets/trks ----------------
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]

            rematch_tracks = [self.active_tracks[idx] for idx in unmatched_trks]
            rematch_pose, rematch_pose_reliability = self._compute_pose_metrics(
                left_dets,
                rematch_tracks,
                keypoints_first[unmatched_dets] if keypoints_first is not None else None,
                keypoint_confidence_threshold,
            )

            rematch_matched, _, _ = associate(
                left_dets[:, 0:5],
                left_trks,
                self.asso_func,
                self.asso_threshold,
                np.zeros((len(unmatched_trks), 2), dtype=float),
                left_trks,
                0.0,
                w,
                h,
                pose_affinity=rematch_pose,
                pose_reliability=rematch_pose_reliability,
                pose_weight=self.pose_association.weight if self.pose_association.enabled else 0.0,
                hybrid_threshold=self.pose_association.rematch.hybrid_threshold,
                box_threshold_floor=self.pose_association.rematch.box_evidence_floor,
                pose_threshold_floor=self.pose_association.rematch.pose_evidence_floor,
            )

            if rematch_matched.size > 0:
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematch_matched:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    kp = keypoints_first[det_ind] if keypoints_first is not None else None
                    self._update_track_with_detection(
                        self.active_tracks[trk_ind],
                        dets[det_ind],
                        kp,
                        keypoint_confidence_threshold,
                    )
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        # mark unmatched tracks
        for m in unmatched_trks:
            self.active_tracks[m].update(None, None, None)

        # ---------------- Create new tracks for unmatched detections ----------------
        if unmatched_dets.shape[0] > 0:
            for i in unmatched_dets:
                kp_i = keypoints_first[i] if keypoints_first is not None else None
                det_conf = float(dets[i, 4])

                # Convert detection centre to map point (if mapper available)
                map_point = None
                if self.pixel_mapper is not None:
                    map_xy = self.pixel_mapper.detection_to_map(xyxy2xywh(dets[i, 0:4]))
                    map_xy = np.asarray(map_xy, dtype=float).reshape(-1)
                    if map_xy.size >= 2 and np.isfinite(map_xy[:2]).all():
                        map_point = geo.Point(float(map_xy[0]), float(map_xy[1]))

                # If the point is inside an entry polygon (door/entry zone), do NOT classify it as the pre-existing enemy.
                # This prevents door entrants from being incorrectly assigned ENEMY_FINAL_ID.
                # Also, once we accept an entrant through the entry logic below, we will mark
                # `enemy_done = True` so the remaining tracks are treated as participants.
                in_entry = False
                if (map_point is not None) and self.entry_polys:
                    in_entry = any(poly.contains(map_point) for poly in self.entry_polys)

                # enemy handling
                if (
                    self.track_enemy
                    and (not self.enemy_active)
                    and (not self.enemy_done)
                    and (map_point is not None)
                    and (self.boundary_padded is not None)
                    and self.boundary_padded.contains(map_point)
                    and (not in_entry)
                ):
                    trk = KalmanBoxTracker(
                        dets[i, :5],
                        dets[i, 5],
                        dets[i, 6],
                        delta_t=self.delta_t,
                        max_obs=self.max_obs,
                        keypoints=kp_i,
                        keypoint_confidence_threshold=keypoint_confidence_threshold,
                    )
                    trk.final_id = self.ENEMY_FINAL_ID
                    self.active_tracks.append(trk)
                    self.enemy_active = True
                    continue

                create = False
                if self.limit_entry:
                    if self.entry_window_active:
                        if self.entry_window_time is not None and self.entry_window_counter >= self.entry_window_time:
                            continue

                    if map_point is None and self.pixel_mapper is not None:
                        map_xy = self.pixel_mapper.detection_to_map(xyxy2xywh(dets[i, 0:4]))
                        map_xy = np.asarray(map_xy, dtype=float).reshape(-1)
                        if map_xy.size >= 2 and np.isfinite(map_xy[:2]).all():
                            map_point = geo.Point(float(map_xy[0]), float(map_xy[1]))

                    for poly in self.entry_polys:
                        if poly.contains(map_point) and det_conf >= self.entry_conf_threshold:
                            create = True
                            break

                    # Require both spatial entry confirmation and sufficient detection
                    # confidence before allowing a new track to be born from an entry region.
                    # This helps suppress one-off false detections near the doorway from
                    # later maturing into valid room identities after they drift elsewhere.
                    if create and not self.entry_window_active:
                        self.entry_window_active = True
                        self.entry_window_counter = 0
                else:
                    create = True

                if create:
                    trk = KalmanBoxTracker(
                        dets[i, :5],
                        dets[i, 5],
                        dets[i, 6],
                        delta_t=self.delta_t,
                        max_obs=self.max_obs,
                        keypoints=kp_i,
                        keypoint_confidence_threshold=keypoint_confidence_threshold,
                    )
                    self.active_tracks.append(trk)

                    # Once a person is legitimately born from an entry region, treat the
                    # scene as having no pre-existing in-room enemy. From this point on,
                    # all entrants are participants and normal participant IDs should start
                    # from 1 instead of allowing a later non-entry detection to be labeled
                    # as ENEMY_FINAL_ID.
                    if self.limit_entry and in_entry:
                        self.enemy_active = False
                        self.enemy_done = True
                        self.enemy_was_in_entry = False

        # ---------------- Build outputs + remove dead tracks ----------------
        i = len(self.active_tracks)
        for trk in reversed(self.active_tracks):
            if trk.last_observation.sum() < 0:
                tlwh = xyxy2tlwh(trk.get_state()[0])
            else:
                tlwh = xyxy2tlwh(trk.last_observation)

            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits):
                if trk.final_id is None:
                    trk.final_id = self.next_final_id
                    self.next_final_id += 1

                track_dict = {
                    "top_left_x": tlwh[0],
                    "top_left_y": tlwh[1],
                    "width": tlwh[2],
                    "height": tlwh[3],
                    "track_id": trk.final_id,
                    "track_id_raw": trk.id + 1,
                    "confidence": trk.conf,
                    "class": trk.cls,
                    "detection_index": trk.det_ind,
                    "keypoints": trk.filtered_keypoints,
                    "ankle_based_point": trk.ankle_based_point,
                }

                if self.pixel_mapper is not None:
                    trk.update_map_position(
                        self.pixel_mapper,
                        xyxy2xywh(trk.get_state()[0]),
                        self.frame_count,
                        keypoints=trk.filtered_keypoints,
                        keypoint_indices=keypoint_indices,
                        smoothing_factor=0.7,
                    )
                    map_vel = trk.calculate_velocity()
                    track_dict["current_map_pos"] = trk.current_map_pos
                    track_dict["map_velocity"] = map_vel
                else:
                    track_dict["current_map_pos"] = None
                    track_dict["map_velocity"] = [None, None]

                outputs.append(track_dict)

            i -= 1
            if trk.time_since_update > self.max_age:
                self.active_tracks.pop(i)

        # ---------------- Enemy exit logic ----------------
        if self.track_enemy and self.enemy_active:
            enemy_trk = next((t for t in self.active_tracks if getattr(t, "final_id", None) == self.ENEMY_FINAL_ID), None)

            if enemy_trk is None:
                self.enemy_active = False
                self.enemy_done = True
                self.enemy_was_in_entry = False
            else:
                # Ensure enemy has a map position even before it becomes output-eligible
                if self.pixel_mapper is not None and enemy_trk.current_map_pos is None:
                    # Use bbox-based mapping as a fallback for enemy bookkeeping
                    xywh = xyxy2xywh(enemy_trk.get_state()[0])
                    fallback_map = self.pixel_mapper.detection_to_map(xywh)
                    fallback_map = np.asarray(fallback_map, dtype=float).reshape(-1)
                    if fallback_map.size >= 2 and np.isfinite(fallback_map[:2]).all():
                        enemy_trk.current_map_pos = fallback_map[:2]

                if enemy_trk.current_map_pos is not None and self.boundary is not None:
                    cur_xy = np.asarray(enemy_trk.current_map_pos, dtype=float).reshape(-1)
                    if cur_xy.size >= 2 and np.isfinite(cur_xy[:2]).all():
                        pt = geo.Point(float(cur_xy[0]), float(cur_xy[1]))
                    else:
                        pt = None

                    if pt is not None:
                        # If the enemy touches any entry polygon, remember it
                        if not self.enemy_was_in_entry and self.entry_polys:
                            self.enemy_was_in_entry = any(poly.contains(pt) for poly in self.entry_polys)

                        # Only declare “enemy left” when it was in entry once *and* is now outside boundary
                        if self.enemy_was_in_entry and (not self.boundary.contains(pt)):
                            self.active_tracks.remove(enemy_trk)
                            self.enemy_active = False
                            self.enemy_done = True
                            self.enemy_was_in_entry = False

        # NOTE: preserved original return type (np.array of dicts -> dtype=object)
        return np.array(outputs) if len(outputs) > 0 else np.array([])