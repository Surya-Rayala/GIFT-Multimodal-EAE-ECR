# Motion Trackers for Video Analysis - Surya

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------
# Pose anchor definitions
# ---------------------------------------------------------------------
# These anchors are derived from the 26-keypoint body layout and are used
# as a compact, robust pose representation for detection-to-track matching.
POSE_ANCHOR_NAMES = (
    "head_core",
    "neck",
    "shoulder",
    "hip",
    "knee",
    "foot",
)

POSE_ANCHOR_COMPONENTS = {
    "head_core": (17, 0, 1, 2, 3, 4),
    "neck": (18,),
    "shoulder": (5, 6),
    "hip": (11, 12, 19),
    "knee": (13, 14),
    "foot": (15, 16, 20, 21, 22, 23, 24, 25),
}

# Relative importance of each anchor during pose similarity scoring.
POSE_ANCHOR_PRIORITY = {
    "head_core": 1.45,
    "neck": 1.25,
    "shoulder": 1.15,
    "hip": 1.05,
    "knee": 0.95,
    "foot": 0.85,
}

# Backward-compatible aliases for any external code that still refers to
# older anchor names.
POSE_ANCHOR_ALIASES = {
    "head": "head_core",
}


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def _build_empty_anchor() -> Dict[str, Any]:
    """Return the standard empty anchor structure."""
    return {
        "point": None,
        "confidence": 0.0,
        "support": 0,
        "valid": False,
    }


def _mean_point(points: List[np.ndarray]) -> Optional[np.ndarray]:
    """Safely compute the mean of 2D points."""
    if not points:
        return None
    return np.mean(np.asarray(points, dtype=float), axis=0)


def _resolve_anchor_name(anchor_name: str) -> str:
    """Map backward-compatible aliases to canonical anchor names."""
    return POSE_ANCHOR_ALIASES.get(anchor_name, anchor_name)


def _extract_anchor_from_indices(
    keypoints: np.ndarray,
    indices: Tuple[int, ...],
    confidence_threshold: float,
) -> Dict[str, Any]:
    """
    Extract one pose anchor from a set of keypoint indices.

    A keypoint contributes only if its confidence is above the threshold.
    The anchor point is the mean of contributing keypoint coordinates.
    """
    valid_points: List[np.ndarray] = []
    valid_confidences: List[float] = []

    for keypoint_index in indices:
        if keypoint_index < len(keypoints) and keypoints[keypoint_index][2] > confidence_threshold:
            valid_points.append(keypoints[keypoint_index][:2])
            valid_confidences.append(float(keypoints[keypoint_index][2]))

    anchor_point = _mean_point(valid_points)
    if anchor_point is None:
        return _build_empty_anchor()

    return {
        "point": np.asarray(anchor_point, dtype=float),
        "confidence": float(np.mean(valid_confidences)) if valid_confidences else 0.0,
        "support": len(valid_points),
        "valid": True,
    }


def _get_anchor_info(
    anchor_dict: Optional[Dict[str, Dict[str, Any]]],
    anchor_name: str,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve anchor info by name, with alias fallback.
    """
    if not anchor_dict:
        return None

    direct_match = anchor_dict.get(anchor_name)
    if direct_match is not None:
        return direct_match

    canonical_name = _resolve_anchor_name(anchor_name)
    return anchor_dict.get(canonical_name)


def _iter_valid_anchor_pairs(
    detection_anchors: Optional[Dict[str, Dict[str, Any]]],
    track_anchors: Optional[Dict[str, Dict[str, Any]]],
):
    """
    Yield valid corresponding anchor pairs between detection and track.

    Each yielded item contains:
      - canonical anchor name
      - detection anchor info
      - track anchor info
      - detection anchor point
      - track anchor point
    """
    for raw_anchor_name in POSE_ANCHOR_NAMES:
        anchor_name = _resolve_anchor_name(raw_anchor_name)

        detection_anchor = _get_anchor_info(detection_anchors, anchor_name)
        track_anchor = _get_anchor_info(track_anchors, anchor_name)

        if not detection_anchor or not track_anchor:
            continue
        if not detection_anchor.get("valid", False) or not track_anchor.get("valid", False):
            continue

        detection_point = np.asarray(detection_anchor["point"], dtype=float)
        track_point = np.asarray(track_anchor["point"], dtype=float)

        if detection_point.shape[0] != 2 or track_point.shape[0] != 2:
            continue
        if not (np.isfinite(detection_point).all() and np.isfinite(track_point).all()):
            continue

        yield anchor_name, detection_anchor, track_anchor, detection_point, track_point


# ---------------------------------------------------------------------
# Anchor extraction
# ---------------------------------------------------------------------
def extract_pose_anchors(
    keypoints: Optional[np.ndarray],
    confidence_threshold: float,
) -> Dict[str, Dict[str, Any]]:
    """
    Build a small set of robust anchors from the 26-keypoint body layout.

    Returns a dictionary per anchor with:
      - point: np.ndarray of shape (2,) or None
      - confidence: float in [0, 1]
      - support: number of joints contributing to the anchor
      - valid: bool
    """
    anchors: Dict[str, Dict[str, Any]] = {
        anchor_name: _build_empty_anchor()
        for anchor_name in POSE_ANCHOR_NAMES
    }

    if keypoints is None:
        return anchors

    keypoints_array = np.asarray(keypoints, dtype=float)
    if keypoints_array.ndim != 2 or keypoints_array.shape[1] < 3:
        return anchors

    for anchor_name, component_indices in POSE_ANCHOR_COMPONENTS.items():
        anchors[anchor_name] = _extract_anchor_from_indices(
            keypoints_array,
            component_indices,
            confidence_threshold,
        )

    # Shoulder fallback from neck when shoulders are not available.
    if (not anchors["shoulder"]["valid"]) and anchors["neck"]["valid"]:
        anchors["shoulder"] = {
            "point": np.asarray(anchors["neck"]["point"], dtype=float),
            "confidence": float(anchors["neck"]["confidence"]),
            "support": 1,
            "valid": True,
        }

    # Neck fallback from shoulder center when neck is not available.
    if (not anchors["neck"]["valid"]) and anchors["shoulder"]["valid"]:
        anchors["neck"] = {
            "point": np.asarray(anchors["shoulder"]["point"], dtype=float),
            "confidence": float(anchors["shoulder"]["confidence"]),
            "support": 1,
            "valid": True,
        }

    # Preserve backward compatibility for code that still expects "head".
    anchors["head"] = dict(anchors["head_core"])

    return anchors


# ---------------------------------------------------------------------
# Pose scale
# ---------------------------------------------------------------------
def compute_pose_scale_from_bbox(bbox_xyxy: Optional[np.ndarray]) -> float:
    """
    Compute a normalization scale from a detection bounding box.

    This is used to normalize anchor distances before converting them to
    similarity scores.
    """
    if bbox_xyxy is None:
        return 1.0

    box = np.asarray(bbox_xyxy, dtype=float).reshape(-1)
    if box.size < 4 or not np.isfinite(box[:4]).all():
        return 1.0

    width = max(float(box[2] - box[0]), 1.0)
    height = max(float(box[3] - box[1]), 1.0)
    return max(height, width * 0.75, 1.0)


# ---------------------------------------------------------------------
# Anchor similarity / reliability scoring
# ---------------------------------------------------------------------
def pose_affinity_and_reliability_from_anchors(
    det_anchors: Optional[Dict[str, Dict[str, Any]]],
    trk_anchors: Optional[Dict[str, Dict[str, Any]]],
    scale_value: float,
    sigma: float = 0.35,
) -> Tuple[float, float]:
    """
    Compute pose similarity and reliability from two anchor dictionaries.

    Returns:
      - similarity in [0, 1]
      - reliability in [0, 1]

    Similarity is based on normalized anchor distance.
    Reliability reflects how much trustworthy shared pose evidence exists.
    """
    if not det_anchors or not trk_anchors:
        return 0.0, 0.0

    scale_value = max(float(scale_value), 1.0)

    weighted_similarity_terms: List[float] = []
    anchor_weights: List[float] = []
    reliability_terms: List[float] = []

    max_possible_weight = float(sum(POSE_ANCHOR_PRIORITY.values()))

    for (
        anchor_name,
        det_anchor_info,
        trk_anchor_info,
        det_anchor_point,
        trk_anchor_point,
    ) in _iter_valid_anchor_pairs(det_anchors, trk_anchors):
        normalized_distance = np.linalg.norm(det_anchor_point - trk_anchor_point) / scale_value
        similarity = float(
            np.exp(-(normalized_distance ** 2) / max(2.0 * (sigma ** 2), 1e-6))
        )

        detection_confidence = float(det_anchor_info.get("confidence", 0.0))
        track_confidence = float(trk_anchor_info.get("confidence", 0.0))
        shared_support = min(
            int(det_anchor_info.get("support", 0)),
            int(trk_anchor_info.get("support", 0)),
        )

        support_factor = min(1.0, 0.35 + 0.15 * shared_support)
        confidence_factor = max(0.05, (detection_confidence + track_confidence) / 2.0)
        anchor_weight = POSE_ANCHOR_PRIORITY[anchor_name] * confidence_factor * support_factor

        weighted_similarity_terms.append(similarity * anchor_weight)
        anchor_weights.append(anchor_weight)
        reliability_terms.append(anchor_weight / max_possible_weight)

    if not anchor_weights or float(np.sum(anchor_weights)) <= 0.0:
        return 0.0, 0.0

    similarity_value = float(np.sum(weighted_similarity_terms) / np.sum(anchor_weights))
    reliability_value = float(min(1.0, np.sum(reliability_terms) * 2.0))
    return similarity_value, reliability_value


def pose_affinity_from_anchors(
    det_anchors: Optional[Dict[str, Dict[str, Any]]],
    trk_anchors: Optional[Dict[str, Dict[str, Any]]],
    scale_value: float,
    sigma: float = 0.35,
) -> float:
    """
    Backward-compatible wrapper returning only similarity.
    """
    similarity_value, _ = pose_affinity_and_reliability_from_anchors(
        det_anchors,
        trk_anchors,
        scale_value,
        sigma=sigma,
    )
    return similarity_value


# ---------------------------------------------------------------------
# Detection-to-track pose scoring
# ---------------------------------------------------------------------
def compute_detection_track_pose_metrics(
    detection_keypoints: Optional[np.ndarray],
    tracks: Iterable[Any],
    det_bbox_xyxy: np.ndarray,
    confidence_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pose similarity and reliability for one detection against all tracks.

    Returns:
      - similarity vector of shape (num_tracks,) in [0, 1]
      - reliability vector of shape (num_tracks,) in [0, 1]
    """
    track_list = list(tracks)
    if detection_keypoints is None or len(track_list) == 0:
        zeros = np.zeros((len(track_list),), dtype=float)
        return zeros, zeros.copy()

    detection_anchors = extract_pose_anchors(detection_keypoints, confidence_threshold)
    detection_scale = compute_pose_scale_from_bbox(det_bbox_xyxy)

    similarities = np.zeros((len(track_list),), dtype=float)
    reliabilities = np.zeros((len(track_list),), dtype=float)

    for track_index, track in enumerate(track_list):
        track_anchors = getattr(track, "predicted_pose_anchors", None) or getattr(track, "pose_anchors", None)
        similarities[track_index], reliabilities[track_index] = pose_affinity_and_reliability_from_anchors(
            detection_anchors,
            track_anchors,
            detection_scale,
        )

    return similarities, reliabilities


def compute_detection_track_pose_affinity(
    detection_keypoints: Optional[np.ndarray],
    tracks: Iterable[Any],
    det_bbox_xyxy: np.ndarray,
    confidence_threshold: float,
) -> np.ndarray:
    """
    Backward-compatible wrapper returning only similarity.
    """
    similarities, _ = compute_detection_track_pose_metrics(
        detection_keypoints,
        tracks,
        det_bbox_xyxy,
        confidence_threshold,
    )
    return similarities