# Motion Trackers for Video Analysis -Surya

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List, Iterable, Tuple

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
POSE_ANCHOR_PRIORITY = {
    "head_core": 1.45,
    "neck": 1.25,
    "shoulder": 1.15,
    "hip": 1.05,
    "knee": 0.90,
    "foot": 0.95,
}


# Backward-compatible aliases for any external code that still refers to the old names.
POSE_ANCHOR_ALIASES = {
    "head": "head_core",
}


def _empty_anchor() -> Dict[str, Any]:
    return {"point": None, "confidence": 0.0, "support": 0, "valid": False}


def _safe_point_mean(points: List[np.ndarray]) -> Optional[np.ndarray]:
    if not points:
        return None
    return np.mean(np.asarray(points, dtype=float), axis=0)


def _normalize_anchor_name(name: str) -> str:
    return POSE_ANCHOR_ALIASES.get(name, name)


def _extract_named_anchor(
    kp: np.ndarray,
    indices: Tuple[int, ...],
    confidence_threshold: float,
) -> Dict[str, Any]:
    pts: List[np.ndarray] = []
    confs: List[float] = []
    for idx in indices:
        if idx < len(kp) and kp[idx][2] > confidence_threshold:
            pts.append(kp[idx][:2])
            confs.append(float(kp[idx][2]))

    mean_pt = _safe_point_mean(pts)
    if mean_pt is None:
        return _empty_anchor()

    return {
        "point": np.asarray(mean_pt, dtype=float),
        "confidence": float(np.mean(confs)) if confs else 0.0,
        "support": len(pts),
        "valid": True,
    }


def extract_pose_anchors(
    keypoints: Optional[np.ndarray],
    confidence_threshold: float,
) -> Dict[str, Dict[str, Any]]:
    """
    Build a small set of robust anchors from the 26-keypoint body layout.
    Returns a dict per anchor with:
      - point: np.ndarray shape (2,) or None
      - confidence: float in [0, 1]
      - support: number of joints used
      - valid: bool
    """
    anchors: Dict[str, Dict[str, Any]] = {
        name: _empty_anchor()
        for name in POSE_ANCHOR_NAMES
    }

    if keypoints is None:
        return anchors

    kp = np.asarray(keypoints, dtype=float)
    if kp.ndim != 2 or kp.shape[1] < 3:
        return anchors

    for name, indices in POSE_ANCHOR_COMPONENTS.items():
        anchors[name] = _extract_named_anchor(kp, indices, confidence_threshold)

    # Shoulder fallback from neck when shoulders are absent but neck is visible.
    if (not anchors["shoulder"]["valid"]) and anchors["neck"]["valid"]:
        anchors["shoulder"] = {
            "point": np.asarray(anchors["neck"]["point"], dtype=float),
            "confidence": float(anchors["neck"]["confidence"]),
            "support": 1,
            "valid": True,
        }

    # Neck fallback from shoulder center when neck is absent but shoulders are visible.
    if (not anchors["neck"]["valid"]) and anchors["shoulder"]["valid"]:
        anchors["neck"] = {
            "point": np.asarray(anchors["shoulder"]["point"], dtype=float),
            "confidence": float(anchors["shoulder"]["confidence"]),
            "support": 1,
            "valid": True,
        }

    # Keep a backward-compatible alias so code using "head" still works.
    anchors["head"] = dict(anchors["head_core"])

    return anchors


def compute_pose_scale_from_bbox(bbox_xyxy: Optional[np.ndarray]) -> float:
    if bbox_xyxy is None:
        return 1.0
    box = np.asarray(bbox_xyxy, dtype=float).reshape(-1)
    if box.size < 4 or not np.isfinite(box[:4]).all():
        return 1.0
    width = max(float(box[2] - box[0]), 1.0)
    height = max(float(box[3] - box[1]), 1.0)
    return max(height, width * 0.75, 1.0)


def _get_anchor(info_dict: Optional[Dict[str, Dict[str, Any]]], name: str) -> Optional[Dict[str, Any]]:
    if not info_dict:
        return None
    direct = info_dict.get(name)
    if direct is not None:
        return direct
    alias_name = _normalize_anchor_name(name)
    return info_dict.get(alias_name)


def _iter_valid_anchor_pairs(
    det_anchors: Optional[Dict[str, Dict[str, Any]]],
    trk_anchors: Optional[Dict[str, Dict[str, Any]]],
):
    for raw_name in POSE_ANCHOR_NAMES:
        name = _normalize_anchor_name(raw_name)
        det_info = _get_anchor(det_anchors, name)
        trk_info = _get_anchor(trk_anchors, name)
        if not det_info or not trk_info:
            continue
        if (not det_info.get("valid", False)) or (not trk_info.get("valid", False)):
            continue

        det_pt = np.asarray(det_info["point"], dtype=float)
        trk_pt = np.asarray(trk_info["point"], dtype=float)
        if det_pt.shape[0] != 2 or trk_pt.shape[0] != 2:
            continue
        if not (np.isfinite(det_pt).all() and np.isfinite(trk_pt).all()):
            continue

        yield name, det_info, trk_info, det_pt, trk_pt


def pose_affinity_and_reliability_from_anchors(
    det_anchors: Optional[Dict[str, Dict[str, Any]]],
    trk_anchors: Optional[Dict[str, Dict[str, Any]]],
    scale_value: float,
    sigma: float = 0.35,
) -> Tuple[float, float]:
    """
    Returns:
      - similarity in [0, 1]
      - reliability in [0, 1]

    Similarity is based on normalized anchor distance.
    Reliability reflects how much trustworthy shared pose evidence exists.
    """
    if not det_anchors or not trk_anchors:
        return 0.0, 0.0

    scale_value = max(float(scale_value), 1.0)
    weighted_scores: List[float] = []
    weighted_weights: List[float] = []
    reliability_terms: List[float] = []

    max_possible_weight = float(sum(POSE_ANCHOR_PRIORITY.values()))

    for name, det_info, trk_info, det_pt, trk_pt in _iter_valid_anchor_pairs(det_anchors, trk_anchors):
        norm_dist = np.linalg.norm(det_pt - trk_pt) / scale_value
        similarity = float(np.exp(-(norm_dist ** 2) / max(2.0 * (sigma ** 2), 1e-6)))

        det_conf = float(det_info.get("confidence", 0.0))
        trk_conf = float(trk_info.get("confidence", 0.0))
        support = min(int(det_info.get("support", 0)), int(trk_info.get("support", 0)))
        support_factor = min(1.0, 0.35 + 0.15 * support)
        confidence_factor = max(0.05, (det_conf + trk_conf) / 2.0)
        weight = POSE_ANCHOR_PRIORITY[name] * confidence_factor * support_factor

        weighted_scores.append(similarity * weight)
        weighted_weights.append(weight)
        reliability_terms.append(weight / max_possible_weight)

    if not weighted_weights or float(np.sum(weighted_weights)) <= 0.0:
        return 0.0, 0.0

    similarity_value = float(np.sum(weighted_scores) / np.sum(weighted_weights))
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


def compute_detection_track_pose_metrics(
    detection_keypoints: Optional[np.ndarray],
    tracks: Iterable[Any],
    det_bbox_xyxy: np.ndarray,
    confidence_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pose similarity and reliability for one detection against all active tracks.
    Returns:
      - similarity vector of shape (num_tracks,) in [0, 1]
      - reliability vector of shape (num_tracks,) in [0, 1]
    """
    tracks = list(tracks)
    if detection_keypoints is None or len(tracks) == 0:
        zeros = np.zeros((len(tracks),), dtype=float)
        return zeros, zeros.copy()

    det_anchors = extract_pose_anchors(detection_keypoints, confidence_threshold)
    det_scale = compute_pose_scale_from_bbox(det_bbox_xyxy)

    sims = np.zeros((len(tracks),), dtype=float)
    reliabilities = np.zeros((len(tracks),), dtype=float)
    for idx, trk in enumerate(tracks):
        trk_anchors = getattr(trk, "predicted_pose_anchors", None) or getattr(trk, "pose_anchors", None)
        sims[idx], reliabilities[idx] = pose_affinity_and_reliability_from_anchors(
            det_anchors,
            trk_anchors,
            det_scale,
        )
    return sims, reliabilities


def compute_detection_track_pose_affinity(
    detection_keypoints: Optional[np.ndarray],
    tracks: Iterable[Any],
    det_bbox_xyxy: np.ndarray,
    confidence_threshold: float,
) -> np.ndarray:
    """
    Backward-compatible wrapper returning only similarity.
    """
    sims, _ = compute_detection_track_pose_metrics(
        detection_keypoints,
        tracks,
        det_bbox_xyxy,
        confidence_threshold,
    )
    return sims