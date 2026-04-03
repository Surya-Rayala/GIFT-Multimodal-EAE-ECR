# Motion Trackers for Video Analysis - Surya
# Extended from Mikel Broström's work on boxmot (10.0.81)

from __future__ import annotations

import numpy as np

from .iou import iou_batch, centroid_batch, run_asso_func


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def _resolve_pose_terms(
    spatial_similarity_matrix: np.ndarray,
    pose_affinity: np.ndarray | None,
    pose_reliability: np.ndarray | None,
    pose_weight: float,
):
    """
    Resolve pose-based matching terms.

    Returns:
      - clipped pose affinity matrix or None
      - effective pose contribution matrix or scalar 0.0

    Behavior is preserved from the original implementation:
      effective_pose_term = pose_affinity * pose_weight * pose_reliability
    with pose_reliability defaulting to 1.0 when not provided.
    """
    if pose_affinity is None:
        return None, 0.0

    pose_affinity = np.asarray(pose_affinity, dtype=float)
    if pose_affinity.shape != spatial_similarity_matrix.shape:
        raise ValueError(
            f"pose_affinity shape mismatch: expected {spatial_similarity_matrix.shape}, got {pose_affinity.shape}"
        )
    pose_affinity = np.clip(pose_affinity, a_min=0.0, a_max=1.0)

    weighted_pose_affinity = pose_affinity * float(pose_weight)

    if pose_reliability is None:
        pose_reliability_term = 1.0
    else:
        pose_reliability = np.asarray(pose_reliability, dtype=float)
        if pose_reliability.shape != spatial_similarity_matrix.shape:
            raise ValueError(
                f"pose_reliability shape mismatch: expected {spatial_similarity_matrix.shape}, got {pose_reliability.shape}"
            )
        pose_reliability_term = np.clip(pose_reliability, a_min=0.0, a_max=1.0)

    effective_pose_term = weighted_pose_affinity * pose_reliability_term
    return pose_affinity, effective_pose_term


def _resolve_hybrid_gate_thresholds(
    iou_threshold,
    hybrid_threshold,
    box_threshold_floor,
    pose_threshold_floor,
):
    """
    Resolve hybrid gate thresholds while preserving existing defaults.
    """
    resolved_hybrid_threshold = (
        float(hybrid_threshold) if hybrid_threshold is not None else float(iou_threshold)
    )
    resolved_box_floor = (
        float(box_threshold_floor) if box_threshold_floor is not None else float(iou_threshold) * 0.5
    )
    resolved_pose_floor = (
        float(pose_threshold_floor) if pose_threshold_floor is not None else 0.30
    )
    return resolved_hybrid_threshold, resolved_box_floor, resolved_pose_floor


def _collect_unmatched_indices(matched_indices, num_detections, num_trackers):
    """
    Build unmatched detection and tracker index lists from matched pairs.
    """
    unmatched_detections = []
    for detection_index in range(num_detections):
        if detection_index not in matched_indices[:, 0]:
            unmatched_detections.append(detection_index)

    unmatched_trackers = []
    for tracker_index in range(num_trackers):
        if tracker_index not in matched_indices[:, 1]:
            unmatched_trackers.append(tracker_index)

    return unmatched_detections, unmatched_trackers


def speed_direction_batch(detections, tracks):
    """
    Compute normalized direction vectors from track centers to detection centers.

    Returns:
      dy, dx with shape [num_tracks, num_detections]
    """
    tracks = tracks[..., np.newaxis]

    detection_center_x = (detections[:, 0] + detections[:, 2]) / 2.0
    detection_center_y = (detections[:, 1] + detections[:, 3]) / 2.0
    track_center_x = (tracks[:, 0] + tracks[:, 2]) / 2.0
    track_center_y = (tracks[:, 1] + tracks[:, 3]) / 2.0

    delta_x = detection_center_x - track_center_x
    delta_y = detection_center_y - track_center_y

    norm = np.sqrt(delta_x ** 2 + delta_y ** 2) + 1e-6
    delta_x = delta_x / norm
    delta_y = delta_y / norm

    return delta_y, delta_x


def linear_assignment(cost_matrix):
    """
    Solve linear assignment using lap if available, otherwise scipy.
    """
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array([list(zip(x, y))])


# ---------------------------------------------------------------------
# Basic IoU-only association
# ---------------------------------------------------------------------
def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assign detections to trackers using IoU only.

    Returns:
      - matches
      - unmatched_detections
      - unmatched_trackers
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        candidate_mask = (iou_matrix > iou_threshold).astype(np.int32)

        if candidate_mask.sum(1).max() == 1 and candidate_mask.sum(0).max() == 1:
            matched_indices = np.stack(np.where(candidate_mask), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections, unmatched_trackers = _collect_unmatched_indices(
        matched_indices,
        len(detections),
        len(trackers),
    )

    matches = []
    for match_pair in matched_indices:
        detection_index, tracker_index = match_pair[0], match_pair[1]

        if iou_matrix[detection_index, tracker_index] < iou_threshold:
            unmatched_detections.append(detection_index)
            unmatched_trackers.append(tracker_index)
        else:
            matches.append(match_pair.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# ---------------------------------------------------------------------
# Embedding weighting
# ---------------------------------------------------------------------
def compute_aw_max_metric(emb_cost, w_association_emb, bottom=0.5):
    """
    Adaptive weighting for embedding similarities.

    Preserved as-is functionally.
    """
    weighted_embedding = np.full_like(emb_cost, w_association_emb)

    for row_index in range(emb_cost.shape[0]):
        sorted_indices = np.argsort(-emb_cost[row_index])

        if len(sorted_indices) < 2:
            continue

        if emb_cost[row_index, sorted_indices[0]] == 0:
            row_weight = 0
        else:
            row_weight = 1 - max(
                (emb_cost[row_index, sorted_indices[1]] / emb_cost[row_index, sorted_indices[0]]) - bottom,
                0,
            ) / (1 - bottom)

        weighted_embedding[row_index] *= row_weight

    for col_index in range(emb_cost.shape[1]):
        sorted_indices = np.argsort(-emb_cost[:, col_index])

        if len(sorted_indices) < 2:
            continue

        if emb_cost[sorted_indices[0], col_index] == 0:
            col_weight = 0
        else:
            col_weight = 1 - max(
                (emb_cost[sorted_indices[1], col_index] / emb_cost[sorted_indices[0], col_index]) - bottom,
                0,
            ) / (1 - bottom)

        weighted_embedding[:, col_index] *= col_weight

    return weighted_embedding * emb_cost


# ---------------------------------------------------------------------
# Main association function
# ---------------------------------------------------------------------
def associate(
    detections,
    trackers,
    asso_func,
    iou_threshold,
    velocities,
    previous_obs,
    vdc_weight,
    w,
    h,
    emb_cost=None,
    w_assoc_emb=None,
    aw_off=None,
    aw_param=None,
    pose_affinity=None,
    pose_reliability=None,
    pose_weight=0.0,
    hybrid_threshold=None,
    box_threshold_floor=None,
    pose_threshold_floor=None,
):
    """
    Associate detections to trackers using spatial similarity, motion consistency,
    optional embedding similarity, and optional pose similarity.

    Returns:
      - matches
      - unmatched_detections
      - unmatched_trackers

    Functionality is preserved from the original implementation.
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    # -------------------------------------------------------------
    # Motion direction consistency term
    # -------------------------------------------------------------
    direction_y, direction_x = speed_direction_batch(detections, previous_obs)

    velocity_y = velocities[:, 0]
    velocity_x = velocities[:, 1]

    velocity_y = np.repeat(velocity_y[:, np.newaxis], direction_y.shape[1], axis=1)
    velocity_x = np.repeat(velocity_x[:, np.newaxis], direction_x.shape[1], axis=1)

    cosine_angle_difference = velocity_x * direction_x + velocity_y * direction_y
    cosine_angle_difference = np.clip(cosine_angle_difference, a_min=-1, a_max=1)

    angle_difference = np.arccos(cosine_angle_difference)
    angle_difference = (np.pi / 2.0 - np.abs(angle_difference)) / np.pi

    valid_previous_observation_mask = np.ones(previous_obs.shape[0])
    valid_previous_observation_mask[np.where(previous_obs[:, 4] < 0)] = 0

    # -------------------------------------------------------------
    # Spatial similarity term
    # -------------------------------------------------------------
    iou_matrix = run_asso_func(asso_func, detections, trackers, w, h)

    detection_scores = np.repeat(
        detections[:, -1][:, np.newaxis],
        trackers.shape[0],
        axis=1,
    )

    valid_previous_observation_mask = np.repeat(
        valid_previous_observation_mask[:, np.newaxis],
        direction_x.shape[1],
        axis=1,
    )

    angle_difference_cost = (valid_previous_observation_mask * angle_difference) * vdc_weight
    angle_difference_cost = angle_difference_cost.T
    angle_difference_cost = angle_difference_cost * detection_scores

    # -------------------------------------------------------------
    # Pose contribution
    # -------------------------------------------------------------
    pose_affinity, effective_pose_term = _resolve_pose_terms(
        iou_matrix,
        pose_affinity,
        pose_reliability,
        pose_weight,
    )

    # -------------------------------------------------------------
    # Optional embedding contribution
    # -------------------------------------------------------------
    final_similarity = None
    if min(iou_matrix.shape) > 0:
        if emb_cost is None:
            emb_cost = 0
        else:
            emb_cost[iou_matrix <= 0] = 0

            if not aw_off:
                emb_cost = compute_aw_max_metric(
                    emb_cost,
                    w_assoc_emb,
                    bottom=aw_param,
                )
            else:
                emb_cost *= w_assoc_emb

        final_similarity = iou_matrix + angle_difference_cost + emb_cost + effective_pose_term
        matched_indices = linear_assignment(-final_similarity)

        if matched_indices.size == 0:
            matched_indices = np.empty(shape=(0, 2))
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections, unmatched_trackers = _collect_unmatched_indices(
        matched_indices,
        len(detections),
        len(trackers),
    )

    # -------------------------------------------------------------
    # Final hybrid gate
    # -------------------------------------------------------------
    matches = []
    (
        resolved_hybrid_threshold,
        resolved_box_floor,
        resolved_pose_floor,
    ) = _resolve_hybrid_gate_thresholds(
        iou_threshold,
        hybrid_threshold,
        box_threshold_floor,
        pose_threshold_floor,
    )

    for match_pair in matched_indices:
        detection_index, tracker_index = match_pair[0], match_pair[1]

        box_similarity = float(iou_matrix[detection_index, tracker_index])
        pose_similarity = (
            float(pose_affinity[detection_index, tracker_index])
            if pose_affinity is not None
            else 0.0
        )
        hybrid_similarity = (
            float(final_similarity[detection_index, tracker_index])
            if final_similarity is not None
            else box_similarity
        )

        hybrid_passes = hybrid_similarity >= resolved_hybrid_threshold
        evidence_passes = (
            (box_similarity >= resolved_box_floor)
            or (pose_similarity >= resolved_pose_floor)
        )

        if not (hybrid_passes and evidence_passes):
            unmatched_detections.append(detection_index)
            unmatched_trackers.append(tracker_index)
        else:
            matches.append(match_pair.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# ---------------------------------------------------------------------
# KITTI-specific association
# ---------------------------------------------------------------------
def associate_kitti(
    detections,
    trackers,
    det_cates,
    iou_threshold,
    velocities,
    previous_obs,
    vdc_weight,
):
    """
    KITTI-style association with category consistency constraint.
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    # -------------------------------------------------------------
    # Motion direction consistency term
    # -------------------------------------------------------------
    direction_y, direction_x = speed_direction_batch(detections, previous_obs)

    velocity_y = velocities[:, 0]
    velocity_x = velocities[:, 1]

    velocity_y = np.repeat(velocity_y[:, np.newaxis], direction_y.shape[1], axis=1)
    velocity_x = np.repeat(velocity_x[:, np.newaxis], direction_x.shape[1], axis=1)

    cosine_angle_difference = velocity_x * direction_x + velocity_y * direction_y
    cosine_angle_difference = np.clip(cosine_angle_difference, a_min=-1, a_max=1)

    angle_difference = np.arccos(cosine_angle_difference)
    angle_difference = (np.pi / 2.0 - np.abs(angle_difference)) / np.pi

    valid_previous_observation_mask = np.ones(previous_obs.shape[0])
    valid_previous_observation_mask[np.where(previous_obs[:, 4] < 0)] = 0
    valid_previous_observation_mask = np.repeat(
        valid_previous_observation_mask[:, np.newaxis],
        direction_x.shape[1],
        axis=1,
    )

    detection_scores = np.repeat(
        detections[:, -1][:, np.newaxis],
        trackers.shape[0],
        axis=1,
    )

    angle_difference_cost = (valid_previous_observation_mask * angle_difference) * vdc_weight
    angle_difference_cost = angle_difference_cost.T
    angle_difference_cost = angle_difference_cost * detection_scores

    # -------------------------------------------------------------
    # IoU term
    # -------------------------------------------------------------
    iou_matrix = iou_batch(detections, trackers)

    # -------------------------------------------------------------
    # Category mismatch penalty
    # -------------------------------------------------------------
    num_detections = detections.shape[0]
    num_trackers = trackers.shape[0]
    category_penalty_matrix = np.zeros((num_detections, num_trackers))

    for detection_index in range(num_detections):
        for tracker_index in range(num_trackers):
            if det_cates[detection_index] != trackers[tracker_index, 4]:
                category_penalty_matrix[detection_index][tracker_index] = -1e6

    cost_matrix = -iou_matrix - angle_difference_cost - category_penalty_matrix

    if min(iou_matrix.shape) > 0:
        candidate_mask = (iou_matrix > iou_threshold).astype(np.int32)

        if candidate_mask.sum(1).max() == 1 and candidate_mask.sum(0).max() == 1:
            matched_indices = np.stack(np.where(candidate_mask), axis=1)
        else:
            matched_indices = linear_assignment(cost_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections, unmatched_trackers = _collect_unmatched_indices(
        matched_indices,
        len(detections),
        len(trackers),
    )

    matches = []
    for match_pair in matched_indices:
        detection_index, tracker_index = match_pair[0], match_pair[1]

        if iou_matrix[detection_index, tracker_index] < iou_threshold:
            unmatched_detections.append(detection_index)
            unmatched_trackers.append(tracker_index)
        else:
            matches.append(match_pair.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)