# Motion Trackers for Video Analysis -Surya ⎜(Extended from Mikel Broström's work on boxmot(10.0.81))

import numpy as np

from .iou import iou_batch, centroid_batch, run_asso_func


# -------------------- Helper Functions --------------------

def _resolve_pose_costs(iou_matrix, pose_affinity, pose_reliability, pose_weight):
    if pose_affinity is None:
        return None, 0.0

    pose_affinity = np.asarray(pose_affinity, dtype=float)
    if pose_affinity.shape != iou_matrix.shape:
        raise ValueError(
            f"pose_affinity shape mismatch: expected {iou_matrix.shape}, got {pose_affinity.shape}"
        )
    pose_affinity = np.clip(pose_affinity, a_min=0.0, a_max=1.0)
    pose_cost = pose_affinity * float(pose_weight)

    if pose_reliability is None:
        pose_reliability_cost = 1.0
    else:
        pose_reliability = np.asarray(pose_reliability, dtype=float)
        if pose_reliability.shape != iou_matrix.shape:
            raise ValueError(
                f"pose_reliability shape mismatch: expected {iou_matrix.shape}, got {pose_reliability.shape}"
            )
        pose_reliability_cost = np.clip(pose_reliability, a_min=0.0, a_max=1.0)

    effective_pose_cost = pose_cost * pose_reliability_cost
    return pose_affinity, effective_pose_cost


def _resolve_hybrid_gate_thresholds(iou_threshold, hybrid_threshold, box_threshold_floor, pose_threshold_floor):
    resolved_hybrid_threshold = float(hybrid_threshold) if hybrid_threshold is not None else float(iou_threshold)
    resolved_box_floor = float(box_threshold_floor) if box_threshold_floor is not None else float(iou_threshold) * 0.5
    resolved_pose_floor = float(pose_threshold_floor) if pose_threshold_floor is not None else 0.30
    return resolved_hybrid_threshold, resolved_box_floor, resolved_pose_floor


def speed_direction_batch(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0
    CX2, CY2 = (tracks[:, 0] + tracks[:, 2]) / 2.0, (tracks[:, 1] + tracks[:, 3]) / 2.0
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx  # size: num_track x num_det


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array([list(zip(x, y))])


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns matches, unmatched_detections, and unmatched_trackers.
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def compute_aw_max_metric(emb_cost, w_association_emb, bottom=0.5):
    w_emb = np.full_like(emb_cost, w_association_emb)

    for idx in range(emb_cost.shape[0]):
        inds = np.argsort(-emb_cost[idx])
        # If there's less than two matches, just keep original weight
        if len(inds) < 2:
            continue
        if emb_cost[idx, inds[0]] == 0:
            row_weight = 0
        else:
            row_weight = 1 - max(
                (emb_cost[idx, inds[1]] / emb_cost[idx, inds[0]]) - bottom, 0
            ) / (1 - bottom)
        w_emb[idx] *= row_weight

    for idj in range(emb_cost.shape[1]):
        inds = np.argsort(-emb_cost[:, idj])
        # If there's less than two matches, just keep original weight
        if len(inds) < 2:
            continue
        if emb_cost[inds[0], idj] == 0:
            col_weight = 0
        else:
            col_weight = 1 - max(
                (emb_cost[inds[1], idj] / emb_cost[inds[0], idj]) - bottom, 0
            ) / (1 - bottom)
        w_emb[:, idj] *= col_weight

    return w_emb * emb_cost


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
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0

    iou_matrix = run_asso_func(asso_func, detections, trackers, w, h)
    #iou_matrix = iou_batch(detections, trackers)
    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    # iou_matrix = iou_matrix * scores # a trick sometiems works, we don't encourage this
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    pose_affinity, effective_pose_cost = _resolve_pose_costs(
        iou_matrix,
        pose_affinity,
        pose_reliability,
        pose_weight,
    )

    final_similarity = None
    if min(iou_matrix.shape):
        if emb_cost is None:
            emb_cost = 0
        else:
            emb_cost[iou_matrix <= 0] = 0
            if not aw_off:
                emb_cost = compute_aw_max_metric(emb_cost, w_assoc_emb, bottom=aw_param)
            else:
                emb_cost *= w_assoc_emb

        final_similarity = iou_matrix + angle_diff_cost + emb_cost + effective_pose_cost
        matched_indices = linear_assignment(-final_similarity)
        if matched_indices.size == 0:
            matched_indices = np.empty(shape=(0, 2))
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter matched pairs with hybrid gating
    matches = []
    resolved_hybrid_threshold, resolved_box_floor, resolved_pose_floor = _resolve_hybrid_gate_thresholds(
        iou_threshold,
        hybrid_threshold,
        box_threshold_floor,
        pose_threshold_floor,
    )

    for m in matched_indices:
        det_idx, trk_idx = m[0], m[1]
        box_value = float(iou_matrix[det_idx, trk_idx])
        pose_value = float(pose_affinity[det_idx, trk_idx]) if pose_affinity is not None else 0.0
        hybrid_value = float(final_similarity[det_idx, trk_idx]) if final_similarity is not None else box_value

        hybrid_ok = hybrid_value >= resolved_hybrid_threshold
        evidence_ok = (box_value >= resolved_box_floor) or (pose_value >= resolved_pose_floor)

        if not (hybrid_ok and evidence_ok):
            unmatched_detections.append(det_idx)
            unmatched_trackers.append(trk_idx)
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def associate_kitti(
    detections, trackers, det_cates, iou_threshold, velocities, previous_obs, vdc_weight
):
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    # Cost from the velocity direction consistency
    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    # Cost from IoU
    iou_matrix = iou_batch(detections, trackers)

    # With multiple categories, generate the cost for category mismatch
    num_dets = detections.shape[0]
    num_trk = trackers.shape[0]
    cate_matrix = np.zeros((num_dets, num_trk))
    for i in range(num_dets):
        for j in range(num_trk):
            if det_cates[i] != trackers[j, 4]:
                cate_matrix[i][j] = -1e6

    cost_matrix = -iou_matrix - angle_diff_cost - cate_matrix

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(cost_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
