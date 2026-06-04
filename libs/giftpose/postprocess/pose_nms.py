"""Pose-based NMS — direct port of mmpose's ``nearby_joints_nms``.

Used to dedupe near-duplicate detections after pose estimation: instances
whose keypoints lie close to a higher-scoring instance's keypoints get
suppressed.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def nearby_joints_nms(
    kpts_db: List[dict],
    dist_thr: float = 0.05,
    num_nearby_joints_thr: Optional[int] = None,
    score_per_joint: bool = False,
    max_dets: int = 30,
) -> List[int]:
    """Suppress instances with too many close-by keypoints to a higher-scoring
    instance.

    Args:
        kpts_db: list of ``{"keypoints": (K, 2|3) ndarray, "score": float}``.
        dist_thr: distance threshold relative to per-pose ``pose_area``
            (defaults match upstream).
        num_nearby_joints_thr: how many close joints make two poses ``close``.
            Defaults to ``num_keypoints // 2`` (upstream default); the
            inferencer call passes ``num_keypoints // 3`` explicitly.
        score_per_joint: if True, scores are per-joint and averaged.
        max_dets: cap on output length.

    Returns:
        List of indices into ``kpts_db`` to keep.
    """
    assert dist_thr > 0
    if not kpts_db:
        return []

    if score_per_joint:
        scores = np.array([np.asarray(k["score"]).mean() for k in kpts_db])
    else:
        scores = np.array([float(k["score"]) for k in kpts_db])

    kpts = np.stack([np.asarray(k["keypoints"])[..., :2] for k in kpts_db], axis=0)
    num_people, num_joints, _ = kpts.shape
    if num_nearby_joints_thr is None:
        num_nearby_joints_thr = num_joints // 2
    assert num_nearby_joints_thr < num_joints

    pose_area = kpts.max(axis=1) - kpts.min(axis=1)              # (N, 2)
    pose_area = np.sqrt((pose_area ** 2).sum(axis=1))            # (N,)
    # broadcast person-i area across (N_other, K_joints) -> (N, N, K)
    pose_area = np.tile(pose_area.reshape(num_people, 1, 1), (num_people, num_joints))
    close_dist_thr = pose_area * dist_thr                        # (N, N, K)

    # pairwise per-joint distances
    instance_dist = kpts[:, None] - kpts[None, :]                # (N, N, K, 2)
    instance_dist = np.sqrt((instance_dist ** 2).sum(axis=3))    # (N, N, K)
    close_instance_num = (instance_dist < close_dist_thr).sum(2) # (N, N)
    close_instance = close_instance_num > num_nearby_joints_thr  # (N, N)

    ignored, keep = set(), []
    order = np.argsort(scores)[::-1]
    for i in order:
        if i in ignored:
            continue
        keep_inds = close_instance[i].nonzero()[0]
        if keep_inds.size == 0:
            # Degenerate: ``pose_area`` collapses to 0 when all 26 keypoints
            # have the same value (e.g., all flagged ``-1`` by SimCC decode
            # for a low-confidence crop), making ``close_dist_thr`` 0 and
            # the strict ``instance_dist < 0`` False for every j (even
            # self). Keep this isolated detection on its own merits so the
            # frame doesn't crash on the np.argmax-of-empty.
            keep.append(int(i))
            ignored.add(int(i))
            continue
        keep_ind = keep_inds[np.argmax(scores[keep_inds])]
        if keep_ind not in ignored:
            keep.append(int(keep_ind))
            ignored = ignored.union(set(keep_inds.tolist()))

    if max_dets > 0 and len(keep) > max_dets:
        sub = np.argsort(scores[keep])[-1:-max_dets - 1:-1]
        keep = [keep[i] for i in sub]
    return keep
