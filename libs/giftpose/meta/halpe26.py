"""Halpe-26 keypoint metainfo, ported from
libs/mmpose/configs/_base_/datasets/halpe26.py.

The pose model is trained on Halpe-26: 17 COCO body keypoints + head/neck/hip
center points + 3 toe + 1 heel keypoint per foot. Indices 15 and 16 are the
ankles, which the existing tracker uses for ground-plane mapping.
"""
from __future__ import annotations

import numpy as np

NUM_KEYPOINTS = 26

KEYPOINT_NAMES: list[str] = [
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
    "head",            # 17
    "neck",            # 18
    "hip",             # 19
    "left_big_toe",    # 20
    "right_big_toe",   # 21
    "left_small_toe",  # 22
    "right_small_toe", # 23
    "left_heel",       # 24
    "right_heel",      # 25
]

# Per-keypoint flip permutation: index i maps to FLIP_INDICES[i] under
# horizontal flip. Center keypoints (nose, head, neck, hip) map to themselves.
_NAME_TO_IDX = {n: i for i, n in enumerate(KEYPOINT_NAMES)}
_SWAP_PAIRS = {
    "left_eye": "right_eye",
    "left_ear": "right_ear",
    "left_shoulder": "right_shoulder",
    "left_elbow": "right_elbow",
    "left_wrist": "right_wrist",
    "left_hip": "right_hip",
    "left_knee": "right_knee",
    "left_ankle": "right_ankle",
    "left_big_toe": "right_big_toe",
    "left_small_toe": "right_small_toe",
    "left_heel": "right_heel",
}
FLIP_INDICES: list[int] = list(range(NUM_KEYPOINTS))
for a, b in _SWAP_PAIRS.items():
    ia, ib = _NAME_TO_IDX[a], _NAME_TO_IDX[b]
    FLIP_INDICES[ia] = ib
    FLIP_INDICES[ib] = ia

# COCO/Halpe OKS sigmas (per-keypoint), used by pose-based NMS for OKS scoring.
SIGMAS: np.ndarray = np.array(
    [
        0.026, 0.025, 0.025, 0.035, 0.035,
        0.079, 0.079, 0.072, 0.072, 0.062, 0.062,
        0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
        0.026, 0.026, 0.066,
        0.079, 0.079, 0.079, 0.079, 0.079, 0.079,
    ],
    dtype=np.float32,
)
assert SIGMAS.shape == (NUM_KEYPOINTS,)

# Skeleton edges (pairs of keypoint indices). Used only by visualizers.
SKELETON: list[tuple[int, int]] = [
    (_NAME_TO_IDX[a], _NAME_TO_IDX[b])
    for a, b in [
        ("left_ankle", "left_knee"),
        ("left_knee", "left_hip"),
        ("left_hip", "hip"),
        ("right_ankle", "right_knee"),
        ("right_knee", "right_hip"),
        ("right_hip", "hip"),
        ("head", "neck"),
        ("neck", "hip"),
        ("neck", "left_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("neck", "right_shoulder"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_eye", "right_eye"),
        ("nose", "left_eye"),
        ("nose", "right_eye"),
        ("left_eye", "left_ear"),
        ("right_eye", "right_ear"),
        ("left_ear", "left_shoulder"),
        ("right_ear", "right_shoulder"),
        ("left_ankle", "left_big_toe"),
        ("left_ankle", "left_small_toe"),
        ("left_ankle", "left_heel"),
        ("right_ankle", "right_big_toe"),
        ("right_ankle", "right_small_toe"),
        ("right_ankle", "right_heel"),
    ]
]
