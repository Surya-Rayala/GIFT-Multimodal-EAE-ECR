"""SimCC label decoding for RTMPose.

Mirrors mmpose's ``get_simcc_maximum`` + ``SimCCLabel.decode`` (with
``use_dark=False``, the configured RTMPose-x setting).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def get_simcc_maximum(
    simcc_x: np.ndarray, simcc_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Argmax decode of paired 1D SimCC distributions.

    Args:
        simcc_x: (N, K, Wx) raw logits/scores along x.
        simcc_y: (N, K, Wy) raw logits/scores along y.

    Returns:
        locs: (N, K, 2) float32 in SimCC scale (input image space *
            simcc_split_ratio).
        vals: (N, K) float32 confidence per keypoint = min(max_x, max_y).
    """
    assert simcc_x.ndim == 3 and simcc_y.ndim == 3
    x_locs = np.argmax(simcc_x, axis=2)
    y_locs = np.argmax(simcc_y, axis=2)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_x = np.amax(simcc_x, axis=2)
    max_y = np.amax(simcc_y, axis=2)
    vals = np.where(max_x < max_y, max_x, max_y).astype(np.float32)
    locs[vals <= 0.0] = -1
    return locs, vals


def decode_simcc(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    simcc_split_ratio: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """SimCC -> input-space keypoint coords + per-keypoint scores.

    Returns:
        keypoints: (N, K, 2) float32 in input-image coordinates (e.g. 288x384).
        scores: (N, K) float32.
    """
    keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints = keypoints / simcc_split_ratio
    return keypoints, scores
