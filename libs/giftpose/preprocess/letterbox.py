"""Letterbox resize with aspect-ratio preservation, matching mmdet's
``Resize(keep_ratio=True) + Pad(size=(640,640), pad_val=114)`` test pipeline.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

PAD_VALUE = 114


def letterbox(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    pad_value: int = PAD_VALUE,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize ``image`` keeping aspect ratio so the long side fits in
    ``target_size``, then pad bottom-right with ``pad_value``.

    Args:
        image: BGR uint8 of shape (H, W, 3).
        target_size: (W, H) of output canvas. RTMDet uses (640, 640).
        pad_value: pad fill value (114 matches RTMDet test pipeline).

    Returns:
        padded: BGR uint8 of shape (target_h, target_w, 3).
        scale: ratio used to resize the original image.
        pad_offset: (pad_left, pad_top) — always (0, 0) for bottom-right padding,
            kept for symmetry with the inverse function.
    """
    src_h, src_w = image.shape[:2]
    tgt_w, tgt_h = target_size
    scale = min(tgt_w / src_w, tgt_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((tgt_h, tgt_w, 3), pad_value, dtype=image.dtype)
    padded[:new_h, :new_w] = resized
    return padded, float(scale), (0, 0)


def undo_letterbox_xyxy(
    boxes_xyxy: np.ndarray,
    scale: float,
    pad_offset: Tuple[int, int],
    original_shape: Tuple[int, int],
) -> np.ndarray:
    """Map boxes from letterboxed canvas coords back to original-image coords.

    Args:
        boxes_xyxy: (N, 4) float in canvas pixel coordinates.
        scale: scale used in :func:`letterbox`.
        pad_offset: (pad_left, pad_top) from :func:`letterbox`.
        original_shape: (H, W) of the original image.

    Returns:
        boxes_xyxy: (N, 4) float in original-image pixel coordinates, clipped
        to image bounds.
    """
    if boxes_xyxy.size == 0:
        return boxes_xyxy.copy()
    pad_l, pad_t = pad_offset
    boxes = boxes_xyxy.astype(np.float32, copy=True)
    boxes[:, 0::2] = (boxes[:, 0::2] - pad_l) / scale
    boxes[:, 1::2] = (boxes[:, 1::2] - pad_t) / scale
    h, w = original_shape
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, w - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, h - 1)
    return boxes
