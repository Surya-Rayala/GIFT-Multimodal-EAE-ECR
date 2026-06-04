"""Top-down crop affine transform — bbox -> 288x384 input.

Mirrors mmpose's ``GetBBoxCenterScale(padding=1.25)`` + ``TopdownAffine`` (with
``use_udp=False``, no rotation at inference) so the warp is bit-identical to
the legacy pipeline.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

DEFAULT_BBOX_PADDING = 1.25  # matches mmpose GetBBoxCenterScale default


def bbox_to_center_scale(
    bbox_xyxy: np.ndarray, padding: float = DEFAULT_BBOX_PADDING
) -> Tuple[np.ndarray, np.ndarray]:
    """(N, 4) xyxy -> (N, 2) center, (N, 2) scale (w*pad, h*pad)."""
    bbox = np.asarray(bbox_xyxy, dtype=np.float32)
    if bbox.ndim == 1:
        bbox = bbox[None, :]
    scale = (bbox[..., 2:] - bbox[..., :2]) * padding
    center = (bbox[..., 2:] + bbox[..., :2]) * 0.5
    return center.astype(np.float32), scale.astype(np.float32)


def fix_aspect_ratio(scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    """Reshape (w, h) so w/h matches ``aspect_ratio``, expanding the short axis."""
    w = scale[..., 0:1]
    h = scale[..., 1:2]
    return np.where(
        w > h * aspect_ratio,
        np.concatenate([w, w / aspect_ratio], axis=-1),
        np.concatenate([h * aspect_ratio, h], axis=-1),
    ).astype(np.float32)


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]], dtype=np.float32)
    return rot_mat @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Third reference point: rotate ``a-b`` by 90deg CCW around ``b``."""
    direction = a - b
    return b + np.array([-direction[1], direction[0]], dtype=np.float32)


def get_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot_deg: float,
    output_size: Tuple[int, int],
    inv: bool = False,
) -> np.ndarray:
    """2x3 affine matrix mapping the bbox region to ``output_size`` (W, H).

    Direct port of mmpose ``get_warp_matrix`` (``fix_aspect_ratio=True``).
    """
    src_w, src_h = float(scale[0]), float(scale[1])
    dst_w, dst_h = output_size

    rot_rad = np.deg2rad(rot_deg)
    src_dir = _rotate_point(np.array([src_w * -0.5, 0.0], dtype=np.float32), rot_rad)
    dst_dir = np.array([dst_w * -0.5, 0.0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + src_dir
    src[2] = _get_3rd_point(src[0], src[1])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0] = [dst_w * 0.5, dst_h * 0.5]
    dst[1] = dst[0] + dst_dir
    dst[2] = _get_3rd_point(dst[0], dst[1])

    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def warp_crop(
    image: np.ndarray,
    bbox_xyxy: np.ndarray,
    output_size: Tuple[int, int] = (288, 384),
    padding: float = DEFAULT_BBOX_PADDING,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop+warp ``image`` (BGR HxWx3) for one bbox into ``output_size`` (W, H).

    Returns:
        warped: BGR uint8 of shape (output_h, output_w, 3).
        warp_mat: 2x3 forward affine matrix (image-space -> input-space).
            Use the inverse to map predicted keypoints back to image space.
    """
    output_w, output_h = output_size
    aspect_ratio = output_w / output_h
    center, scale = bbox_to_center_scale(np.asarray(bbox_xyxy), padding=padding)
    scale = fix_aspect_ratio(scale, aspect_ratio)
    warp_mat = get_warp_matrix(center[0], scale[0], 0.0, output_size)
    warped = cv2.warpAffine(
        image, warp_mat, (output_w, output_h), flags=cv2.INTER_LINEAR
    )
    return warped, warp_mat


def apply_inverse_warps_batched(
    warp_mats: list[np.ndarray] | np.ndarray,
    kpts_in_input: np.ndarray,
) -> np.ndarray:
    """Map per-detection keypoints from input-space back to image-space.

    Replaces the per-detection ``cv2.invertAffineTransform`` + ``cv2.transform``
    Python loop with a single batched ``np.linalg.inv`` + matmul. Output is
    bit-equivalent: ``cv2.invertAffineTransform`` is just analytic inversion of
    a 2x3 augmented to 3x3.

    Args:
        warp_mats: list of N (2, 3) forward affine matrices (image -> input).
        kpts_in_input: (N, K, 2) keypoints in input-image coords.

    Returns:
        kpts_in_image: (N, K, 2) keypoints in original image coords (float32).
    """
    mats = np.stack(warp_mats, axis=0).astype(np.float32) if isinstance(warp_mats, list) else warp_mats.astype(np.float32)
    n = mats.shape[0]
    augmented = np.zeros((n, 3, 3), dtype=np.float32)
    augmented[:, :2, :] = mats
    augmented[:, 2, 2] = 1.0
    inv = np.linalg.inv(augmented)[:, :2, :]   # (N, 2, 3) — first two rows
    A = inv[:, :, :2]                           # (N, 2, 2)
    b = inv[:, :, 2:3]                          # (N, 2, 1)
    # Per-detection: out[n,k,:] = A[n] @ kpts_in_input[n,k,:] + b[n]
    out = (A @ kpts_in_input.transpose(0, 2, 1) + b).transpose(0, 2, 1)
    return out.astype(np.float32)
