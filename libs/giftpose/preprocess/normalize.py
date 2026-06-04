"""Per-task input normalization.

Pose: ImageNet-style mean/std with BGR->RGB swap, per the RTMPose-x config's
``PoseDataPreprocessor``.

Detector: RTMDet ``DetDataPreprocessor`` with mean/std in BGR ordering and
``bgr_to_rgb=False``, so the values are written in BGR (B, G, R) directly.
"""
from __future__ import annotations

import numpy as np
import torch

# Pose: applied AFTER bgr->rgb conversion. Values are mean/std in RGB order
# matching ImageNet (123.675, 116.28, 103.53).
POSE_MEAN_BGR = np.array([103.53, 116.28, 123.675], dtype=np.float32)  # BGR order
POSE_STD_BGR = np.array([57.375, 57.12, 58.395], dtype=np.float32)

# Detector: bgr_to_rgb=False — mean/std applied directly in BGR.
DET_MEAN_BGR = np.array([103.53, 116.28, 123.675], dtype=np.float32)
DET_STD_BGR = np.array([57.375, 57.12, 58.395], dtype=np.float32)


def _to_torch_chw(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    if image_bgr.ndim == 3:
        image_bgr = image_bgr[None]  # (1, H, W, 3)
    # ``non_blocking=True`` is a no-op on CPU/MPS but lets CUDA overlap the
    # host->device copy with the prior frame's compute when input is pinned.
    tensor = torch.from_numpy(np.ascontiguousarray(image_bgr)).to(device, non_blocking=True)
    return tensor.permute(0, 3, 1, 2).contiguous().to(torch.float32)


def normalize_pose_batch(
    crops_bgr: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Stack of BGR uint8 crops -> normalized RGB float tensor (N, 3, H, W).

    BGR -> RGB swap, then ``(x - mean) / std`` in RGB (mean/std stored in BGR
    are reversed to RGB on the fly).
    """
    nchw_bgr = _to_torch_chw(crops_bgr, device)
    # Swap channels: BGR -> RGB
    nchw_rgb = nchw_bgr.flip(dims=(1,))
    mean = torch.tensor(
        [POSE_MEAN_BGR[2], POSE_MEAN_BGR[1], POSE_MEAN_BGR[0]],
        device=device, dtype=torch.float32,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [POSE_STD_BGR[2], POSE_STD_BGR[1], POSE_STD_BGR[0]],
        device=device, dtype=torch.float32,
    ).view(1, 3, 1, 1)
    return ((nchw_rgb - mean) / std).to(dtype)


def normalize_det_input(
    image_bgr: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Single BGR uint8 letterboxed image -> normalized BGR float tensor
    (1, 3, H, W) for the RTMDet detector.
    """
    nchw_bgr = _to_torch_chw(image_bgr, device)
    mean = torch.tensor(DET_MEAN_BGR, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(DET_STD_BGR, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    return ((nchw_bgr - mean) / std).to(dtype)
