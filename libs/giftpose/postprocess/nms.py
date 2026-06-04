"""Class-agnostic batched NMS via torchvision.ops (ONNX-exportable since opset 12).

Also exposes ``multiclass_nms_numpy`` for runtime backends (ONNX / TRT) that
receive dense pre-NMS detector outputs as numpy arrays and do NMS in Python.
"""
from __future__ import annotations

import numpy as np
import torch
from torchvision.ops import nms


def multiclass_nms_numpy(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    score_thr: float,
    iou_threshold: float,
    nms_pre: int = 30000,
    max_per_img: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Numpy adapter — wraps ``torchvision.ops.nms`` for backends that emit
    dense pre-NMS detector outputs (ONNX / TRT graphs without graph-internal
    NMS — see ``libs/giftpose/export/onnx_export.py:_DetectorWithDecode``).
    """
    keep_mask = scores > score_thr
    boxes_xyxy = boxes_xyxy[keep_mask]
    scores = scores[keep_mask]
    if scores.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    if scores.size > nms_pre:
        order = np.argpartition(-scores, nms_pre)[:nms_pre]
        boxes_xyxy = boxes_xyxy[order]
        scores = scores[order]
    bt = torch.from_numpy(boxes_xyxy)
    st = torch.from_numpy(scores)
    keep = nms(bt, st, iou_threshold).cpu().numpy()
    if keep.size > max_per_img:
        keep = keep[:max_per_img]
    return boxes_xyxy[keep], scores[keep]


def _nms_device_aware(
    boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float
) -> torch.Tensor:
    """Run torchvision NMS on the input device when possible, falling back to
    a per-op CPU bounce for MPS (which lacks the kernel as of torch 2.8).

    The tensor is small after score-thresholding (~hundreds of boxes), so
    the MPS->CPU transfer is cheap relative to the network forward.
    """
    if boxes.device.type == "mps":
        keep = nms(boxes.cpu(), scores.cpu(), iou_threshold)
        return keep.to(boxes.device)
    return nms(boxes, scores, iou_threshold)


def multiclass_nms_torch(
    boxes_xyxy: torch.Tensor,
    scores: torch.Tensor,
    score_thr: float,
    iou_threshold: float,
    nms_pre: int = 30000,
    max_per_img: int = 300,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-class NMS path (the GIFT detector is person-only).

    Args:
        boxes_xyxy: (M, 4) float — flat across all FPN levels.
        scores: (M,) float — sigmoid class scores.
        score_thr: keep only detections with score > score_thr (post-NMS apply
            ``bbox_thr`` separately at the inferencer layer).
        iou_threshold: IoU threshold for NMS.
        nms_pre: keep top-``nms_pre`` by score before running NMS.
        max_per_img: cap on detections returned per image.

    Returns:
        boxes: (K, 4) float xyxy.
        scores: (K,) float.
    """
    keep_mask = scores > score_thr
    boxes_xyxy = boxes_xyxy[keep_mask]
    scores = scores[keep_mask]
    if scores.numel() == 0:
        return boxes_xyxy, scores
    if scores.numel() > nms_pre:
        topk = torch.topk(scores, nms_pre)
        boxes_xyxy = boxes_xyxy[topk.indices]
        scores = topk.values
    keep = _nms_device_aware(boxes_xyxy, scores, iou_threshold)
    if keep.numel() > max_per_img:
        keep = keep[:max_per_img]
    return boxes_xyxy[keep], scores[keep]
