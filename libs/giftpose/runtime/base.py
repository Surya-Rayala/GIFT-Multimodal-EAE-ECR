"""Backend abstract base — uniform API across PyTorch / ONNX / TensorRT.

A backend is responsible for:
  1. ``predict_detector(frame_bgr)`` -> per-frame (boxes_xyxy, scores) in
     original-image pixel coordinates.
  2. ``predict_pose(frame_bgr, boxes_xyxy)`` -> per-bbox (keypoints, scores)
     in original-image pixel coordinates.

The inferencer composes the two and assembles the schema consumed by
``processing_engine.py``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Default pose batch sizes warmed at backend init: powers of two covering the
# common 1-16-people-per-frame range, plus 32 — the TRT pose engine's max
# profile batch and the size of every full chunk when a busy frame is split
# (``trt_backend.predict_pose``). cuDNN's per-shape autotune (when
# ``benchmark=True``) and TRT's first-call shape resolution are primed for these
# sizes so a frame with N detections in this range hits a cached plan instead of
# paying a fresh ~5-50ms tune mid-pipeline.
DEFAULT_WARMUP_POSE_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16, 32)


@dataclass
class Detection:
    """Output of detector backend for one frame."""

    boxes_xyxy: np.ndarray  # (N, 4) float32, original-image coords
    scores: np.ndarray      # (N,) float32 in [0, 1]


class Backend(ABC):
    @abstractmethod
    def predict_detector(self, frame_bgr: np.ndarray) -> Detection:
        raise NotImplementedError

    @abstractmethod
    def predict_pose(
        self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns:
            keypoints: (N, K, 2) float32 in original-image coords.
            scores: (N, K) float32.
        """
        raise NotImplementedError

    def warmup(
        self, pose_batch_sizes: Sequence[int] = DEFAULT_WARMUP_POSE_SIZES
    ) -> None:
        """Run one synthetic detector + pose-per-size pass to prime caches.

        Why: cuDNN's ``benchmark=True`` autotunes per input shape; the pose
        head sees ``(N, 3, 384, 288)`` where N varies frame-to-frame, so a
        first-time encounter with each N stalls (~5-50ms). TensorRT and
        TorchScript also pay first-call kernel/jit setup. Running this once
        at init amortizes those costs out of the per-frame timing.
        """
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.predict_detector(dummy)
        for n in pose_batch_sizes:
            if n <= 0:
                continue
            # Non-overlapping synthetic boxes so warp_crop produces valid crops.
            boxes = np.stack(
                [np.array([10 + i * 5, 10, 110 + i * 5, 230], dtype=np.float32)
                 for i in range(n)],
                axis=0,
            )
            self.predict_pose(dummy, boxes)


def to_numpy(t: "torch.Tensor") -> np.ndarray:
    """Detach a torch tensor and materialize as fp32 numpy on CPU.

    Used by every backend to convert post-NMS / post-decode results to the
    numpy contract required by ``Detection`` and ``predict_pose``.
    """
    return t.detach().float().cpu().numpy()
