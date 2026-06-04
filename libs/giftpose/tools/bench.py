"""Microbench across PyTorch / ONNX / TRT backends on a single video frame.

Usage:
    python -m libs.giftpose.tools.bench \
        --video input/Videos/Crested_Gecko/8-Trimmed.mp4 \
        --frame 60 \
        --runs 50
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def _sync(device: str) -> None:
    """Force pending GPU work to complete before timing — required for fair
    comparison since CUDA / MPS launches return before the kernel runs.
    """
    try:
        import torch
    except ImportError:
        return
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _bench(name: str, fn, runs: int, device: str, warmup: int) -> tuple[str, float, float]:
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times)
    mean_ms = arr.mean() * 1000
    p95_ms = np.percentile(arr, 95) * 1000
    print(f"  {name}: mean={mean_ms:.1f}ms  p95={p95_ms:.1f}ms  fps={1000/mean_ms:.1f}")
    return name, mean_ms, p95_ms


def _main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--video", required=True)
    p.add_argument("--frame", type=int, default=60)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    p.add_argument("--det-weights", default="models/detect-best-mAP.pth")
    p.add_argument("--pose-weights", default="models/pose.pth")
    p.add_argument("--det-onnx", default="models/detect-best-mAP.onnx")
    p.add_argument("--pose-onnx", default="models/pose.onnx")
    p.add_argument("--det-engine", default="models/detect-best-mAP.engine")
    p.add_argument("--pose-engine", default="models/pose.engine")
    args = p.parse_args(argv)

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None
    print(f"frame: {frame.shape}  device: {args.device}  runs: {args.runs}")

    def _try_backend(name: str, ctor, *guards: str) -> tuple[str, object] | None:
        if guards and not all(Path(g).exists() for g in guards):
            return None
        try:
            return (name, ctor())
        except Exception as e:
            print(f"  {name} unavailable: {e}")
            return None

    def _make_pytorch():
        from libs.giftpose.runtime.pytorch_backend import PyTorchBackend
        return PyTorchBackend(args.det_weights, args.pose_weights, device=args.device)

    def _make_onnx():
        from libs.giftpose.runtime.onnx_backend import ONNXBackend
        return ONNXBackend(args.det_onnx, args.pose_onnx, device=args.device)

    def _make_trt():
        from libs.giftpose.runtime.trt_backend import TRTBackend
        return TRTBackend(args.det_engine, args.pose_engine)

    backends: list[tuple[str, object]] = [
        b for b in (
            _try_backend("pytorch", _make_pytorch),
            _try_backend("onnx", _make_onnx, args.det_onnx, args.pose_onnx),
            _try_backend("trt", _make_trt, args.det_engine, args.pose_engine),
        )
        if b is not None
    ]

    print("=== detector ===")
    for name, bb in backends:
        _bench(name, lambda b=bb: b.predict_detector(frame), args.runs, args.device, args.warmup)

    # Pose: use detector boxes from PyTorch as a fixed input set so all backends pose on the same N boxes.
    if backends:
        det_pt = backends[0][1].predict_detector(frame)
        keep = det_pt.scores > 0.3
        boxes = det_pt.boxes_xyxy[keep]
        print(f"=== pose (over {len(boxes)} detections) ===")
        for name, bb in backends:
            _bench(name, lambda b=bb, bx=boxes: b.predict_pose(frame, bx), args.runs, args.device, args.warmup)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
