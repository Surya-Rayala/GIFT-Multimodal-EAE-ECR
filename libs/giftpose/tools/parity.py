"""Layered parity harness — localize where the ONNX/TRT path diverges from the
working PyTorch / TorchScript path.

PyTorch and TorchScript are the accuracy reference (they share the same eager
math). ONNX and TRT are under suspicion. This tool runs ONE real frame through
each *available* backend and diffs at four taps so the divergence is pinned to a
specific stage AND a specific backend instead of guessed:

  Tap 1  detector dense pre-NMS  (boxes (M,4), scores (M,))  -> export vs engine
  Tap 2  detector post-NMS boxes (count + IoU recall vs eager) -> missing dets
  Tap 3  pose SimCC (pred_x, pred_y) on fixed crops          -> pose divergence
  Tap 4  final keypoints (mean px error vs eager)            -> end-to-end

Decision gate:
  * Tap 1/3 diverge under **onnxruntime**  -> ONNX export bug (or stale artifact).
  * Match under onnxruntime, diverge under **TRT** -> engine / precision bug.

On a machine without CUDA/TRT (e.g. macOS dev box) the eager-vs-onnxruntime legs
already separate "export bug" from "engine bug". The TRT legs auto-skip when no
``.engine`` is present and run on the GPU box.

Usage:
    python -m libs.giftpose.tools.parity \
        --video input/Videos/Crested_Gecko/8-Trimmed.mp4 --frame 60
    # add --clip-frames 200 to measure per-stage timing drift over a clip
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

from libs.giftpose.codecs.simcc import decode_simcc
from libs.giftpose.meta import FLIP_INDICES
from libs.giftpose.models.detector import build_rtmdet_m_person
from libs.giftpose.models.pose_estimator import build_rtmpose_x_halpe26
from libs.giftpose.models.rtmdet_head import decode_rtmdet_dense
from libs.giftpose.export.onnx_export import (
    _DetectorWithDecode,
    export_onnx_detector,
    export_onnx_pose,
)
from libs.giftpose.postprocess.nms import multiclass_nms_torch
from libs.giftpose.preprocess.letterbox import letterbox, undo_letterbox_xyxy
from libs.giftpose.preprocess.normalize import normalize_det_input, normalize_pose_batch
from libs.giftpose.preprocess.topdown_affine import warp_crop
from libs.giftpose.weights.loader import load_state_dict_from_pth, strict_load


# ---- small numeric helpers -------------------------------------------------

def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 and b.size == 0:
        return 0.0
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(Na,4) x (Nb,4) xyxy -> (Na,Nb) IoU."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def _recall_at_iou(ref_boxes: np.ndarray, test_boxes: np.ndarray, iou_thr: float = 0.5) -> float:
    """Fraction of reference boxes matched by some test box at >= iou_thr."""
    if len(ref_boxes) == 0:
        return 1.0
    if len(test_boxes) == 0:
        return 0.0
    iou = _iou_matrix(ref_boxes, test_boxes)
    return float((iou.max(axis=1) >= iou_thr).mean())


# ---- eager reference (PyTorch math, == TorchScript) ------------------------

class _Eager:
    """Eager detector+pose used as the accuracy reference for every tap."""

    def __init__(self, det_weights: str, pose_weights: str, device: str = "cpu"):
        self.device = torch.device(device)
        det = build_rtmdet_m_person().to(self.device).eval()
        strict_load(det, load_state_dict_from_pth(det_weights, "detector"))
        self.det_wrapped = _DetectorWithDecode(det).to(self.device).eval()
        self.pose = build_rtmpose_x_halpe26().to(self.device).eval()
        strict_load(self.pose, load_state_dict_from_pth(pose_weights, "pose"))

    @torch.inference_mode()
    def det_dense(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        boxes, scores = self.det_wrapped(x)
        return boxes.cpu().numpy(), scores.cpu().numpy()

    @torch.inference_mode()
    def pose_simcc(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        px, py = self.pose(x)
        return px.cpu().numpy(), py.cpu().numpy()


# ---- onnxruntime sessions (export under suspicion) -------------------------

class _Ort:
    def __init__(self, det_onnx: str, pose_onnx: str):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.det = ort.InferenceSession(det_onnx, sess_options=so, providers=["CPUExecutionProvider"])
        self.pose = ort.InferenceSession(pose_onnx, sess_options=so, providers=["CPUExecutionProvider"])

    def det_dense(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        boxes, scores = self.det.run(None, {"images": x})
        return boxes, scores

    def pose_simcc(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        px, py = self.pose.run(None, {"images": x})
        return px, py


# ---- main ------------------------------------------------------------------

def _read_frame(video: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"could not read frame {frame_idx} from {video}")
    return frame


def _main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--video", default="input/Videos/Crested_Gecko/8-Trimmed.mp4")
    p.add_argument("--frame", type=int, default=60)
    p.add_argument("--det-weights", default="models/detect-best-mAP.pth")
    p.add_argument("--pose-weights", default="models/pose.pth")
    p.add_argument("--det-onnx", default=None,
                   help="ONNX detector to test. Default: export fresh from --det-weights into a temp dir.")
    p.add_argument("--pose-onnx", default=None)
    p.add_argument("--det-engine", default="models/detect-best-mAP.engine")
    p.add_argument("--pose-engine", default="models/pose.engine")
    p.add_argument("--device", default="cpu")
    p.add_argument("--score-thr", type=float, default=0.05)
    p.add_argument("--bbox-thr", type=float, default=0.5,
                   help="Post-NMS gate matching processing_engine's box_conf_threshold default.")
    args = p.parse_args(argv)

    frame = _read_frame(args.video, args.frame)
    print(f"frame {args.frame}: {frame.shape}  device={args.device}")

    eager = _Eager(args.det_weights, args.pose_weights, device=args.device)
    dev = torch.device(args.device)

    # Shared detector input.
    padded, scale, pad = letterbox(frame, (640, 640))
    det_x_t = normalize_det_input(padded, dev, dtype=torch.float32)
    det_x_np = det_x_t.cpu().numpy()

    # Eager reference: dense + post-NMS + gated boxes.
    ref_boxes_d, ref_scores_d = eager.det_dense(det_x_t)
    rb_t, rs_t = multiclass_nms_torch(
        torch.from_numpy(ref_boxes_d), torch.from_numpy(ref_scores_d),
        score_thr=args.score_thr, iou_threshold=0.6, max_per_img=100,
    )
    ref_boxes_nms = undo_letterbox_xyxy(rb_t.numpy(), scale, pad, frame.shape[:2])
    ref_scores_nms = rs_t.numpy()
    ref_gate = ref_scores_nms > args.bbox_thr
    print(f"\n[eager reference] dense={len(ref_scores_d)}  "
          f"scores>{args.score_thr}={int((ref_scores_d > args.score_thr).sum())}  "
          f"post-NMS={len(ref_scores_nms)}  >{args.bbox_thr}(gated)={int(ref_gate.sum())}")

    # Fixed crop set (from eager gated boxes) so pose taps compare on identical input.
    gated_boxes = ref_boxes_nms[ref_gate]
    if len(gated_boxes) == 0:
        gated_boxes = ref_boxes_nms[:1] if len(ref_boxes_nms) else np.zeros((0, 4), np.float32)
    crops, warp_mats = [], []
    for box in gated_boxes:
        c, m = warp_crop(frame, box, output_size=(288, 384))
        crops.append(c); warp_mats.append(m)
    if crops:
        crops_arr = np.stack(crops, axis=0)
        pose_x_t = normalize_pose_batch(crops_arr, dev, dtype=torch.float32)
        pose_x_np = pose_x_t.cpu().numpy()
        ref_px, ref_py = eager.pose_simcc(pose_x_t)
        ref_kpts, _ = decode_simcc(ref_px, ref_py, simcc_split_ratio=2.0)
    else:
        pose_x_np = None
        ref_px = ref_py = ref_kpts = None
        print("  (no gated boxes — pose taps skipped)")

    # ---- ONNX leg (export fresh unless paths given) ----
    tmp = None
    det_onnx, pose_onnx = args.det_onnx, args.pose_onnx
    if det_onnx is None or pose_onnx is None:
        tmp = tempfile.mkdtemp(prefix="giftpose_parity_")
        det_onnx = det_onnx or str(Path(tmp) / "det.onnx")
        pose_onnx = pose_onnx or str(Path(tmp) / "pose.onnx")
        print(f"\n[onnx] exporting fresh -> {tmp}")
        export_onnx_detector(args.det_weights, det_onnx)
        export_onnx_pose(args.pose_weights, pose_onnx)

    print("\n=== ONNX Runtime vs eager ===")
    try:
        ort = _Ort(det_onnx, pose_onnx)
        o_boxes_d, o_scores_d = ort.det_dense(det_x_np)
        print(f"  Tap1 detector dense: boxes maxdiff={_max_abs_diff(ref_boxes_d, o_boxes_d):.3e}  "
              f"scores maxdiff={_max_abs_diff(ref_scores_d, o_scores_d):.3e}  "
              f"(scores>{args.score_thr}: eager={int((ref_scores_d>args.score_thr).sum())} "
              f"onnx={int((o_scores_d>args.score_thr).sum())})")
        ob_t, os_t = multiclass_nms_torch(
            torch.from_numpy(o_boxes_d), torch.from_numpy(o_scores_d),
            score_thr=args.score_thr, iou_threshold=0.6, max_per_img=100,
        )
        o_boxes_nms = undo_letterbox_xyxy(ob_t.numpy(), scale, pad, frame.shape[:2])
        o_scores_nms = os_t.numpy()
        o_gate = int((o_scores_nms > args.bbox_thr).sum())
        rec = _recall_at_iou(ref_boxes_nms[ref_gate], o_boxes_nms[o_scores_nms > args.bbox_thr])
        print(f"  Tap2 detector post-NMS: eager_gated={int(ref_gate.sum())} onnx_gated={o_gate}  "
              f"recall@0.5(of eager gated)={rec:.3f}")
        if pose_x_np is not None:
            o_px, o_py = ort.pose_simcc(pose_x_np)
            print(f"  Tap3 pose SimCC: pred_x maxdiff={_max_abs_diff(ref_px, o_px):.3e}  "
                  f"pred_y maxdiff={_max_abs_diff(ref_py, o_py):.3e}")
            o_kpts, _ = decode_simcc(o_px, o_py, simcc_split_ratio=2.0)
            kerr = float(np.abs(ref_kpts - o_kpts).mean())
            print(f"  Tap4 keypoints (input space): mean abs err={kerr:.4f}px")
    except Exception as e:
        print(f"  ONNX leg failed: {e}")

    # ---- TRT leg (GPU box only) ----
    if Path(args.det_engine).exists() and Path(args.pose_engine).exists():
        print("\n=== TensorRT vs eager ===")
        try:
            from libs.giftpose.runtime.trt_backend import TRTBackend, _load_engine
            trt = TRTBackend(args.det_engine, args.pose_engine, warmup=False)
            det_x_cuda = det_x_t.to("cuda").contiguous()
            t_outs = trt._run_engine(trt._det_ctx, det_x_cuda, out_shapes={})
            t_boxes_d = t_outs["boxes"].cpu().numpy()
            t_scores_d = t_outs["scores"].cpu().numpy()
            print(f"  Tap1 detector dense: boxes maxdiff={_max_abs_diff(ref_boxes_d, t_boxes_d):.3e}  "
                  f"scores maxdiff={_max_abs_diff(ref_scores_d, t_scores_d):.3e}  "
                  f"(scores>{args.score_thr}: eager={int((ref_scores_d>args.score_thr).sum())} "
                  f"trt={int((t_scores_d>args.score_thr).sum())})")
            if pose_x_np is not None:
                t_kpts, _ = trt.predict_pose(frame, gated_boxes)
                # eager final keypoints (image space) for the same boxes
                from libs.giftpose.preprocess.topdown_affine import apply_inverse_warps_batched
                ref_kpts_img = apply_inverse_warps_batched(warp_mats, ref_kpts)
                kerr = float(np.abs(ref_kpts_img - t_kpts).mean())
                print(f"  Tap4 keypoints (image space): mean abs err={kerr:.4f}px")
        except Exception as e:
            print(f"  TRT leg failed: {e}")
    else:
        print("\n[trt] engines not present — skipping (run on GPU box).")

    print("\nDecision: divergence under ONNX => export bug (fix B1). "
          "Match under ONNX but diverge under TRT => engine/precision (fix B2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
