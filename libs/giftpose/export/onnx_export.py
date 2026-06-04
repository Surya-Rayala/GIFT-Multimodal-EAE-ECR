"""ONNX export for the giftpose detector + pose models.

Detector ONNX wraps the post-processing decode + NMS inside the graph so
``onnxruntime`` consumers receive ``(boxes_xyxy, scores)`` directly. Pose ONNX
returns the raw SimCC ``(pred_x, pred_y)`` tensors; flip-test averaging stays
in the runtime backend (cheaper than baking it into the graph).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from libs.giftpose.models.detector import build_rtmdet_m_person
from libs.giftpose.models.pose_estimator import build_rtmpose_x_halpe26
from libs.giftpose.models.rtmdet_head import decode_rtmdet_dense
from libs.giftpose.weights.loader import load_state_dict_from_pth, strict_load


class _DetectorWithDecode(nn.Module):
    """Wrap the detector so ``forward`` returns the **dense pre-NMS** boxes +
    scores. NMS is intentionally kept out of the ONNX graph and run in Python
    by the runtime backends.

    Output shapes (for 640x640 input):
        boxes:  (M, 4) float32 — M = sum of FPN-level grid points
                                (80x80 + 40x40 + 20x20 = 8400). Static.
        scores: (M,)  float32 — sigmoid-applied class scores. Static.

    Why dense pre-NMS (vs. graph-internal NMS): TRT 10's ``enqueueV3`` cannot
    resolve data-dependent output shapes from ONNX's ``NonMaxSuppression`` op
    (the symbolic dim propagates and surfaces as ``shape calculation overflow
    -36028…``). Emitting dense outputs keeps the graph fully static; the
    runtime backends filter by ``score_thr`` and run NMS via
    ``torchvision.ops.nms`` in Python (~ms on 8400 boxes).
    """

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        cls_scores, bbox_preds = self.model.forward_features(x)
        return decode_rtmdet_dense(
            cls_scores, bbox_preds, strides=self.model.bbox_head.strides,
        )


def export_onnx_detector(
    weights_path: str | Path,
    out_path: str | Path,
    input_size: Tuple[int, int] = (640, 640),
    opset: int = 17,
) -> None:
    model = build_rtmdet_m_person().eval()
    strict_load(model, load_state_dict_from_pth(weights_path, "detector"))
    wrapped = _DetectorWithDecode(model).eval()
    H, W = input_size
    dummy = torch.zeros(1, 3, H, W, dtype=torch.float32)
    # No dynamic_axes — the wrapped detector pads to fixed (max_per_img, 4)
    # and (max_per_img,), so both ONNX outputs have fully static shapes.
    torch.onnx.export(
        wrapped,
        dummy,
        str(out_path),
        input_names=["images"],
        output_names=["boxes", "scores"],
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"wrote {out_path}")


def export_onnx_pose(
    weights_path: str | Path,
    out_path: str | Path,
    input_size: Tuple[int, int] = (288, 384),
    opset: int = 17,
) -> None:
    model = build_rtmpose_x_halpe26(input_size=input_size).eval()
    strict_load(model, load_state_dict_from_pth(weights_path, "pose"))
    W, H = input_size
    dummy = torch.zeros(1, 3, H, W, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["images"],
        output_names=["pred_x", "pred_y"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={
            "images": {0: "batch"},
            "pred_x": {0: "batch"},
            "pred_y": {0: "batch"},
        },
    )
    print(f"wrote {out_path}")


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--det-weights", default="models/detect-best-mAP.pth")
    p.add_argument("--pose-weights", default="models/pose.pth")
    p.add_argument("--det-out", default="models/detect-best-mAP.onnx")
    p.add_argument("--pose-out", default="models/pose.onnx")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--skip-detector", action="store_true")
    p.add_argument("--skip-pose", action="store_true")
    args = p.parse_args(argv)

    if not args.skip_detector:
        export_onnx_detector(args.det_weights, args.det_out, opset=args.opset)
    if not args.skip_pose:
        export_onnx_pose(args.pose_weights, args.pose_out, opset=args.opset)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
