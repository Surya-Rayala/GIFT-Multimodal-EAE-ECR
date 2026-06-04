"""RTMDet detection head — single-class person variant of
``RTMDetSepBNHead``.

The head uses point-anchor (``MlvlPointGenerator(offset=0)``) priors at strides
[8, 16, 32], per-level cls/reg conv stacks, and a final 1x1 ``rtm_cls`` /
``rtm_reg`` projection. Distance regression is decoded via
``DistancePointBBoxCoder`` (point + (l, t, r, b) -> xyxy).

Configuration (matches ``models/detect-best-mAP.pth``):
  - ``feat_channels = 192``, ``in_channels = 192`` (RTMDet-m widen=0.75)
  - ``stacked_convs = 2``
  - ``num_classes = 1`` (person)
  - ``exp_on_reg = True``, ``pred_kernel_size = 1``
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from libs.giftpose.models.csp_blocks import ConvBNAct
from libs.giftpose.postprocess.nms import multiclass_nms_torch


# Upstream RTMDet head uses ``norm_cfg=dict(type='SyncBN')`` (default PyTorch
# eps/momentum) — matched here to keep fp32 outputs bit-identical.
_DEFAULT_NORM = dict(type="BN")
_DEFAULT_ACT = dict(type="SiLU", inplace=True)


class RTMDetSepBNHead(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        in_channels: int = 192,
        feat_channels: int = 192,
        stacked_convs: int = 2,
        strides: Sequence[int] = (8, 16, 32),
        exp_on_reg: bool = True,
        pred_kernel_size: int = 1,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = list(strides)
        self.exp_on_reg = exp_on_reg
        self.pred_kernel_size = pred_kernel_size
        norm_cfg = norm_cfg or _DEFAULT_NORM
        act_cfg = act_cfg or _DEFAULT_ACT

        # Per-level cls + reg conv stacks. Each entry of cls_convs is itself an
        # nn.ModuleList of ``stacked_convs`` ConvBNAct layers — matches mmdet
        # ``RTMDetSepBNHead._init_layers``.
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for _ in range(len(self.strides)):
            cls_stack = nn.ModuleList()
            reg_stack = nn.ModuleList()
            for i in range(stacked_convs):
                chn = in_channels if i == 0 else feat_channels
                cls_stack.append(
                    ConvBNAct(chn, feat_channels, 3, stride=1, padding=1,
                              norm_cfg=norm_cfg, act_cfg=act_cfg)
                )
                reg_stack.append(
                    ConvBNAct(chn, feat_channels, 3, stride=1, padding=1,
                              norm_cfg=norm_cfg, act_cfg=act_cfg)
                )
            self.cls_convs.append(cls_stack)
            self.reg_convs.append(reg_stack)

        # Per-level 1x1 cls/reg projections.
        pad = pred_kernel_size // 2
        self.rtm_cls = nn.ModuleList([
            nn.Conv2d(feat_channels, num_classes, pred_kernel_size, padding=pad)
            for _ in range(len(self.strides))
        ])
        self.rtm_reg = nn.ModuleList([
            nn.Conv2d(feat_channels, 4, pred_kernel_size, padding=pad)
            for _ in range(len(self.strides))
        ])

    def forward(self, feats: Tuple[torch.Tensor, ...]) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        cls_scores: List[torch.Tensor] = []
        bbox_preds: List[torch.Tensor] = []
        for idx, x in enumerate(feats):
            cls_feat = x
            reg_feat = x
            for layer in self.cls_convs[idx]:
                cls_feat = layer(cls_feat)
            cls_score = self.rtm_cls[idx](cls_feat)
            for layer in self.reg_convs[idx]:
                reg_feat = layer(reg_feat)
            reg_dist = self.rtm_reg[idx](reg_feat)
            if self.exp_on_reg:
                reg_dist = reg_dist.exp() * self.strides[idx]
            else:
                reg_dist = reg_dist * self.strides[idx]
            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
        return cls_scores, bbox_preds


# Per-(feat_sizes, strides, device, dtype) cache. Priors depend only on the
# detector's input resolution + FPN strides + device + dtype, all fixed across
# frames in a giftpose session — recomputing them per detector forward burns
# tens of microseconds and a kernel launch for no gain.
_PRIORS_CACHE: dict[
    tuple[tuple[tuple[int, int], ...], tuple[int, ...], str, str],
    List[torch.Tensor],
] = {}


def make_grid_priors(
    feat_sizes: Sequence[Tuple[int, int]],
    strides: Sequence[int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    """``MlvlPointGenerator(offset=0)`` priors — per-level grid centers
    (x_center, y_center) in input-image pixel space.

    For each level: a (H_fm * W_fm, 2) tensor where each row is the (x, y)
    pixel-center of the cell. Cached across calls for a fixed detector input.
    """
    key = (
        tuple(tuple(s) for s in feat_sizes),
        tuple(strides),
        str(device),
        str(dtype),
    )
    cached = _PRIORS_CACHE.get(key)
    if cached is not None:
        return cached
    priors: List[torch.Tensor] = []
    for (h, w), stride in zip(feat_sizes, strides):
        ys = torch.arange(h, device=device, dtype=dtype) * stride
        xs = torch.arange(w, device=device, dtype=dtype) * stride
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        priors.append(torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1))
    _PRIORS_CACHE[key] = priors
    return priors


def distance_to_bbox(points: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    """``DistancePointBBoxCoder.decode``: (cx, cy) + (l, t, r, b) -> (x1, y1, x2, y2)."""
    x1 = points[..., 0] - distances[..., 0]
    y1 = points[..., 1] - distances[..., 1]
    x2 = points[..., 0] + distances[..., 2]
    y2 = points[..., 1] + distances[..., 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


def decode_rtmdet_dense(
    cls_scores: List[torch.Tensor],
    bbox_preds: List[torch.Tensor],
    strides: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid + distance-to-bbox + flatten across all FPN levels.

    Returns the dense pre-NMS outputs:
        boxes:  (M, 4) xyxy in letterboxed-canvas coords.
        scores: (M,) sigmoid'd class scores in [0, 1].

    M = sum of per-level grid points (e.g. 80x80 + 40x40 + 20x20 = 8400 for
    640x640 input). Used by both the runtime decode (followed by NMS) and the
    ONNX export wrapper (which keeps NMS out of the graph for static shapes).
    """
    assert len(cls_scores) == len(bbox_preds) == len(strides)
    device = cls_scores[0].device
    feat_sizes = [tuple(c.shape[-2:]) for c in cls_scores]
    priors = make_grid_priors(feat_sizes, strides, device=device)

    flat_scores: List[torch.Tensor] = []
    flat_boxes: List[torch.Tensor] = []
    for cls_score, bbox_pred, prior in zip(cls_scores, bbox_preds, priors):
        # Single-class: cls_score is (1, 1, H, W) — squeeze to (H*W,).
        s = cls_score.permute(0, 2, 3, 1).reshape(-1).sigmoid()
        # b is (l, t, r, b) distances in pixel units.
        b = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        flat_scores.append(s)
        flat_boxes.append(distance_to_bbox(prior, b))
    return torch.cat(flat_boxes, dim=0), torch.cat(flat_scores, dim=0)


def decode_rtmdet(
    cls_scores: List[torch.Tensor],
    bbox_preds: List[torch.Tensor],
    strides: Sequence[int],
    score_thr: float = 0.05,
    iou_threshold: float = 0.6,
    nms_pre: int = 30000,
    max_per_img: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid + distance-to-bbox + NMS for one image, single-class output.

    Returns:
        boxes: (K, 4) xyxy in letterboxed-canvas coords.
        scores: (K,) in [0, 1].
    """
    boxes, scores = decode_rtmdet_dense(cls_scores, bbox_preds, strides)
    return multiclass_nms_torch(boxes, scores, score_thr, iou_threshold, nms_pre, max_per_img)
