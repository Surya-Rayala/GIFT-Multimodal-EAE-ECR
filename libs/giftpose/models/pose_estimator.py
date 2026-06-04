"""TopdownPoseEstimator — backbone + head wrapper for RTMPose.

Submodule names ``backbone`` and ``head`` mirror upstream so checkpoints with
the prefix layout ``backbone.*`` / ``head.*`` load via ``strict=True``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from libs.giftpose.models.cspnext import CSPNeXt
from libs.giftpose.models.rtmcc_head import RTMCCHead


class TopdownPoseEstimator(nn.Module):
    def __init__(self, backbone: CSPNeXt, head: RTMCCHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        return self.head(feats[-1])


def build_rtmpose_x_halpe26(input_size: tuple[int, int] = (288, 384)) -> TopdownPoseEstimator:
    """RTMPose-x configured for Halpe-26 at 288x384 — matches
    ``libs/mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py``.
    """
    backbone = CSPNeXt(
        deepen_factor=1.33,
        widen_factor=1.25,
        out_indices=(4,),
        expand_ratio=0.5,
        channel_attention=True,
    )
    # in_channels = 1024 * widen_factor = 1280; in_featuremap_size = (W/32, H/32) = (9, 12).
    head = RTMCCHead(
        in_channels=1280,
        num_keypoints=26,
        input_size=input_size,
        in_featuremap_size=(input_size[0] // 32, input_size[1] // 32),
        simcc_split_ratio=2.0,
        final_layer_kernel_size=7,
        gau_hidden_dims=256,
        gau_s=128,
        gau_expansion_factor=2.0,
        gau_act_fn="SiLU",
    )
    return TopdownPoseEstimator(backbone, head)
