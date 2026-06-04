"""RTMCC head — port of mmpose ``RTMCCHead`` for SimCC-style 1D keypoint heads.

The forward returns ``(pred_x, pred_y)`` SimCC logit tensors. Decoding into
keypoint coordinates lives in ``libs.giftpose.codecs.simcc``; flip-test
averaging lives in the runtime backend, not here.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from libs.giftpose.models.rtmcc_block import RTMCCBlock, ScaleNorm


class RTMCCHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_keypoints: int,
        input_size: Tuple[int, int],          # (W, H) in input-image pixels
        in_featuremap_size: Tuple[int, int],  # (W, H) of last backbone fmap
        simcc_split_ratio: float = 2.0,
        final_layer_kernel_size: int = 7,
        gau_hidden_dims: int = 256,
        gau_s: int = 128,
        gau_expansion_factor: float = 2.0,
        gau_act_fn: str = "SiLU",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio

        flatten_dims = in_featuremap_size[0] * in_featuremap_size[1]

        self.final_layer = nn.Conv2d(
            in_channels, num_keypoints,
            kernel_size=final_layer_kernel_size,
            stride=1, padding=final_layer_kernel_size // 2,
        )
        # mlp: ScaleNorm + Linear(flatten_dims -> hidden) — exact upstream layout
        # so checkpoint keys ``head.mlp.0.g`` (ScaleNorm) and ``head.mlp.1.weight``
        # (Linear) round-trip directly.
        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, gau_hidden_dims, bias=False),
        )
        self.gau = RTMCCBlock(
            in_token_dims=gau_hidden_dims,
            out_token_dims=gau_hidden_dims,
            s=gau_s,
            expansion_factor=gau_expansion_factor,
            act_fn=gau_act_fn,
            bias=False,
        )
        W = int(input_size[0] * simcc_split_ratio)
        H = int(input_size[1] * simcc_split_ratio)
        self.cls_x = nn.Linear(gau_hidden_dims, W, bias=False)
        self.cls_y = nn.Linear(gau_hidden_dims, H, bias=False)

    def forward(self, feat_last: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """feat_last: (B, in_channels, H_fm, W_fm)."""
        feat = self.final_layer(feat_last)        # (B, K, H_fm, W_fm)
        feat = torch.flatten(feat, start_dim=2)   # (B, K, H_fm*W_fm)
        feat = self.mlp(feat)                     # (B, K, hidden)
        feat = self.gau(feat)                     # (B, K, hidden)
        return self.cls_x(feat), self.cls_y(feat)  # (B, K, W*split), (B, K, H*split)
