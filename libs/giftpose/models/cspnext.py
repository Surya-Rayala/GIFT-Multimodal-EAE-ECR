"""CSPNeXt backbone — port of mmpose ``CSPNeXt`` (P5 arch).

Module attribute names match upstream so checkpoints load via ``strict=True``:
``stem`` (Sequential of 3 ConvBNAct) + ``stage1`` ... ``stage4`` (each Sequential
of stride-2 Conv -> optional SPPBottleneck -> CSPLayer).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

from libs.giftpose.models.csp_blocks import ConvBNAct, CSPLayer, SPPBottleneck


# Matches upstream backbone configs which use ``norm_cfg=dict(type='SyncBN')``
# with no overrides, i.e. PyTorch defaults (eps=1e-5, momentum=0.1). At
# inference, mmengine's ``revert_sync_batchnorm`` swaps SyncBN for a regular BN
# carrying these same defaults — values must match exactly or fp32 outputs
# diverge.
_DEFAULT_NORM = dict(type="BN")
_DEFAULT_ACT = dict(type="SiLU", inplace=True)

# (in_ch, out_ch, num_blocks, add_identity, use_spp) — matches upstream.
_ARCH_P5 = [
    (64, 128, 3, True, False),
    (128, 256, 6, True, False),
    (256, 512, 6, True, False),
    (512, 1024, 3, False, True),
]


class CSPNeXt(nn.Module):
    """RTMDet/RTMPose backbone."""

    def __init__(
        self,
        deepen_factor: float,
        widen_factor: float,
        out_indices: Sequence[int] = (2, 3, 4),
        expand_ratio: float = 0.5,
        channel_attention: bool = True,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
        spp_kernel_sizes: Sequence[int] = (5, 9, 13),
    ) -> None:
        super().__init__()
        norm_cfg = norm_cfg or _DEFAULT_NORM
        act_cfg = act_cfg or _DEFAULT_ACT
        self.out_indices = tuple(out_indices)

        stem_w0 = int(_ARCH_P5[0][0] * widen_factor // 2)
        stem_w1 = int(_ARCH_P5[0][0] * widen_factor)
        # Stem: 3 -> stem_w0 -> stem_w0 -> stem_w1, all 3x3 ConvBNAct.
        self.stem = nn.Sequential(
            ConvBNAct(3, stem_w0, 3, stride=2, padding=1,
                      norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvBNAct(stem_w0, stem_w0, 3, stride=1, padding=1,
                      norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvBNAct(stem_w0, stem_w1, 3, stride=1, padding=1,
                      norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        self._layer_names = ["stem"]

        for i, (in_ch_arch, out_ch_arch, num_blocks_arch, add_identity, use_spp) in enumerate(_ARCH_P5):
            in_ch = int(in_ch_arch * widen_factor)
            out_ch = int(out_ch_arch * widen_factor)
            num_blocks = max(round(num_blocks_arch * deepen_factor), 1)
            stage_modules: list[nn.Module] = [
                ConvBNAct(in_ch, out_ch, 3, stride=2, padding=1,
                          norm_cfg=norm_cfg, act_cfg=act_cfg),
            ]
            if use_spp:
                stage_modules.append(
                    SPPBottleneck(out_ch, out_ch, kernel_sizes=spp_kernel_sizes,
                                  norm_cfg=norm_cfg, act_cfg=act_cfg)
                )
            stage_modules.append(
                CSPLayer(
                    out_ch, out_ch, expand_ratio=expand_ratio,
                    num_blocks=num_blocks, add_identity=add_identity,
                    use_cspnext_block=True, channel_attention=channel_attention,
                    norm_cfg=norm_cfg, act_cfg=act_cfg,
                )
            )
            stage = nn.Sequential(*stage_modules)
            self.add_module(f"stage{i + 1}", stage)
            self._layer_names.append(f"stage{i + 1}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outs: list[torch.Tensor] = []
        for i, name in enumerate(self._layer_names):
            x = getattr(self, name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
