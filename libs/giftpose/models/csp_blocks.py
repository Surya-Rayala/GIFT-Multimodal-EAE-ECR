"""CSP / RTMDet building blocks — pure PyTorch ports of mmcv ConvModule and the
mmpose CSPLayer / SPPBottleneck / ChannelAttention / CSPNeXtBlock /
DepthwiseSeparableConvModule.

Module attribute names mirror the upstream layout so vendored mmengine state
dicts load via ``strict=True`` with no key remapping.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def _build_act(act_cfg: dict | None) -> nn.Module:
    if act_cfg is None:
        return nn.Identity()
    typ = act_cfg.get("type", "ReLU")
    inplace = act_cfg.get("inplace", True)
    if typ == "SiLU":
        return nn.SiLU(inplace=inplace)
    if typ == "ReLU":
        return nn.ReLU(inplace=inplace)
    if typ == "Swish":
        # mmcv "Swish" alias for SiLU
        return nn.SiLU(inplace=inplace)
    if typ == "LeakyReLU":
        return nn.LeakyReLU(act_cfg.get("negative_slope", 0.01), inplace=inplace)
    raise ValueError(f"Unsupported act type: {typ}")


def _build_norm(norm_cfg: dict | None, channels: int) -> nn.Module:
    if norm_cfg is None:
        return nn.Identity()
    typ = norm_cfg.get("type", "BN")
    eps = norm_cfg.get("eps", 1e-5)
    momentum = norm_cfg.get("momentum", 0.1)
    # SyncBN at training time -> standard BN at inference (state-dict compatible)
    if typ in ("BN", "SyncBN", "BN2d"):
        return nn.BatchNorm2d(channels, eps=eps, momentum=momentum)
    raise ValueError(f"Unsupported norm type: {typ}")


class ConvBNAct(nn.Module):
    """Mirror of mmcv ``ConvModule`` with submodule names ``conv`` / ``bn`` /
    ``activate`` so checkpoints load without renames.

    The bias on ``conv`` is auto-disabled when a norm is present (mmcv default).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
        bias: bool | str = "auto",
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        if bias == "auto":
            bias = norm_cfg is None
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            groups=groups, bias=bool(bias),
        )
        self.bn = _build_norm(norm_cfg, out_channels)
        self.activate = _build_act(act_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class DepthwiseSeparableConvBNAct(nn.Module):
    """Mirror of mmcv ``DepthwiseSeparableConvModule`` with submodule names
    ``depthwise_conv`` and ``pointwise_conv``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.depthwise_conv = ConvBNAct(
            in_channels, in_channels, kernel_size, stride=stride, padding=padding,
            groups=in_channels, norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.pointwise_conv = ConvBNAct(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise_conv(self.depthwise_conv(x))


class ChannelAttention(nn.Module):
    """Mirror of mmpose ``ChannelAttention`` — global avg pool + 1x1 conv + Hardsigmoid."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out


class CSPNeXtBlock(nn.Module):
    """The basic bottleneck used in CSPNeXt: 3x3 ConvBNAct + DW-Sep 5x5 ConvBNAct."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        add_identity: bool = True,
        kernel_size: int = 5,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBNAct(
            in_channels, hidden_channels, 3, stride=1, padding=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.conv2 = DepthwiseSeparableConvBNAct(
            hidden_channels, out_channels, kernel_size, stride=1,
            padding=kernel_size // 2, norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.add_identity = add_identity and (in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return out + x if self.add_identity else out


class DarknetBottleneck(nn.Module):
    """1x1 ConvBNAct (squeeze) + 3x3 ConvBNAct (expand) — DarknetBottleneck variant."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        add_identity: bool = True,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBNAct(
            in_channels, hidden_channels, 1, stride=1, padding=0,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.conv2 = ConvBNAct(
            hidden_channels, out_channels, 3, stride=1, padding=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.add_identity = add_identity and (in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return out + x if self.add_identity else out


class CSPLayer(nn.Module):
    """Cross Stage Partial layer with optional CSPNeXt blocks + channel attn."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        add_identity: bool = True,
        use_cspnext_block: bool = False,
        channel_attention: bool = False,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        block_cls = CSPNeXtBlock if use_cspnext_block else DarknetBottleneck
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.main_conv = ConvBNAct(
            in_channels, mid_channels, 1, stride=1, padding=0,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.short_conv = ConvBNAct(
            in_channels, mid_channels, 1, stride=1, padding=0,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.final_conv = ConvBNAct(
            2 * mid_channels, out_channels, 1, stride=1, padding=0,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.blocks = nn.Sequential(*[
            block_cls(
                mid_channels, mid_channels, expansion=1.0,
                add_identity=add_identity,
                norm_cfg=norm_cfg, act_cfg=act_cfg,
            )
            for _ in range(num_blocks)
        ])
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_short = self.short_conv(x)
        x_main = self.blocks(self.main_conv(x))
        x_final = torch.cat((x_main, x_short), dim=1)
        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)


class SPPBottleneck(nn.Module):
    """Spatial Pyramid Pooling — 1x1 conv -> N maxpools -> concat -> 1x1 conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Sequence[int] = (5, 9, 13),
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = ConvBNAct(
            in_channels, mid_channels, 1, stride=1, padding=0,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        self.conv2 = ConvBNAct(
            mid_channels * (len(kernel_sizes) + 1), out_channels, 1, stride=1, padding=0,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.cat([x] + [p(x) for p in self.poolings], dim=1)
        return self.conv2(x)
