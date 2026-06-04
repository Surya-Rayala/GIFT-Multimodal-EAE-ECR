"""RTMCC GAU block — port of mmpose ``RTMCCBlock`` (self-attn variant) and
``ScaleNorm`` from mmpose.

The RTMPose-x config disables relative-position bias (``use_rel_bias=False``)
and rotary positional encoding (``pos_enc=False``), so the inference path is
just: ScaleNorm -> Linear (uv) -> SiLU -> split (u, v, base) -> base * gamma +
beta -> q/k from base -> attention -> u * (kernel @ v) -> Linear (o).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleNorm(nn.Module):
    """ScaleNorm from mmpose: trainable scalar gamma multiplied with sqrt(dim) /
    L2-norm-of-x division, with min-clamp.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class _Scale(nn.Module):
    """Per-channel learnable scalar, used as the residual scale in RTMCC."""

    def __init__(self, dim: int, init_value: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class RTMCCBlock(nn.Module):
    """Gated Attention Unit (GAU). Self-attention only, no rel-bias, no RoPE."""

    def __init__(
        self,
        in_token_dims: int,
        out_token_dims: int,
        s: int = 128,
        expansion_factor: float = 2.0,
        act_fn: str = "SiLU",
        bias: bool = False,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.s = s
        self.e = int(in_token_dims * expansion_factor)
        self.sqrt_s = math.sqrt(s)
        self.shortcut = in_token_dims == out_token_dims

        # uv: in_token_dims -> 2*e + s; o: e -> out_token_dims.
        self.o = nn.Linear(self.e, out_token_dims, bias=bias)
        self.uv = nn.Linear(in_token_dims, 2 * self.e + self.s, bias=bias)
        # gamma/beta: per-(q,k)-channel scale/shift on the shared base projection.
        self.gamma = nn.Parameter(torch.rand((2, self.s)))
        self.beta = nn.Parameter(torch.rand((2, self.s)))
        # Pre-attention norm. Residual is scaled per-channel.
        self.ln = ScaleNorm(in_token_dims, eps=eps)
        if self.shortcut:
            self.res_scale = _Scale(in_token_dims)
        if act_fn == "SiLU":
            self.act_fn = nn.SiLU(inplace=True)
        elif act_fn == "ReLU":
            self.act_fn = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported act_fn: {act_fn}")

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        x_n = self.ln(x)
        uv = self.act_fn(self.uv(x_n))
        u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=2)
        # base: (B, K, s) -> (B, K, 1, s) * (1, 1, 2, s) + (2, s) -> (B, K, 2, s)
        base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta
        q, k = torch.unbind(base, dim=2)
        # ReLU^2 attention kernel — squared ReLU (per the GAU paper).
        qk = torch.bmm(q, k.permute(0, 2, 1))
        kernel = torch.square(F.relu(qk / self.sqrt_s))
        return self.o(u * torch.bmm(kernel, v))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shortcut:
            return self.res_scale(x) + self._attention(x)
        return self._attention(x)
