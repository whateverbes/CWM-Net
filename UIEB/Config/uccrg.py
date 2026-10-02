# -*- coding: utf-8 -*-
"""Underwater Cross-Channel Residual Gate (UCCRG) used inside CCRM.

Gamma maps follow UCCNet (TMM 2024) Eq. (4):
  Gamma_lk(x) = F( (I_k(x) - I_l(x)) * (1 - I_l(x)) )

Inserted after Inc, before CAM: features are modulated by pixel-wise gates
derived from input RGB channel residuals (meaningful from initialization).
proj is zero-init -> identity gate at start.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightResidualF(nn.Module):
    """Compact F(x): two 3x3 convs + BN + LeakyReLU (UCCNet residual block style)."""

    def __init__(self, channels: int = 1, mid: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def channel_residual_pair(i_lambda: torch.Tensor, i_kappa: torch.Tensor) -> torch.Tensor:
    """(I_k - I_l) * (1 - I_l), clipped to valid RGB range."""
    i_lambda = i_lambda.clamp(0.0, 1.0)
    i_kappa = i_kappa.clamp(0.0, 1.0)
    return (i_kappa - i_lambda) * (1.0 - i_lambda)


class UCCRGBranchGate(nn.Module):
    """Pixel-wise scalar gate (UCCNet WPMap style) for one R/G/B branch."""

    BRANCHES = ("r", "g", "b")

    def __init__(self, branch: str, feat_channels: int = 256, f_mid: int = 8, gate_scale: float = 1.0):
        super().__init__()
        if branch not in self.BRANCHES:
            raise ValueError("branch must be r|g|b, got {}".format(branch))
        self.branch = branch
        self.gate_scale = float(gate_scale)
        self.f0 = LightResidualF(1, f_mid)
        self.f1 = LightResidualF(1, f_mid)
        # Scalar spatial gate (like UCCNet W_lk), not 256-ch — saves ~256x activation memory.
        self.proj = nn.Conv2d(2, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def _residual_maps(self, rgb: torch.Tensor):
        r = rgb[:, 0:1]
        g = rgb[:, 1:2]
        b = rgb[:, 2:3]
        if self.branch == "r":
            return channel_residual_pair(r, g), channel_residual_pair(r, b)
        if self.branch == "g":
            return channel_residual_pair(g, r), channel_residual_pair(g, b)
        return channel_residual_pair(b, r), channel_residual_pair(b, g)

    def forward(self, x_inc: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """Return spatial gate tensor (B,1,H,W); identity when proj=0 or gate_scale=0."""
        if self.gate_scale == 0.0:
            return torch.ones(
                x_inc.shape[0], 1, x_inc.shape[2], x_inc.shape[3],
                device=x_inc.device, dtype=x_inc.dtype,
            )
        d0, d1 = self._residual_maps(rgb)
        if d0.shape[-2:] != x_inc.shape[-2:]:
            size = x_inc.shape[-2:]
            d0 = F.interpolate(d0, size=size, mode="bilinear", align_corners=False)
            d1 = F.interpolate(d1, size=size, mode="bilinear", align_corners=False)
        g0 = self.f0(d0)
        g1 = self.f1(d1)
        gamma = torch.cat([g0, g1], dim=1)
        return 1.0 + self.gate_scale * torch.tanh(self.proj(gamma))


class UCCRGCAM(nn.Module):
    """Inc -> CAM channel weights -> UCCRG spatial gate (same order as PG-CAM)."""

    def __init__(self, in_channels: int, reduction_ratio: int, branch: str, f_mid: int = 8, gate_scale: float = 1.0):
        super().__init__()
        self.uccrg_gate = UCCRGBranchGate(branch, in_channels, f_mid=f_mid, gate_scale=gate_scale)
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.Softsign(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Softsign(),
        )

    def forward(self, x_inc: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        w = self.module(x_inc).unsqueeze(2).unsqueeze(3)
        x_ch = x_inc * w
        gate = self.uccrg_gate(x_inc, rgb)
        return x_ch * gate
