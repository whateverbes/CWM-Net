# -*- coding: utf-8 -*-
"""Physics-Guided Channel-Spatial Attention (PG-CAM) for Stage-1 branches."""
from __future__ import annotations

import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return inp.view(inp.size(0), -1)


class CAM(nn.Module):
    """Channel attention used after each RGB branch."""

    def __init__(self, in_channels: int, reduction_ratio: int):
        super().__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.Softsign(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Softsign(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.module(x).unsqueeze(2).unsqueeze(3)
        return x * w


class PGCAM(nn.Module):
    """Channel attention (same weights as CAM) + physics-map spatial gate.

    X_out = (X_inc ⊙ a) ⊙ (1 + tanh(Conv1x1(M)))
    Conv1x1 is zero-init → spatial gate starts at 1 (identity).
    Submodule name `module` matches CAM keys for checkpoint loading.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 4):
        super().__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.Softsign(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Softsign(),
        )
        self.spatial_conv = nn.Conv2d(1, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.spatial_conv.weight)
        nn.init.zeros_(self.spatial_conv.bias)

    def forward(self, x_inc: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        if m.shape[-2:] != x_inc.shape[-2:]:
            m = torch.nn.functional.interpolate(
                m, size=x_inc.shape[-2:], mode="bilinear", align_corners=False
            )
        w = self.module(x_inc).unsqueeze(2).unsqueeze(3)
        x_ch = x_inc * w
        gate = 1.0 + torch.tanh(self.spatial_conv(m))
        return x_ch * gate
