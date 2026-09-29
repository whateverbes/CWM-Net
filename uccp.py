# -*- coding: utf-8 -*-
"""UCCP FiLM (AGFM) on concatenated features."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def water_appearance_feats(img: torch.Tensor) -> torch.Tensor:
    """B,11 interpretable global stats."""
    rgb_m = img.mean(dim=[2, 3])
    rgb_s = img.std(dim=[2, 3]).clamp_min(1e-6)
    r, g, b = rgb_m[:, 0:1], rgb_m[:, 1:2], rgb_m[:, 2:3]
    blue_cast = b - g
    green_cast = g - r
    red_def = torch.clamp(0.5 * (g + b) - r, 0.0, 1.0)
    contrast = rgb_s.mean(dim=1, keepdim=True)
    mx = rgb_m.max(dim=1, keepdim=True).values
    mn = rgb_m.min(dim=1, keepdim=True).values
    sat = (mx - mn) / mx.clamp_min(1e-6)
    return torch.cat(
        [rgb_m, rgb_s, blue_cast, green_cast, red_def, contrast, sat], dim=1
    )


class UCCPFilm(nn.Module):
    """FiLM on concat features; identity at init."""

    def __init__(self, channels: int = 768, feat_dim: int = 11, mid: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(feat_dim, mid),
            nn.GELU(),
            nn.Linear(mid, mid),
            nn.GELU(),
        )
        self.gamma = nn.Linear(mid, channels)
        self.beta = nn.Linear(mid, channels)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, feat: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        h = self.enc(water_appearance_feats(img))
        g = 1.0 + self.gamma(h).view(feat.shape[0], -1, 1, 1)
        b = self.beta(h).view(feat.shape[0], -1, 1, 1)
        return feat * g + b
