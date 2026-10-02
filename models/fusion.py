# -*- coding: utf-8 -*-
"""Fusion substitutes at the concat point (paper Table 4): SE / CBAM / unconditional FiLM / WAM."""
from __future__ import annotations

import torch
import torch.nn as nn

from models.wam import WAM, water_appearance_feats


class SEFusion(nn.Module):
    def __init__(self, channels: int = 768, mid: int = 32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=False),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, img: torch.Tensor = None) -> torch.Tensor:
        w = self.fc(feat.mean(dim=[2, 3]))
        return feat * w.view(feat.shape[0], -1, 1, 1)


class CBAMFusion(nn.Module):
    def __init__(self, channels: int = 768, mid: int = 32):
        super().__init__()
        self.channel = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=False),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, img: torch.Tensor = None) -> torch.Tensor:
        w = self.channel(feat.mean(dim=[2, 3])).view(feat.shape[0], -1, 1, 1)
        x = feat * w
        spa = torch.cat([x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True).values], dim=1)
        return x * self.spatial(spa)


class UncondFilm(nn.Module):
    def __init__(self, channels: int = 768, mid: int = 32, zero_init: bool = True):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(1, mid),
            nn.GELU(),
            nn.Linear(mid, mid),
            nn.GELU(),
        )
        self.gamma = nn.Linear(mid, channels)
        self.beta = nn.Linear(mid, channels)
        if zero_init:
            nn.init.zeros_(self.gamma.weight)
            nn.init.zeros_(self.gamma.bias)
            nn.init.zeros_(self.beta.weight)
            nn.init.zeros_(self.beta.bias)

    def forward(self, feat: torch.Tensor, img: torch.Tensor = None) -> torch.Tensor:
        ones = torch.ones(feat.shape[0], 1, device=feat.device, dtype=feat.dtype)
        h = self.enc(ones)
        g = 1.0 + self.gamma(h).view(feat.shape[0], -1, 1, 1)
        b = self.beta(h).view(feat.shape[0], -1, 1, 1)
        return feat * g + b


class WAMFilm(WAM):
    def __init__(self, channels=768, feat_dim=11, mid=32, zero_init=True):
        nn.Module.__init__(self)
        self.enc = nn.Sequential(
            nn.Linear(feat_dim, mid),
            nn.GELU(),
            nn.Linear(mid, mid),
            nn.GELU(),
        )
        self.gamma = nn.Linear(mid, channels)
        self.beta = nn.Linear(mid, channels)
        if zero_init:
            nn.init.zeros_(self.gamma.weight)
            nn.init.zeros_(self.gamma.bias)
            nn.init.zeros_(self.beta.weight)
            nn.init.zeros_(self.beta.bias)
        else:
            nn.init.xavier_uniform_(self.gamma.weight)
            nn.init.zeros_(self.gamma.bias)
            nn.init.xavier_uniform_(self.beta.weight)
            nn.init.zeros_(self.beta.bias)

    def forward(self, feat: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        h = self.enc(water_appearance_feats(img))
        g = 1.0 + self.gamma(h).view(feat.shape[0], -1, 1, 1)
        b = self.beta(h).view(feat.shape[0], -1, 1, 1)
        return feat * g + b


def build_fusion_module(name: str, channels: int = 768, mid: int = 32, zero_init: bool = True) -> nn.Module:
    name = name.lower().strip()
    if name in ("wam", "film"):
        return WAMFilm(channels=channels, mid=mid, zero_init=zero_init)
    if name == "se":
        return SEFusion(channels=channels, mid=mid)
    if name == "cbam":
        return CBAMFusion(channels=channels, mid=mid)
    if name in ("film_uncond", "uncond", "uncond_film"):
        return UncondFilm(channels=channels, mid=mid, zero_init=zero_init)
    if name in ("film_randinit", "randinit"):
        return WAMFilm(channels=channels, mid=mid, zero_init=False)
    raise ValueError("Unknown fusion variant: {}".format(name))
