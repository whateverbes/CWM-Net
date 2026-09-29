# -*- coding: utf-8 -*-
"""Stage-1 UCCRG (no AGFM)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_UIEB = Path(__file__).resolve().parents[1]
if str(_UIEB) not in sys.path:
    sys.path.insert(0, str(_UIEB))

from Config.models import Inc  # noqa: E402
from Config.pg_cam import CAM  # noqa: E402
from Config.uccrg import UCCRGCAM  # noqa: E402


class DICAM_Stage1UCCRG(nn.Module):
    """Stage-1: Inc -> CAM -> UCCRG gate per branch; Stage-2 unchanged."""

    def __init__(self, f_mid: int = 8, gate_scale: float = 1.0):
        super().__init__()
        self.layer_1_r = Inc(in_channels=1, filters=64)
        self.layer_1_g = Inc(in_channels=1, filters=64)
        self.layer_1_b = Inc(in_channels=1, filters=64)

        self.layer_2_r = UCCRGCAM(256, 4, "r", f_mid=f_mid, gate_scale=gate_scale)
        self.layer_2_g = UCCRGCAM(256, 4, "g", f_mid=f_mid, gate_scale=gate_scale)
        self.layer_2_b = UCCRGCAM(256, 4, "b", f_mid=f_mid, gate_scale=gate_scale)

        self.layer_3 = Inc(768, 64)
        self.layer_4 = CAM(256, 4)
        self.layer_tail = nn.Sequential(
            nn.Conv2d(256, 24, 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(24, 3, 1, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_r = input[:, 0:1]
        input_g = input[:, 1:2]
        input_b = input[:, 2:3]

        layer_2_r = self.layer_2_r(self.layer_1_r(input_r), input)
        layer_2_g = self.layer_2_g(self.layer_1_g(input_g), input)
        layer_2_b = self.layer_2_b(self.layer_1_b(input_b), input)

        layer_concat = torch.cat([layer_2_r, layer_2_g, layer_2_b], dim=1)
        layer_3 = self.layer_3(layer_concat)
        layer_4 = self.layer_4(layer_3)
        return self.layer_tail(layer_4)


def load_dicam_into_stage1_uccrg(model, ckpt_path: str, device):
    from Config.stage1_load_dicam import load_dicam_into_stage1

    return load_dicam_into_stage1(model, ckpt_path, device, tag="Stage1UCCRG")
