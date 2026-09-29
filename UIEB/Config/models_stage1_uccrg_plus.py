# -*- coding: utf-8 -*-
"""CWM-Net: Stage-1 UCCRG + AGFM."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_UIEB = Path(__file__).resolve().parents[1]
_ROOT = _UIEB.parent
for _p in (_UIEB, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Config.models import Inc  # noqa: E402
from Config.uccrg import UCCRGCAM  # noqa: E402
from Config.pg_cam import CAM  # noqa: E402
from uccp import UCCPFilm  # noqa: E402


class DICAM_UCCP_Stage1UCCRG(nn.Module):
    """Stage-1 UCCRG + fusion AGFM."""

    def __init__(self, f_mid: int = 8):
        super().__init__()
        self.layer_1_r = Inc(in_channels=1, filters=64)
        self.layer_1_g = Inc(in_channels=1, filters=64)
        self.layer_1_b = Inc(in_channels=1, filters=64)

        self.layer_2_r = UCCRGCAM(256, 4, "r", f_mid=f_mid)
        self.layer_2_g = UCCRGCAM(256, 4, "g", f_mid=f_mid)
        self.layer_2_b = UCCRGCAM(256, 4, "b", f_mid=f_mid)

        self.uccp = UCCPFilm(768)
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
        layer_concat = self.uccp(layer_concat, input)

        layer_3 = self.layer_3(layer_concat)
        layer_4 = self.layer_4(layer_3)
        return self.layer_tail(layer_4)


def load_stage1uccrg_into_uccp(
    model: DICAM_UCCP_Stage1UCCRG,
    ckpt_path: str,
    device: torch.device,
):
    """Load Stage-1 UCCRG weights; AGFM (uccp.*) stays at identity init."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError("Unexpected keys: {}".format(unexpected[:8]))
    bad = [k for k in missing if not k.startswith("uccp.")]
    if bad:
        raise RuntimeError("Missing keys: {}".format(bad[:8]))
    return missing, unexpected
