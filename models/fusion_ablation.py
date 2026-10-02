# -*- coding: utf-8 -*-
"""CWMNetBase + pluggable fusion at concat (paper Table 4)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

_ROOT = Path(__file__).resolve().parents[1]
_UIEB = _ROOT / "UIEB"
for _p in (_UIEB, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Config.models import CAM, Inc  # noqa: E402
from models.fusion import build_fusion_module  # noqa: E402
from models.cwmnet import remap_state_dict  # noqa: E402


def _ckpt(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if not torch.is_grad_enabled():
        return module(x)
    try:
        return checkpoint(module, x, use_reentrant=False)
    except TypeError:
        return checkpoint(module, x)


class FusionAblation(nn.Module):
    def __init__(self, variant: str = "wam", mid: int = 32, zero_init: bool = True, fuse_pos: str = "concat"):
        super().__init__()
        self.fuse_pos = fuse_pos
        self.variant = variant
        self.stage1_inference = False
        self.checkpoint_stage2 = False
        self.layer_1_r = Inc(in_channels=1, filters=64)
        self.layer_1_g = Inc(in_channels=1, filters=64)
        self.layer_1_b = Inc(in_channels=1, filters=64)
        self.layer_2_r = CAM(256, 4)
        self.layer_2_g = CAM(256, 4)
        self.layer_2_b = CAM(256, 4)
        fuse_ch = 768 if fuse_pos == "concat" else 256
        self.fuse = build_fusion_module(variant, channels=fuse_ch, mid=mid, zero_init=zero_init)
        self.layer_3 = Inc(768, 64)
        self.layer_4 = CAM(256, 4)
        self.layer_tail = nn.Sequential(
            nn.Conv2d(256, 24, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(24, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def encode_stage1(self, input: torch.Tensor) -> torch.Tensor:
        layer_2_r = self.layer_2_r(self.layer_1_r(input[:, 0:1]))
        layer_2_g = self.layer_2_g(self.layer_1_g(input[:, 1:2]))
        layer_2_b = self.layer_2_b(self.layer_1_b(input[:, 2:3]))
        return torch.cat([layer_2_r, layer_2_g, layer_2_b], dim=1)

    def decode_from_concat(self, layer_concat: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        if self.fuse_pos == "concat":
            layer_concat = self.fuse(layer_concat, img)
        layer_3 = _ckpt(self.layer_3, layer_concat) if self.checkpoint_stage2 else self.layer_3(layer_concat)
        if self.fuse_pos == "after_inc":
            layer_3 = self.fuse(layer_3, img)
        layer_4 = _ckpt(self.layer_4, layer_3) if self.checkpoint_stage2 else self.layer_4(layer_3)
        if self.fuse_pos == "after_cam":
            layer_4 = self.fuse(layer_4, img)
        return self.layer_tail(layer_4)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.decode_from_concat(self.encode_stage1(input), input)


def load_base_into_ablation(model: FusionAblation, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = remap_state_dict(ckpt.get("model_state_dict", ckpt.get("model", ckpt)))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError("Unexpected keys: {}".format(unexpected[:8]))
    bad = [k for k in missing if not k.startswith("fuse.")]
    if bad:
        raise RuntimeError("Missing non-fuse keys: {}".format(bad[:8]))
