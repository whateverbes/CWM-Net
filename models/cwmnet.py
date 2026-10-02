# -*- coding: utf-8 -*-
"""CWM-Net models: Base / CCRM / full CWM-Net."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
_UIEB = _ROOT / "UIEB"
for _p in (_UIEB, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Config.models import CAM, Inc, CWMNetBase  # noqa: E402
from Config.uccrg import UCCRGCAM  # noqa: E402
from models.wam import WAM  # noqa: E402

__all__ = [
    "CWMNetBase",
    "CWMNetCCRM",
    "CWMNetWAM",
    "CWMNet",
    "remap_state_dict",
    "load_base_into_ccrm",
    "load_base_into_wam",
    "load_ccrm_into_cwmnet",
    "load_partial_into_cwmnet",
]


def remap_state_dict(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        k = k.replace("uccp.", "wam.")
        out[k] = v
    return out


def _load_state(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    return remap_state_dict(state)


class CWMNetWAM(nn.Module):
    """CWMNetBase + WAM (no UCCRG). Used for Table 7 in-domain EUVP."""

    def __init__(self):
        super().__init__()
        self.layer_1_r = Inc(in_channels=1, filters=64)
        self.layer_1_g = Inc(in_channels=1, filters=64)
        self.layer_1_b = Inc(in_channels=1, filters=64)
        self.layer_2_r = CAM(256, 4)
        self.layer_2_g = CAM(256, 4)
        self.layer_2_b = CAM(256, 4)
        self.wam = WAM(768)
        self.layer_3 = Inc(768, 64)
        self.layer_4 = CAM(256, 4)
        self.layer_tail = nn.Sequential(
            nn.Conv2d(256, 24, 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(24, 3, 1, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        layer_2_r = self.layer_2_r(self.layer_1_r(input[:, 0:1]))
        layer_2_g = self.layer_2_g(self.layer_1_g(input[:, 1:2]))
        layer_2_b = self.layer_2_b(self.layer_1_b(input[:, 2:3]))
        layer_concat = torch.cat([layer_2_r, layer_2_g, layer_2_b], dim=1)
        layer_concat = self.wam(layer_concat, input)
        return self.layer_tail(self.layer_4(self.layer_3(layer_concat)))


class CWMNetCCRM(nn.Module):
    """CCRM with UCCRG, without WAM."""

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
        layer_2_r = self.layer_2_r(self.layer_1_r(input[:, 0:1]), input)
        layer_2_g = self.layer_2_g(self.layer_1_g(input[:, 1:2]), input)
        layer_2_b = self.layer_2_b(self.layer_1_b(input[:, 2:3]), input)
        layer_concat = torch.cat([layer_2_r, layer_2_g, layer_2_b], dim=1)
        return self.layer_tail(self.layer_4(self.layer_3(layer_concat)))


class CWMNet(nn.Module):
    """Full CWM-Net: CCRM + WAM + color correction."""

    def __init__(self, f_mid: int = 8):
        super().__init__()
        self.layer_1_r = Inc(in_channels=1, filters=64)
        self.layer_1_g = Inc(in_channels=1, filters=64)
        self.layer_1_b = Inc(in_channels=1, filters=64)
        self.layer_2_r = UCCRGCAM(256, 4, "r", f_mid=f_mid)
        self.layer_2_g = UCCRGCAM(256, 4, "g", f_mid=f_mid)
        self.layer_2_b = UCCRGCAM(256, 4, "b", f_mid=f_mid)
        self.wam = WAM(768)
        self.layer_3 = Inc(768, 64)
        self.layer_4 = CAM(256, 4)
        self.layer_tail = nn.Sequential(
            nn.Conv2d(256, 24, 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(24, 3, 1, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        layer_2_r = self.layer_2_r(self.layer_1_r(input[:, 0:1]), input)
        layer_2_g = self.layer_2_g(self.layer_1_g(input[:, 1:2]), input)
        layer_2_b = self.layer_2_b(self.layer_1_b(input[:, 2:3]), input)
        layer_concat = torch.cat([layer_2_r, layer_2_g, layer_2_b], dim=1)
        layer_concat = self.wam(layer_concat, input)
        return self.layer_tail(self.layer_4(self.layer_3(layer_concat)))


def load_base_into_ccrm(model, ckpt_path: str, device):
    state = _load_state(ckpt_path, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError("Unexpected keys: {}".format(unexpected[:8]))
    new_keys = [k for k in missing if "uccrg_gate" in k or k.startswith("wam.")]
    backbone_missing = [k for k in missing if k not in new_keys]
    if backbone_missing:
        raise RuntimeError("Missing backbone keys: {}".format(backbone_missing[:8]))
    print("[CWMNetCCRM] loaded base from", ckpt_path)
    return new_keys


def load_base_into_wam(model: CWMNetWAM, ckpt_path: str, device):
    state = _load_state(ckpt_path, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError("Unexpected keys: {}".format(unexpected[:8]))
    bad = [k for k in missing if not k.startswith("wam.")]
    if bad:
        raise RuntimeError("Missing keys: {}".format(bad[:8]))
    print("[CWMNetWAM] loaded base from", ckpt_path)
    return missing, unexpected


def load_ccrm_into_cwmnet(model: CWMNet, ckpt_path: str, device):
    state = _load_state(ckpt_path, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError("Unexpected keys: {}".format(unexpected[:8]))
    bad = [k for k in missing if not k.startswith("wam.")]
    if bad:
        raise RuntimeError("Missing keys: {}".format(bad[:8]))
    return missing, unexpected


def load_partial_into_cwmnet(model: CWMNet, ckpt_path: str, device, tag: str = "CWM-Net"):
    state = _load_state(ckpt_path, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError("[{}] unexpected keys: {}".format(tag, unexpected[:8]))
    new_keys = [k for k in missing if ("uccrg_gate" in k or k.startswith("wam."))]
    backbone_missing = [k for k in missing if k not in new_keys]
    if backbone_missing:
        raise RuntimeError("[{}] missing backbone keys: {}".format(tag, backbone_missing[:8]))
    print("[{}] loaded from {}".format(tag, ckpt_path))
    return new_keys
