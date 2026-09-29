# -*- coding: utf-8 -*-
"""Load DICAM checkpoint into Stage-1 models (shared backbone keys)."""
from __future__ import annotations

import torch


def _load_state(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    return ckpt.get("model_state_dict", ckpt.get("model", ckpt))


def _is_new_module_key(name: str) -> bool:
    return any(
        s in name
        for s in ("uccrg_gate", ".prompt.", "spatial_conv", "prompt_bank", "uccp.")
    )


def load_dicam_into_stage1(model, ckpt_path: str, device: torch.device, tag: str = "Stage1"):
    """Copy all matching keys from DICAM_* ckpt; new module keys stay at init."""
    state = _load_state(ckpt_path, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError("[{}] unexpected keys from DICAM ckpt: {}".format(tag, unexpected[:8]))
    new_keys = [k for k in missing if _is_new_module_key(k)]
    backbone_missing = [k for k in missing if k not in new_keys]
    if backbone_missing:
        raise RuntimeError("[{}] missing backbone keys: {}".format(tag, backbone_missing[:8]))
    print("[{}] loaded DICAM from {}".format(tag, ckpt_path))
    if new_keys:
        print("[{}] new module keys kept at init ({}): {}".format(tag, len(new_keys), new_keys[:6]))
    return new_keys
