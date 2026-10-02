#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leave-one-dimension-out ablation of the 11-D water appearance vector (inference only)."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
UIEB = ROOT / "UIEB"
sys.path.insert(0, str(UIEB))
sys.path.insert(0, str(ROOT))

from models.cwmnet import CWMNet, remap_state_dict
from models.wam import WAM, water_appearance_feats
from eval_simple import eval_paired_dirs

DIM_NAMES = [
    "mu_R", "mu_G", "mu_B",
    "sigma_R", "sigma_G", "sigma_B",
    "c_b", "c_g", "d_r",
    "kappa", "rho",
]


class WAMDimMask(WAM):
    def __init__(self, drop_idx: Optional[int] = None, keep_only: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.drop_idx = drop_idx
        self.keep_only = keep_only

    def forward(self, feat: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        f = water_appearance_feats(img)
        if self.drop_idx is not None:
            f = f.clone()
            f[:, self.drop_idx] = 0.0
        elif self.keep_only is not None:
            mask = torch.zeros_like(f)
            mask[:, self.keep_only] = 1.0
            f = f * mask
        h = self.enc(f)
        g = 1.0 + self.gamma(h).view(feat.shape[0], -1, 1, 1)
        b = self.beta(h).view(feat.shape[0], -1, 1, 1)
        return feat * g + b


def load_model(ckpt: Path, device: torch.device) -> CWMNet:
    obj = torch.load(str(ckpt), map_location=device, weights_only=False)
    state = remap_state_dict(obj.get("model_state_dict", obj.get("model", obj)))
    model = CWMNet().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def swap(model: CWMNet, mod: nn.Module) -> CWMNet:
    m = copy.deepcopy(model)
    new = mod.to(next(m.parameters()).device)
    new.load_state_dict(m.wam.state_dict(), strict=False)
    m.wam = new
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ckpts/CWM-Net/CWMNet.pt")
    ap.add_argument("--hazy_dir", default="Data/UIEB/hazy_test")
    ap.add_argument("--clean_dir", default="Data/UIEB/clean_test")
    ap.add_argument("--out", default="ckpts/ablation/eval_wam_per_dim.json")
    ap.add_argument("--mode", default="drop", choices=["drop", "keep_only", "both"])
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    hazy = Path(args.hazy_dir)
    clean = Path(args.clean_dir)
    if not hazy.is_absolute():
        hazy = ROOT / hazy
        clean = ROOT / clean
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = load_model(ckpt, device)
    report = {"checkpoint": str(ckpt), "leave_one_out": {}, "keep_one": {}}

    with torch.no_grad():
        full = eval_paired_dirs(base, hazy, clean, device)
        report["full"] = full
        print("[full] PSNR={:.4f}".format(full["psnr_paper"]))
        if args.mode in ("drop", "both"):
            for i, name in enumerate(DIM_NAMES):
                m = swap(base, WAMDimMask(drop_idx=i))
                metrics = eval_paired_dirs(m, hazy, clean, device)
                row = {
                    "dim": i,
                    "name": name,
                    "psnr_paper": metrics["psnr_paper"],
                    "ssim_paper": metrics["ssim_paper"],
                    "delta_psnr": metrics["psnr_paper"] - full["psnr_paper"],
                }
                report["leave_one_out"][name] = row
                print("[drop {:8s}] PSNR={:.4f} ({:+.3f})".format(name, row["psnr_paper"], row["delta_psnr"]))
        if args.mode in ("keep_only", "both"):
            for i, name in enumerate(DIM_NAMES):
                m = swap(base, WAMDimMask(keep_only=i))
                metrics = eval_paired_dirs(m, hazy, clean, device)
                row = {
                    "dim": i,
                    "name": name,
                    "psnr_paper": metrics["psnr_paper"],
                    "ssim_paper": metrics["ssim_paper"],
                    "delta_psnr": metrics["psnr_paper"] - full["psnr_paper"],
                }
                report["keep_one"][name] = row
                print("[keep {:8s}] PSNR={:.4f} ({:+.3f})".format(name, row["psnr_paper"], row["delta_psnr"]))

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved:", out)


if __name__ == "__main__":
    main()
