# -*- coding: utf-8 -*-
"""Folder-wise PSNR / SSIM on paired hazy/clean test images."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_msssim import ssim as ssim_fn


def _read_rgb(path: Path, size: int) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError("failed to read {}".format(path))
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    im = im[:, :, ::-1].astype(np.float32) / 255.0
    return im


def eval_paired_dirs(model, hazy_dir, clean_dir, device, size=256):
    hazy_dir = Path(hazy_dir)
    clean_dir = Path(clean_dir)
    names = [
        n for n in os.listdir(hazy_dir)
        if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]
    if not names:
        raise FileNotFoundError("no images in {}".format(hazy_dir))
    model.eval()
    psnrs, ssims = [], []
    with torch.no_grad():
        for name in names:
            hp = hazy_dir / name
            cp = clean_dir / name
            if not cp.is_file():
                cands = list(clean_dir.glob(Path(name).stem + ".*"))
                if not cands:
                    continue
                cp = cands[0]
            hazy = _read_rgb(hp, size)
            clean = _read_rgb(cp, size)
            x = torch.from_numpy(hazy.transpose(2, 0, 1)[None]).to(device)
            y = model(x).clamp(0.0, 1.0)
            gt = torch.from_numpy(clean.transpose(2, 0, 1)[None]).to(device)
            mse = float(torch.mean((y - gt) ** 2).item())
            psnrs.append(10.0 * np.log10(1.0 / max(mse, 1e-12)))
            ssims.append(float(ssim_fn(y, gt, data_range=1.0, size_average=True).item()))
    model.train()
    return {
        "psnr_paper": float(np.mean(psnrs)),
        "ssim_paper": float(np.mean(ssims)),
        "n": len(psnrs),
    }
