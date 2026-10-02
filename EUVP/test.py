# -*- coding: utf-8 -*-
"""EUVP test: enhance Inp/ images and write to result_dir."""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

_EUVP = Path(__file__).resolve().parent
_ROOT = _EUVP.parent
_UIEB = _ROOT / "UIEB"
for _p in (_EUVP, _UIEB, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from models.cwmnet import CWMNet, CWMNetBase, CWMNetCCRM, CWMNetWAM, remap_state_dict

device = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = {"cwmnet": CWMNet, "base": CWMNetBase, "ccrm": CWMNetCCRM, "wam": CWMNetWAM}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="../ckpts/CWM-Net/CWMNet.pt")
    ap.add_argument("--model", default="cwmnet", choices=sorted(MODELS.keys()))
    ap.add_argument("--inp_dir", default="../Data/EUVP/test_samples/Inp/")
    ap.add_argument("--result_dir", default="../results/EUVP_CWM-Net/")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = (_EUVP / ckpt_path).resolve()
    inp_dir = Path(args.inp_dir)
    if not inp_dir.is_absolute():
        inp_dir = (_EUVP / inp_dir).resolve()
    result_dir = Path(args.result_dir)
    if not result_dir.is_absolute():
        result_dir = (_EUVP / result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    network = MODELS[args.model]()
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    network.load_state_dict(remap_state_dict(state), strict=True)
    network.eval().to(device)

    files = [m for m in os.listdir(inp_dir) if m.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    with tqdm(total=len(files)) as t:
        for name in files:
            img = cv2.imread(str(inp_dir / name))
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)[:, :, ::-1]
            x = torch.from_numpy(np.float32(img).transpose(2, 0, 1)[None] / 255.0).to(device)
            y = network(x).clamp(0, 1)[0].detach().cpu().numpy().transpose(1, 2, 0)
            out = (y * 255.0).astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(str(result_dir / name), out)
            t.update(1)
    print("Saved:", result_dir)


if __name__ == "__main__":
    main()
