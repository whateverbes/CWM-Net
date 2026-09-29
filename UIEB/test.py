# -*- coding: utf-8 -*-
"""CWM-Net test: save enhanced images (256x256)."""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

_UIEB = Path(__file__).resolve().parent
_ROOT = _UIEB.parent
for _p in (_UIEB, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Config.models_stage1_uccrg_plus import DICAM_UCCP_Stage1UCCRG
from Config.options import opt

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default="../ckpts/CWM-Net/DICAM_UCCP_Stage1UCCRG_4.pt",
    )
    ap.add_argument("--inp_dir", default="")
    ap.add_argument("--result_dir", default="../results/UIEB_CWM-Net/")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = (_UIEB / ckpt_path).resolve()
    inp_dir = args.inp_dir or opt.testing_dir_inp
    result_dir = args.result_dir
    if not Path(result_dir).is_absolute():
        result_dir = str((_UIEB / result_dir).resolve())
    os.makedirs(result_dir, exist_ok=True)

    network = DICAM_UCCP_Stage1UCCRG()
    checkpoint = torch.load(str(ckpt_path), map_location=torch.device("cpu"), weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    network.load_state_dict(state)
    network.eval()
    network.to(device)

    total_files = [m for m in os.listdir(inp_dir) if m.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    st = time.time()
    ch = 3
    with tqdm(total=len(total_files)) as t:
        for m in total_files:
            img = cv2.imread(os.path.join(inp_dir, str(m)))
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            img = img[:, :, ::-1]
            img = np.float32(img) / 255.0
            h, w, c = img.shape
            train_x = np.zeros((1, ch, h, w)).astype(np.float32)
            train_x[0, 0, :, :] = img[:, :, 0]
            train_x[0, 1, :, :] = img[:, :, 1]
            train_x[0, 2, :, :] = img[:, :, 2]
            dataset_torchx = torch.from_numpy(train_x).to(device)
            output = network(dataset_torchx)
            output = (output.clamp_(0.0, 1.0)[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            output = output[:, :, ::-1]
            cv2.imwrite(os.path.join(result_dir, str(m)), output)
            t.set_postfix_str("name: {} | old [hw]: {}/{} | new [hw]: {}/{}".format(
                str(m), h, w, output.shape[0], output.shape[1]
            ))
            t.update(1)
    end = time.time()
    print("Total time taken in secs : " + str(end - st))
    print("Per image (avg): " + str(float((end - st) / max(len(total_files), 1))))
    print("Saved:", result_dir)


if __name__ == "__main__":
    main()
