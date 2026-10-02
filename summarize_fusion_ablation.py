#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print fusion-ablation best scores from ckpts/ablation/*/train_log.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

VARIANTS = ["se", "cbam", "film_uncond", "film_randinit", "wam"]


def read_best(run_dir: Path) -> dict:
    log_p = run_dir / "train_log.json"
    if log_p.is_file():
        obj = json.loads(log_p.read_text(encoding="utf-8"))
        best = obj.get("best", {})
        return {
            "psnr": best.get("psnr"),
            "ssim": best.get("ssim"),
            "epoch": best.get("epoch"),
            "last_epoch": max((e.get("epoch", 0) for e in obj.get("log", [])), default=0),
        }
    return {"psnr": None, "ssim": None, "epoch": None, "last_epoch": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", default="ckpts/ablation")
    args = ap.parse_args()
    root = Path(args.ckpt_root)
    if not root.is_absolute():
        root = Path(__file__).resolve().parent / root
    print("| variant | PSNR | SSIM | best_ep | last_ep |")
    print("|---------|------|------|---------|---------|")
    for name in VARIANTS:
        r = read_best(root / name)
        ps = "{:.4f}".format(r["psnr"]) if r["psnr"] not in (None, -1.0) else "—"
        ss = "{:.4f}".format(r["ssim"]) if r["ssim"] not in (None, -1.0) else "—"
        print("| {} | {} | {} | {} | {} |".format(name, ps, ss, r["epoch"], r["last_epoch"]))


if __name__ == "__main__":
    main()
