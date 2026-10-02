#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train fusion substitutes at concat (paper Table 4). Backbone frozen; only fuse.* is trained.

Default eval_every=0 (no mid-training test). Loss: L1 + (1-SSIM) + VGG19 relu4_3.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

try:
    from pytorch_msssim import ssim as ssim_fn
except ImportError as e:
    raise ImportError("pip install pytorch-msssim") from e

ROOT = Path(__file__).resolve().parents[1]
UIEB = Path(__file__).resolve().parent
sys.path.insert(0, str(UIEB))
sys.path.insert(0, str(ROOT))

from models.fusion_ablation import FusionAblation, load_base_into_ablation  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_backbone_eval(model: nn.Module) -> int:
    n = 0
    for name, p in model.named_parameters():
        train_it = name.startswith("fuse.")
        p.requires_grad = train_it
        if train_it:
            n += p.numel()
    model.stage1_inference = True
    model.checkpoint_stage2 = True
    model.eval()
    model.fuse.train()
    return n


def model_forward_train(model: FusionAblation, hazy: torch.Tensor) -> torch.Tensor:
    if getattr(model, "stage1_inference", False) and model.fuse_pos == "concat":
        with torch.no_grad():
            feat = model.encode_stage1(hazy)
        feat = feat.detach().contiguous()
        return model.decode_from_concat(feat, hazy)
    return model(hazy)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hazydir", default="../Data/UIEB/hazy_train/")
    ap.add_argument("--cleandir", default="../Data/UIEB/clean_train/")
    ap.add_argument("--testing_dir_inp", default="../Data/UIEB/hazy_test/")
    ap.add_argument("--testing_dir_gt", default="../Data/UIEB/clean_test/")
    ap.add_argument("--init_from_base", default="../ckpts/UIEB/CWMNetBase_60.pt")
    ap.add_argument("--out_dir", default="../ckpts/ablation/wam")
    ap.add_argument("--variant", default="wam", choices=["wam", "se", "cbam", "film_uncond", "film_randinit"])
    ap.add_argument("--mid", type=int, default=32)
    ap.add_argument("--fuse_pos", default="concat", choices=["concat", "after_inc", "after_cam"])
    ap.add_argument("--zero_init", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eval_every", type=int, default=0)
    ap.add_argument("--lambda_l1", type=float, default=1.0)
    ap.add_argument("--lambda_ssim", type=float, default=1.0)
    ap.add_argument("--lambda_per", type=float, default=1.0)
    ap.add_argument("--num_images", type=int, default=800)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import Config.dataset as dataset  # noqa: E402
    from Config.vgg import Vgg19, normalize_batch  # noqa: E402

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    init_path = Path(args.init_from_base)
    if not init_path.is_absolute():
        init_path = (UIEB / init_path).resolve()

    vgg = None
    if args.lambda_per > 0:
        vgg = Vgg19(requires_grad=False).to(device)
        vgg.eval()

    model = FusionAblation(
        variant=args.variant,
        mid=args.mid,
        zero_init=bool(args.zero_init) and args.variant != "film_randinit",
        fuse_pos=args.fuse_pos,
    ).to(device)
    load_base_into_ablation(model, str(init_path), device)
    n_train = freeze_backbone_eval(model)
    print("[Fusion] variant={} mid={} pos={} trainable={}".format(
        args.variant, args.mid, args.fuse_pos, n_train))

    mae = nn.L1Loss()
    optim_g = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-5,
    )

    hazy_dir = Path(args.hazydir)
    clean_dir = Path(args.cleandir)
    if not hazy_dir.is_absolute():
        hazy_dir = (UIEB / hazy_dir).resolve()
        clean_dir = (UIEB / clean_dir).resolve()
    ds = dataset.Dataset_Load(str(hazy_dir), str(clean_dir), transform=dataset.ToTensor())
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False
    )

    log = []
    best = {"psnr": -1.0, "epoch": None, "ssim": None}

    for epoch in range(1, args.epochs + 1):
        freeze_backbone_eval(model)
        meter = 0.0
        n_batches = 0
        for batch in tqdm(loader, desc="[Fusion {}/{}]".format(epoch, args.epochs), leave=False):
            hazy = batch["hazy"].to(device).float()
            clean = batch["clean"].to(device).float()
            pred = model_forward_train(model, hazy)
            loss = args.lambda_l1 * mae(pred, clean)
            loss = loss + args.lambda_ssim * (1.0 - ssim_fn(pred, clean, data_range=1.0, size_average=True))
            if vgg is not None:
                with torch.no_grad():
                    clean_f = vgg(normalize_batch(clean))
                pred_f = vgg(normalize_batch(pred))
                loss = loss + args.lambda_per * mae(pred_f.relu4_3, clean_f.relu4_3)
            loss.backward()
            optim_g.step()
            optim_g.zero_grad(set_to_none=True)
            meter += float(loss.item())
            n_batches += 1
        mean_loss = meter / max(n_batches, 1)
        print("[ep {:02d}] loss={:.4f}".format(epoch, mean_loss))
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim_g.state_dict(),
            "variant": args.variant,
            "train_loss": mean_loss,
        }
        torch.save(payload, out_dir / "ablation_last.pt")
        torch.save(payload, out_dir / "CWMNetFusion_{}_{}.pt".format(args.variant, epoch))
        entry = {"epoch": epoch, "train_loss": mean_loss}
        if args.eval_every > 0 and (epoch % args.eval_every == 0 or epoch == args.epochs):
            from eval_simple import eval_paired_dirs

            inp = Path(args.testing_dir_inp)
            gt = Path(args.testing_dir_gt)
            if not inp.is_absolute():
                inp = (UIEB / inp).resolve()
                gt = (UIEB / gt).resolve()
            model.eval()
            with torch.no_grad():
                m = eval_paired_dirs(model, inp, gt, device)
            freeze_backbone_eval(model)
            entry["test_psnr_paper"] = m["psnr_paper"]
            entry["test_ssim_paper"] = m["ssim_paper"]
            print("  test PSNR={:.4f} SSIM={:.4f}".format(m["psnr_paper"], m["ssim_paper"]))
            if m["psnr_paper"] > best["psnr"]:
                best = {"psnr": m["psnr_paper"], "ssim": m["ssim_paper"], "epoch": epoch}
                torch.save(payload, out_dir / "best_test.pt")
        log.append(entry)
        (out_dir / "train_log.json").write_text(
            json.dumps({"best": best, "args": vars(args), "log": log}, indent=2),
            encoding="utf-8",
        )

    print("Done. best=", best, "dir=", out_dir)


if __name__ == "__main__":
    main()
