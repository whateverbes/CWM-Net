"""EUVP in-domain training (paper Table 7).

Architectures:
  base    -> CWMNetBase
  wam     -> CWMNetWAM  (base + WAM, no UCCRG)
  cwmnet  -> CWM-Net    (CCRM + WAM)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_msssim import ssim
from torch.utils.data import DataLoader

_EUVP = Path(__file__).resolve().parent
_ROOT = _EUVP.parent
_UIEB = _ROOT / "UIEB"
for _p in (_EUVP, _UIEB, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import Config.euvp_dataset as dataset
from Config.euvp_options import device, opt
from Config.vgg import Vgg19, normalize_batch
from models.cwmnet import (
    CWMNet,
    CWMNetBase,
    CWMNetWAM,
    load_base_into_wam,
    load_partial_into_cwmnet,
    remap_state_dict,
)

STEM = {"base": "CWMNetBase", "wam": "CWMNetWAM", "cwmnet": "CWMNet"}
BUILD = {"base": CWMNetBase, "wam": CWMNetWAM, "cwmnet": CWMNet}
ARCH_CKPT = {
    "base": "../ckpts/EUVP_Base/",
    "wam": "../ckpts/EUVP_WAM/",
    "cwmnet": "../ckpts/EUVP_CWM-Net/",
}


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def _latest(stem: str, ckpt_dir: str):
    if not os.path.isdir(ckpt_dir):
        return None
    epochs = []
    prefix = stem + "_"
    for name in os.listdir(ckpt_dir):
        s, ext = os.path.splitext(name)
        if ext == ".pt" and s.startswith(prefix) and s[len(prefix):].isdigit():
            epochs.append(int(s[len(prefix):]))
    if not epochs:
        return None
    return "{}_{}.pt".format(stem, max(epochs))


def _resolve(path_str: str) -> str:
    raw = Path(path_str)
    if raw.is_absolute():
        return str(raw)
    for cand in ((_EUVP / path_str).resolve(), (_ROOT / path_str).resolve()):
        if cand.exists() or cand.parent.exists():
            return str(cand)
    return str((_ROOT / path_str).resolve())


if __name__ == "__main__":
    arch = opt.arch
    stem = STEM[arch]
    model = BUILD[arch]().to(device)
    print("[EUVP] arch={} params={}".format(arch, sum(p.numel() for p in model.parameters())))

    if opt.checkpoints_dir in ("../ckpts/EUVP_CWM-Net/", "../ckpts/EUVP_CWM-Net"):
        opt.checkpoints_dir = ARCH_CKPT[arch]
    opt.checkpoints_dir = _resolve(opt.checkpoints_dir)
    os.makedirs(opt.checkpoints_dir, exist_ok=True)

    if not opt.pretrained and arch == "wam":
        opt.pretrained = "../ckpts/EUVP/CWMNetBase.pt"

    if opt.pretrained:
        pt = _resolve(opt.pretrained)
        if arch == "wam":
            load_base_into_wam(model, pt, device)
        elif arch == "cwmnet":
            load_partial_into_cwmnet(model, pt, device)
        else:
            ckpt = torch.load(pt, map_location=device, weights_only=False)
            state = remap_state_dict(ckpt.get("model_state_dict", ckpt.get("model", ckpt)))
            model.load_state_dict(state, strict=True)
        print("[Init] loaded", pt)

    mae_loss = nn.L1Loss()
    vgg = Vgg19(requires_grad=False).to(device)
    vgg.eval()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    optim_g = optim.Adam(
        model.parameters(),
        lr=opt.learning_rate_g,
        betas=(opt.beta1, opt.beta2),
        weight_decay=opt.wd_g,
    )

    data_path = _resolve(opt.data_path)
    ds = dataset.Dataset_Load(data_path=data_path, transform=dataset.ToTensor())
    batches = max(1, int(ds.len / opt.batch_size))
    loader_kw = dict(
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if opt.num_workers > 0:
        loader_kw["persistent_workers"] = True
    dataloader = DataLoader(ds, **loader_kw)
    print("[Data] images={} batch={} steps/ep={}".format(ds.len, opt.batch_size, batches))

    start_epoch = 1
    if opt.fresh:
        for p in Path(opt.checkpoints_dir).glob("{}_*.pt".format(stem)):
            p.unlink()
    else:
        latest = _latest(stem, opt.checkpoints_dir)
        if latest:
            ckpt = torch.load(os.path.join(opt.checkpoints_dir, latest), map_location=device, weights_only=False)
            start_epoch = ckpt["epoch"] + 1
            model.load_state_dict(remap_state_dict(ckpt["model_state_dict"]))
            optim_g.load_state_dict(ckpt["optimizer_state_dict"])
            print("[Resume]", latest, "start_epoch=", start_epoch)

    model.train()
    for epoch in range(start_epoch, opt.end_epoch + 1):
        opt.total_mae_loss = opt.total_vgg_loss = opt.total_ssim_loss = opt.total_loss = 0.0
        for i_batch, sample in enumerate(dataloader):
            hazy = sample["hazy"].to(device)
            clean = sample["clean"].to(device)
            pred = model(hazy)
            l_mae = torch.mul(opt.lambda_mae, mae_loss(pred, clean))
            l_ssim = 1 - ssim(pred, clean, data_range=1, size_average=True)
            with torch.no_grad():
                clean_vgg = vgg(normalize_batch(clean))
            pred_vgg = vgg(normalize_batch(pred))
            l_vgg = torch.mul(opt.lambda_vgg, mae_loss(pred_vgg.relu4_3, clean_vgg.relu4_3))
            (l_mae + l_ssim + l_vgg).backward()
            optim_g.step()
            optim_g.zero_grad()
            opt.total_mae_loss += l_mae.item()
            opt.total_ssim_loss += l_ssim.item()
            opt.total_vgg_loss += l_vgg.item()
            print(
                "\r Epoch : {} | ({}/{}) | l_mae: {} | l_ssim: {} | l_vgg: {}".format(
                    epoch, i_batch + 1, batches, l_mae.item() / 2, 1 - l_ssim.item(), l_vgg.item()
                ),
                end="",
                flush=True,
            )
        print(
            "\n\nFinished ep. %d, lr = %.6f, mean_mae = %.6f, mean_ssim = %.6f, mean_vgg = %.6f"
            % (
                epoch,
                get_lr(optim_g),
                (opt.total_mae_loss / batches) / 2,
                1 - (opt.total_ssim_loss / batches),
                opt.total_vgg_loss / batches,
            )
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim_g.state_dict(),
            },
            os.path.join(opt.checkpoints_dir, "{}_{}.pt".format(stem, epoch)),
        )
