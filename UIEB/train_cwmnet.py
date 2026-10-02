"""Train WAM on a CCRM checkpoint to obtain full CWM-Net."""
import os
import sys
from pathlib import Path

_UIEB = Path(__file__).resolve().parent
_ROOT = _UIEB.parent
for _p in (_UIEB, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_msssim import ssim
from torch.utils.data import DataLoader

import Config.dataset as dataset
from Config.models_stage1_uccrg_plus import CWMNet, load_ccrm_into_cwmnet
from Config.options import device, opt
from Config.vgg import Vgg19, normalize_batch
from models.cwmnet import remap_state_dict

CKPT_STEM = "CWMNet"


def _set_stage1_frozen(model, frozen):
    for name, p in model.named_parameters():
        if name.startswith(("layer_1_", "layer_2_")):
            p.requires_grad = not frozen


def _set_decoder_frozen(model, frozen):
    for name, p in model.named_parameters():
        if name.startswith(("layer_3", "layer_4", "layer_tail")):
            p.requires_grad = not frozen


def _set_wam_trainable(model, trainable):
    for name, p in model.named_parameters():
        if name.startswith("wam."):
            p.requires_grad = trainable


def _apply_phase(model, epoch, decoder_warmup_epochs):
    if epoch <= decoder_warmup_epochs:
        _set_stage1_frozen(model, True)
        _set_decoder_frozen(model, False)
        _set_wam_trainable(model, False)
        return "decoder"
    _set_stage1_frozen(model, True)
    _set_decoder_frozen(model, True)
    _set_wam_trainable(model, True)
    return "wam"


def _build_optimizer(model):
    wam_lr = opt.learning_rate_g * opt.wam_lr_scale
    backbone = [p for n, p in model.named_parameters() if not n.startswith("wam.") and p.requires_grad]
    wam = [p for n, p in model.named_parameters() if n.startswith("wam.") and p.requires_grad]
    if wam and backbone and opt.wam_lr_scale != 1.0:
        return optim.Adam(
            [{"params": backbone, "lr": opt.learning_rate_g}, {"params": wam, "lr": wam_lr}],
            betas=(opt.beta1, opt.beta2),
            weight_decay=opt.wd_g,
        )
    trainable = [p for p in model.parameters() if p.requires_grad]
    lr = wam_lr if wam and not backbone else opt.learning_rate_g
    return optim.Adam(trainable, lr=lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.wd_g)


def _abs_ckpt_path(d, epoch):
    return os.path.abspath(os.path.join(d, "{}_{}.pt".format(CKPT_STEM, epoch)))


def _get_latest_checkpoint_name():
    if not os.path.exists(opt.checkpoints_dir):
        return None
    epochs = []
    prefix = CKPT_STEM + "_"
    for name in os.listdir(opt.checkpoints_dir):
        stem, ext = os.path.splitext(name)
        if ext == ".pt" and stem.startswith(prefix):
            tail = stem[len(prefix):]
            if tail.isdigit():
                epochs.append(int(tail))
    return "{}_{}.pt".format(CKPT_STEM, max(epochs)) if epochs else None


def _resolve_ckpt_path(raw, root):
    if os.path.isabs(raw):
        p = Path(raw)
    else:
        uieb = Path(__file__).resolve().parent
        candidates = [(uieb / raw).resolve(), (root / raw).resolve()]
        p = next((c for c in candidates if c.is_file()), (uieb / raw).resolve())
    if not p.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(p))
    return str(p)


if __name__ == "__main__":
    import argparse

    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--init_from_ccrm", default="../ckpts/CCRM/CWMNetCCRM.pt")
    extra.add_argument("--decoder_warmup_epochs", type=int, default=0)
    extra.add_argument("--resume_epoch", type=int, default=0)
    extra.add_argument("--reset_optimizer", action="store_true")
    extra.add_argument("--batch_size", type=int, default=3)
    extra.add_argument("--end_epoch", type=int, default=30)
    extra.add_argument("--grad_clip", type=float, default=0.5)
    extra.add_argument("--checkpoints_dir", default="../ckpts/CWM-Net/")
    extra_args, _ = extra.parse_known_args()
    opt.batch_size = extra_args.batch_size
    opt.end_epoch = extra_args.end_epoch
    opt.grad_clip = extra_args.grad_clip
    opt.checkpoints_dir = extra_args.checkpoints_dir

    print("[CWM-Net] decoder_warmup_epochs={} wam_lr_scale={} grad_clip={}".format(
        extra_args.decoder_warmup_epochs, opt.wam_lr_scale, opt.grad_clip))

    batches = int(opt.num_images / opt.batch_size)
    model = CWMNet().to(device)

    start_epoch = 1
    checkpoint_g = None
    latest = None
    if extra_args.resume_epoch > 0:
        latest = "{}_{}.pt".format(CKPT_STEM, extra_args.resume_epoch)
        ckpt_file = os.path.join(opt.checkpoints_dir, latest)
        if os.path.isfile(ckpt_file):
            checkpoint_g = torch.load(ckpt_file, map_location=device, weights_only=False)
            start_epoch = checkpoint_g["epoch"] + 1
            model.load_state_dict(remap_state_dict(checkpoint_g["model_state_dict"]))
            print("[CWM-Net] resume from", latest, "start_epoch=", start_epoch)
    if checkpoint_g is None and not opt.fresh:
        latest = _get_latest_checkpoint_name()
        ckpt_file = os.path.join(opt.checkpoints_dir, latest) if latest else None
        if latest and os.path.isfile(ckpt_file):
            checkpoint_g = torch.load(ckpt_file, map_location=device, weights_only=False)
            start_epoch = checkpoint_g["epoch"] + 1
            model.load_state_dict(remap_state_dict(checkpoint_g["model_state_dict"]))
            print("[CWM-Net] resume from", latest, "start_epoch=", start_epoch)
    if checkpoint_g is None:
        if not extra_args.init_from_ccrm:
            raise ValueError("Need a CCRM checkpoint at the default path, or pass --init_from_ccrm")
        init_path = _resolve_ckpt_path(extra_args.init_from_ccrm, _ROOT)
        missing, _ = load_ccrm_into_cwmnet(model, init_path, device)
        print("[CWM-Net] loaded CCRM from", init_path)
        print("[CWM-Net] new WAM keys:", [k for k in missing if k.startswith("wam.")])

    mae_loss = nn.L1Loss()
    vgg = Vgg19(requires_grad=False).to(device)
    vgg.eval()
    ds = dataset.Dataset_Load(opt.hazydir, opt.cleandir, transform=dataset.ToTensor())
    dataloader = DataLoader(ds, batch_size=opt.batch_size, num_workers=5, pin_memory=True, shuffle=True)
    os.makedirs(opt.checkpoints_dir, exist_ok=True)

    phase = _apply_phase(model, start_epoch, extra_args.decoder_warmup_epochs)
    optim_g = _build_optimizer(model)
    if checkpoint_g is not None and not extra_args.reset_optimizer:
        try:
            optim_g.load_state_dict(checkpoint_g["optimizer_state_dict"])
        except Exception as e:
            print("[Warn] optimizer reset due to:", e)

    best_state = {"psnr": -1.0, "epoch": None, "ssim": None}
    last_phase = phase
    model.train()

    for epoch in range(start_epoch, opt.end_epoch + 1):
        phase = _apply_phase(model, epoch, extra_args.decoder_warmup_epochs)
        if phase != last_phase:
            optim_g = _build_optimizer(model)
            print("[CWM-Net] ep {}: switch to phase '{}' (optimizer rebuilt)".format(epoch, phase))
            last_phase = phase
        elif epoch == start_epoch:
            print("[CWM-Net] ep {}: phase '{}'".format(epoch, phase))

        opt.total_mae_loss = opt.total_vgg_loss = opt.total_ssim_loss = 0.0
        for i_batch, sample in enumerate(dataloader):
            hazy = sample["hazy"].to(device)
            clean = sample["clean"].to(device)
            pred = model(hazy)

            l_mae = torch.mul(opt.lambda_mae, mae_loss(pred, clean))
            l_ssim = 1 - ssim(pred, clean, data_range=1, size_average=True)
            l_vgg = torch.mul(
                opt.lambda_vgg,
                mae_loss(vgg(normalize_batch(pred)).relu4_3, vgg(normalize_batch(clean)).relu4_3),
            )
            (l_mae + l_ssim + l_vgg).backward()
            if opt.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], opt.grad_clip)
            optim_g.step()
            optim_g.zero_grad()
            opt.total_mae_loss += l_mae.item()
            opt.total_ssim_loss += l_ssim.item()
            opt.total_vgg_loss += l_vgg.item()
            print("\r Epoch {} ({}/{}) phase={}".format(epoch, i_batch + 1, batches, phase), end="", flush=True)

        print("\nFinished ep {} phase={} mean_ssim={:.4f}".format(
            epoch, phase, 1 - opt.total_ssim_loss / batches))

        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim_g.state_dict(),
            "opt": opt,
        }
        ckpt_path = _abs_ckpt_path(opt.checkpoints_dir, epoch)
        torch.save(payload, ckpt_path)

        if opt.eval_every > 0 and (epoch % opt.eval_every == 0 or epoch == opt.end_epoch):
            from eval_simple import eval_paired_dirs

            m = eval_paired_dirs(model, opt.testing_dir_inp, opt.testing_dir_gt, device)
            tp, ts = m["psnr_paper"], m["ssim_paper"]
            print("[Eval] ep {:02d} test_psnr={:.4f} ssim={:.4f}".format(epoch, tp, ts))
            if tp > best_state["psnr"]:
                best_state.update(psnr=tp, ssim=ts, epoch=epoch)
                torch.save(payload, str(Path(opt.checkpoints_dir) / "{}_best_test.pt".format(CKPT_STEM)))
                print("[Best] {:.4f} ep {}".format(tp, epoch))

    if best_state.get("epoch"):
        print("\n[Done] best {:.4f} dB @ ep {}".format(best_state["psnr"], best_state["epoch"]))
