"""Train AGFM on a Stage-1 UCCRG checkpoint."""
import json
import os
import sys
from pathlib import Path

_UIEB = Path(__file__).resolve().parent
if str(_UIEB) not in sys.path:
    sys.path.insert(0, str(_UIEB))

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_msssim import ssim
from torch.utils.data import DataLoader

import Config.dataset as dataset
from Config.models_stage1_uccrg_plus import DICAM_UCCP_Stage1UCCRG, load_stage1uccrg_into_uccp
from Config.options import device, opt
from Config.vgg import Vgg19, normalize_batch

CKPT_STEM = "DICAM_UCCP_Stage1UCCRG"


def _set_stage1_frozen(model, frozen):
    for name, p in model.named_parameters():
        if name.startswith(("layer_1_", "layer_2_")):
            p.requires_grad = not frozen


def _set_decoder_frozen(model, frozen):
    for name, p in model.named_parameters():
        if name.startswith(("layer_3", "layer_4", "layer_tail")):
            p.requires_grad = not frozen


def _set_uccp_trainable(model, trainable):
    for name, p in model.named_parameters():
        if name.startswith("uccp."):
            p.requires_grad = trainable


def _apply_phase(model, epoch, decoder_warmup_epochs):
    if epoch <= decoder_warmup_epochs:
        _set_stage1_frozen(model, True)
        _set_decoder_frozen(model, False)
        _set_uccp_trainable(model, False)
        return "decoder"
    _set_stage1_frozen(model, True)
    _set_decoder_frozen(model, True)
    _set_uccp_trainable(model, True)
    return "agfm"


def _build_optimizer(model):
    uccp_lr = opt.learning_rate_g * opt.uccp_lr_scale
    backbone = [p for n, p in model.named_parameters() if not n.startswith("uccp.") and p.requires_grad]
    uccp = [p for n, p in model.named_parameters() if n.startswith("uccp.") and p.requires_grad]
    if uccp and backbone and opt.uccp_lr_scale != 1.0:
        return optim.Adam(
            [{"params": backbone, "lr": opt.learning_rate_g}, {"params": uccp, "lr": uccp_lr}],
            betas=(opt.beta1, opt.beta2),
            weight_decay=opt.wd_g,
        )
    trainable = [p for p in model.parameters() if p.requires_grad]
    lr = uccp_lr if uccp and not backbone else opt.learning_rate_g
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


def _resolve_data_dir(root):
    return str((root / "Data" / "UIEB").resolve())


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
    extra.add_argument("--init_from_stage1uccrg", default="")
    extra.add_argument("--decoder_warmup_epochs", type=int, default=0,
                       help="0=only train AGFM (recommended after tiered UCCRG)")
    extra.add_argument("--resume_epoch", type=int, default=0)
    extra.add_argument("--reset_optimizer", action="store_true")
    extra_args, _ = extra.parse_known_args()

    _ROOT = Path(__file__).resolve().parents[1]
    print("[UCCRG+AGFM] decoder_warmup_epochs={} uccp_lr_scale={} grad_clip={}".format(
        extra_args.decoder_warmup_epochs, opt.uccp_lr_scale, opt.grad_clip))

    batches = int(opt.num_images / opt.batch_size)
    model = DICAM_UCCP_Stage1UCCRG().to(device)

    start_epoch = 1
    checkpoint_g = None
    if opt.fresh:
        for p in Path(opt.checkpoints_dir).glob("{}_*.pt".format(CKPT_STEM)):
            p.unlink()
        if not extra_args.init_from_stage1uccrg:
            raise ValueError("--fresh requires --init_from_stage1uccrg")
        init_path = _resolve_ckpt_path(extra_args.init_from_stage1uccrg, _ROOT)
        missing, _ = load_stage1uccrg_into_uccp(model, init_path, device)
        print("[UCCRG+AGFM] loaded Stage1UCCRG from", init_path)
        print("[UCCRG+AGFM] new AGFM keys:", [k for k in missing if k.startswith("uccp.")])
    else:
        if extra_args.resume_epoch > 0:
            latest = "{}_{}.pt".format(CKPT_STEM, extra_args.resume_epoch)
            ckpt_file = os.path.join(opt.checkpoints_dir, latest)
        else:
            latest = _get_latest_checkpoint_name()
            ckpt_file = os.path.join(opt.checkpoints_dir, latest) if latest else None
        if not latest or not os.path.isfile(ckpt_file):
            raise FileNotFoundError("No checkpoint to resume: {}".format(ckpt_file))
        checkpoint_g = torch.load(ckpt_file, map_location=device, weights_only=False)
        start_epoch = checkpoint_g["epoch"] + 1
        model.load_state_dict(checkpoint_g["model_state_dict"])
        print("[UCCRG+AGFM] resume from", latest, "start_epoch=", start_epoch)

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
            print("[UCCRG+AGFM] ep {}: switch to phase '{}' (optimizer rebuilt)".format(epoch, phase))
            last_phase = phase
        elif epoch == start_epoch:
            print("[UCCRG+AGFM] ep {}: phase '{}'".format(epoch, phase))

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
            sys.path.insert(0, str(_ROOT))
            from eval_stage1_uccrg_agfm_uieb import eval_checkpoint_stage1uccrg_agfm

            m = eval_checkpoint_stage1uccrg_agfm(ckpt_path, _resolve_data_dir(_ROOT), "test")
            tp, ts = m["test"]["psnr_paper"], m["test"]["ssim_paper"]
            print("[Eval] ep {:02d} test_psnr={:.4f} ssim={:.4f}".format(epoch, tp, ts))
            if tp > best_state["psnr"]:
                best_state.update(psnr=tp, ssim=ts, epoch=epoch)
                torch.save(payload, str(Path(opt.checkpoints_dir) / "{}_best_test.pt".format(CKPT_STEM)))
                print("[Best] {:.4f} ep {}".format(tp, epoch))

    if best_state.get("epoch"):
        print("\n[Done] best {:.4f} dB @ ep {}".format(best_state["psnr"], best_state["epoch"]))
