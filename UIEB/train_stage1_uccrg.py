"""Train Stage-1 UCCRG."""
import json
import os
import sys
from pathlib import Path

_UIEB = Path(__file__).resolve().parent
_ROOT = _UIEB.parent
for _p in (_UIEB, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torchimport torch.nn as nn
import torch.optim as optim
from pytorch_msssim import ssim
from torch.utils.data import DataLoader

import Config.dataset as dataset
from Config.models_stage1_uccrg import CWMNetCCRM
from Config.options import device, opt
from Config.vgg import Vgg19, normalize_batch

try:
    from torchsummary import summary
except Exception:
    summary = None

CKPT_STEM = "CWMNetCCRM"


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def _abs_ckpt_path(checkpoints_dir, epoch):
    return os.path.abspath(os.path.join(checkpoints_dir, "{}_{}.pt".format(CKPT_STEM, epoch)))


def _get_latest_checkpoint_name():
    if not os.path.exists(opt.checkpoints_dir):
        return None
    epochs = []
    prefix = CKPT_STEM + "_"
    for name in os.listdir(opt.checkpoints_dir):
        stem, ext = os.path.splitext(name)
        if ext != ".pt" or not stem.startswith(prefix):
            continue
        tail = stem[len(prefix) :]
        if tail.isdigit():
            epochs.append(int(tail))
    if not epochs:
        return None
    return "{}_{}.pt".format(CKPT_STEM, max(epochs))


def _resolve_data_dir(root):
    for cand in [(root / "Data" / "UIEB").resolve(), Path(opt.data_dir)]:
        if (cand / "split_mapping.json").is_file():
            return str(cand)
    return str((root / "Data" / "UIEB").resolve())


def _is_uccrg_param(name: str) -> bool:
    return "uccrg_gate" in name


def _set_uccrg_trainable(model, trainable: bool) -> None:
    for name, p in model.named_parameters():
        if _is_uccrg_param(name):
            p.requires_grad = trainable


def _set_backbone_trainable(model, trainable: bool) -> None:
    """Freeze all except uccrg_gate (for finetune from CWMNetBase)."""
    for name, p in model.named_parameters():
        if "uccrg_gate" not in name:
            p.requires_grad = trainable


def _is_cam_param(name: str) -> bool:
    return name.startswith("layer_2_") and ".module." in name


def _param_bucket(name: str) -> str:
    if _is_uccrg_param(name):
        return "gate"
    if _is_cam_param(name):
        return "cam"
    return "backbone"


def _set_tiered_trainable(model, gate: bool, cam: bool, backbone: bool) -> None:
    for name, p in model.named_parameters():
        b = _param_bucket(name)
        if b == "gate":
            p.requires_grad = gate
        elif b == "cam":
            p.requires_grad = cam
        else:
            p.requires_grad = backbone


def _build_tiered_optimizer(model, lr_gate: float, lr_cam: float, lr_backbone: float):
    groups = {"gate": [], "cam": [], "backbone": []}
    for name, p in model.named_parameters():
        if p.requires_grad:
            groups[_param_bucket(name)].append(p)
    param_groups = []
    if groups["gate"]:
        param_groups.append({"params": groups["gate"], "lr": lr_gate})
    if groups["cam"]:
        param_groups.append({"params": groups["cam"], "lr": lr_cam})
    if groups["backbone"]:
        param_groups.append({"params": groups["backbone"], "lr": lr_backbone})
    print(
        "[CCRM] tiered lr gate={} cam={} backbone={} | n_params gate/cam/backbone: {}/{}/{}".format(
            lr_gate, lr_cam, lr_backbone,
            len(groups["gate"]), len(groups["cam"]), len(groups["backbone"]),
        )
    )
    return optim.Adam(param_groups, betas=(opt.beta1, opt.beta2), weight_decay=opt.wd_g)


def _load_init_ckpt(model, ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=True)
    print("[CCRM] init_ckpt={} (epoch {})".format(ckpt_path, ckpt.get("epoch")))


def _build_optimizer(model, uccrg_lr_scale: float, finetune_gate_only: bool):
    base = [p for n, p in model.named_parameters() if not _is_uccrg_param(n)]
    uccrg = [p for n, p in model.named_parameters() if _is_uccrg_param(n)]
    ulr = opt.learning_rate_g * uccrg_lr_scale
    if finetune_gate_only and uccrg:
        print("[CCRM] finetune gate-only lr={}".format(ulr))
        return optim.Adam(
            uccrg, lr=ulr, betas=(opt.beta1, opt.beta2), weight_decay=opt.wd_g
        )
    if uccrg and uccrg_lr_scale != 1.0:
        print("[CCRM] lr backbone={} uccrg={}".format(opt.learning_rate_g, ulr))
        return optim.Adam(
            [{"params": base, "lr": opt.learning_rate_g}, {"params": uccrg, "lr": ulr}],
            betas=(opt.beta1, opt.beta2),
            weight_decay=opt.wd_g,
        )
    return optim.Adam(
        model.parameters(),
        lr=opt.learning_rate_g,
        betas=(opt.beta1, opt.beta2),
        weight_decay=opt.wd_g,
    )


if __name__ == "__main__":
    import argparse

    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--freeze_uccrg_epochs", type=int, default=0)
    extra.add_argument("--uccrg_lr_scale", type=float, default=1.0,
                       help="LR on uccrg_gate.*; use 0.1 when --init_from_base")
    extra.add_argument("--init_from_base", default="../ckpts/UIEB/CWMNetBase_60.pt")
    extra.add_argument("--gate_scale", type=float, default=1.0)
    extra.add_argument("--freeze_backbone", action="store_true")
    extra.add_argument("--no_freeze_backbone", action="store_true")
    extra.add_argument("--init_ckpt", default="")
    extra.add_argument("--finetune_tiered", type=int, default=1)
    extra.add_argument("--lr_gate", type=float, default=1e-4)
    extra.add_argument("--lr_cam", type=float, default=5e-5)
    extra.add_argument("--lr_backbone", type=float, default=1e-5)
    extra.add_argument("--tiered_after_epochs", type=int, default=0)
    extra.add_argument("--batch_size", type=int, default=3)
    extra.add_argument("--end_epoch", type=int, default=30)
    extra.add_argument("--grad_clip", type=float, default=0.5)
    extra.add_argument("--checkpoints_dir", default="../ckpts/CCRM/")
    extra_args, _ = extra.parse_known_args()
    opt.batch_size = extra_args.batch_size
    opt.end_epoch = extra_args.end_epoch
    opt.grad_clip = extra_args.grad_clip
    opt.checkpoints_dir = extra_args.checkpoints_dir
    os.makedirs(opt.checkpoints_dir, exist_ok=True)

    finetune_gate_only = (
        not bool(extra_args.finetune_tiered)
        and (
            extra_args.freeze_backbone
            or (bool(extra_args.init_from_base) and not extra_args.no_freeze_backbone)
        )
    )

    print("[CCRM] train UCCRG (no WAM)")
    print("[CCRM] hazydir=", opt.hazydir)
    print("[CCRM] ckpts=", opt.checkpoints_dir)
    print(
        "[CCRM] lr={} finetune_gate_only={} finetune_tiered={} tiered_after={} grad_clip={} epochs={}".format(
            opt.learning_rate_g,
            finetune_gate_only,
            extra_args.finetune_tiered,
            extra_args.tiered_after_epochs,
            opt.grad_clip,
            opt.end_epoch,
        )
    )

    probe = os.path.join(opt.hazydir, "img1" + opt.img_extension)
    if not os.path.isfile(probe):
        raise FileNotFoundError("Missing {}".format(probe))

    batches = int(opt.num_images / opt.batch_size)
    if opt.shared_init_seed >= 0:
        torch.manual_seed(opt.shared_init_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opt.shared_init_seed)

    model = CWMNetCCRM(gate_scale=extra_args.gate_scale).to(device)
    if extra_args.init_ckpt:
        init_p = extra_args.init_ckpt
        if not os.path.isabs(init_p):
            init_p = str((_ROOT / init_p).resolve())
        _load_init_ckpt(model, init_p, device)
    elif extra_args.init_from_base:
        from Config.models_stage1_uccrg import load_base_into_ccrm

        init_p = extra_args.init_from_base
        if not os.path.isabs(init_p):
            init_p = str((_ROOT / init_p).resolve())
        load_base_into_ccrm(model, init_p, device)
        print("[CCRM] init_from_base={}".format(init_p))
    if summary is not None:
        try:
            summary(model, input_size=(3, 256, 256))
        except Exception as e:
            print("[Warn] torchsummary skipped:", e)

    mae_loss = nn.L1Loss()
    vgg = Vgg19(requires_grad=False).to(device)
    vgg.eval()
    train_state = {"optim_g": None, "tiered_mode": None}

    def _setup_epoch_mode(epoch):
        if extra_args.finetune_tiered:
            if epoch <= extra_args.tiered_after_epochs:
                mode = "gate"
                _set_tiered_trainable(model, gate=True, cam=False, backbone=False)
            else:
                mode = "tiered"
                _set_tiered_trainable(model, gate=True, cam=True, backbone=True)
            if mode != train_state["tiered_mode"]:
                if mode == "gate":
                    gate_p = [p for _, p in model.named_parameters() if p.requires_grad]
                    train_state["optim_g"] = optim.Adam(
                        gate_p, lr=extra_args.lr_gate,
                        betas=(opt.beta1, opt.beta2), weight_decay=opt.wd_g,
                    )
                    print("[CCRM] ep {}: gate-only lr={}".format(epoch, extra_args.lr_gate))
                else:
                    train_state["optim_g"] = _build_tiered_optimizer(
                        model,
                        extra_args.lr_gate,
                        extra_args.lr_cam,
                        extra_args.lr_backbone,
                    )
                    print("[CCRM] ep {}: tiered unfreeze".format(epoch))
                train_state["tiered_mode"] = mode
            return
        if finetune_gate_only:
            _set_backbone_trainable(model, False)
            _set_uccrg_trainable(model, True)
            if train_state["optim_g"] is None:
                train_state["optim_g"] = _build_optimizer(model, extra_args.uccrg_lr_scale, True)
        else:
            uccrg_on = epoch > extra_args.freeze_uccrg_epochs
            _set_uccrg_trainable(model, uccrg_on)
            if train_state["optim_g"] is None:
                train_state["optim_g"] = _build_optimizer(model, extra_args.uccrg_lr_scale, False)
            if not uccrg_on:
                print("[CCRM] ep {}: UCCRG gates frozen (backbone only)".format(epoch))

    if not extra_args.finetune_tiered:
        train_state["optim_g"] = _build_optimizer(model, extra_args.uccrg_lr_scale, finetune_gate_only)

    ds = dataset.Dataset_Load(
        hazy_path=opt.hazydir,
        clean_path=opt.cleandir,
        transform=dataset.ToTensor(),
    )
    dataloader = DataLoader(
        ds, batch_size=opt.batch_size, num_workers=5, pin_memory=True, shuffle=True
    )

    if not os.path.exists(opt.checkpoints_dir):
        os.makedirs(opt.checkpoints_dir)

    start_epoch = 1
    if opt.fresh:
        for p in Path(opt.checkpoints_dir).glob("{}_*.pt".format(CKPT_STEM)):
            p.unlink()
        best = Path(opt.checkpoints_dir) / "{}_best_test.pt".format(CKPT_STEM)
        if best.is_file():
            best.unlink()
    else:
        latest = _get_latest_checkpoint_name()
        if latest:
            ckpt = torch.load(
                os.path.join(opt.checkpoints_dir, latest),
                map_location=device,
                weights_only=False,
            )
            start_epoch = ckpt["epoch"] + 1
            model.load_state_dict(ckpt["model_state_dict"])
            train_state["optim_g"].load_state_dict(ckpt["optimizer_state_dict"])
            print("[CCRM] resume", latest, "start_epoch=", start_epoch)

    eval_log_path = Path(opt.checkpoints_dir) / "train_eval_log_ccrm.json"
    best_state = {"psnr": -1.0, "epoch": None, "ssim": None}
    last_best_epoch = start_epoch - 1
    model.train()

    for epoch in range(start_epoch, opt.end_epoch + 1):
        _setup_epoch_mode(epoch)

        opt.total_mae_loss = 0.0
        opt.total_vgg_loss = 0.0
        opt.total_loss = 0.0
        opt.total_ssim_loss = 0.0

        for i_batch, sample_batched in enumerate(dataloader):
            hazy_batch = sample_batched["hazy"].to(device)
            clean_batch = sample_batched["clean"].to(device)
            pred_batch = model(hazy_batch)

            batch_mae_loss = torch.mul(opt.lambda_mae, mae_loss(pred_batch, clean_batch))
            batch_ssim_loss = 1 - ssim(pred_batch, clean_batch, data_range=1, size_average=True)
            clean_vgg_feats = vgg(normalize_batch(clean_batch))
            pred_vgg_feats = vgg(normalize_batch(pred_batch))
            batch_vgg_loss = torch.mul(
                opt.lambda_vgg, mae_loss(pred_vgg_feats.relu4_3, clean_vgg_feats.relu4_3)
            )
            (batch_mae_loss + batch_ssim_loss + batch_vgg_loss).backward()

            if opt.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

            opt.batch_mae_loss = batch_mae_loss.detach().cpu().item()
            opt.total_mae_loss += opt.batch_mae_loss
            opt.batch_ssim_loss = batch_ssim_loss.detach().cpu().item()
            opt.total_ssim_loss += opt.batch_ssim_loss
            opt.batch_vgg_loss = batch_vgg_loss.detach().cpu().item()
            opt.total_vgg_loss += opt.batch_vgg_loss
            opt.batch_loss = opt.batch_mae_loss + opt.batch_ssim_loss + opt.batch_vgg_loss
            opt.total_loss += opt.batch_loss

            train_state["optim_g"].step()
            train_state["optim_g"].zero_grad()

            print(
                "\r Epoch : {} | ({}/{}) | l_mae: {} | l_ssim: {} | l_vgg: {}".format(
                    epoch,
                    i_batch + 1,
                    batches,
                    opt.batch_mae_loss / 2,
                    1 - opt.batch_ssim_loss,
                    opt.batch_vgg_loss,
                ),
                end="",
                flush=True,
            )

        print(
            "\n\nFinished ep. %d, lr = %.6f, mean_mae = %.6f, mean_ssim = %.6f, mean_vgg = %.6f"
            % (
                epoch,
                get_lr(train_state["optim_g"]),
                (opt.total_mae_loss / batches) / 2,
                1 - (opt.total_ssim_loss / batches),
                opt.total_vgg_loss / batches,
            )
        )

        ckpt_path = _abs_ckpt_path(opt.checkpoints_dir, epoch)
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": train_state["optim_g"].state_dict(),
            "mae_loss": opt.total_mae_loss,
            "ssim_loss": opt.total_ssim_loss,
            "vgg_loss": opt.total_vgg_loss,
            "opt": opt,
            "total_loss": opt.total_loss,
        }
        torch.save(payload, ckpt_path)

        if opt.eval_every > 0 and (epoch % opt.eval_every == 0 or epoch == opt.end_epoch):
            sys.path.insert(0, str(_ROOT))
            from eval_simple import eval_paired_dirs

            metrics = eval_paired_dirs(model, opt.testing_dir_inp, opt.testing_dir_gt, device)
            test_p = metrics["psnr_paper"]
            test_s = metrics["ssim_paper"]
            print("[Eval] ep {:02d} test_psnr_paper={:.4f} ssim_paper={:.4f}".format(epoch, test_p, test_s))

            log = []
            if eval_log_path.is_file():
                try:
                    log = json.loads(eval_log_path.read_text(encoding="utf-8")).get("log", [])
                except Exception:
                    log = []
            log.append({"epoch": epoch, "test_psnr_paper": test_p, "test_ssim_paper": test_s})

            if test_p > best_state["psnr"]:
                best_state["psnr"] = test_p
                best_state["epoch"] = epoch
                best_state["ssim"] = test_s
                best_path = Path(opt.checkpoints_dir).resolve() / "{}_best_test.pt".format(CKPT_STEM)
                torch.save(payload, str(best_path))
                print("[Best test] {:.4f} (ep {}) -> {}".format(test_p, epoch, best_path))
                last_best_epoch = epoch

            eval_log_path.write_text(
                json.dumps(
                    {
                        "best_test_paper": best_state["psnr"],
                        "best_epoch": best_state.get("epoch"),
                        "best_ssim_paper": best_state.get("ssim"),
                        "log": log,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            if (
                opt.early_stop_patience > 0
                and best_state["epoch"] is not None
                and epoch - last_best_epoch >= opt.early_stop_patience
            ):
                print("[EarlyStop] since epoch {}".format(last_best_epoch))
                break

    if best_state.get("epoch") is not None:
        print(
            "\n[Done] best test psnr_paper={:.4f} (epoch {})".format(
                best_state["psnr"], best_state["epoch"]
            )
        )
