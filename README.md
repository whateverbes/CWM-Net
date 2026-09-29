# CWM-Net

Underwater image enhancement .

## Setup

```bash
pip install -r requirements.txt
```

Put UIEB `raw-890` and `reference-890` somewhere, then:

```bash
python prepare_data.py --uieb_root /path/to/UIEB --out Data/UIEB --force
```

This writes `Data/UIEB/hazy_train`, `clean_train`, `hazy_test`, `clean_test`.

## Train

Run from `UIEB/`. Paths in `Config/options.py` are relative (`../Data/UIEB/`, `../ckpts/`).

**1) DICAM backbone (60 epochs)**

```bash
cd UIEB
python train_uieb.py --checkpoints_dir ../ckpts/UIEB --end_epoch 60
```

**2) Stage-1 UCCRG (tiered finetune from DICAM)**

```bash
python train_stage1_uccrg.py \
  --fresh \
  --init_from_dicam ../ckpts/UIEB/DICAM_60.pt \
  --finetune_tiered \
  --lr_gate 1e-4 --lr_cam 5e-5 --lr_backbone 1e-5 \
  --checkpoints_dir ../ckpts/stage1_uccrg \
  --batch_size 3 --end_epoch 30 --grad_clip 0.5
```

**3) CWM-Net: freeze backbone, train AGFM**

```bash
python train_stage1_uccrg_agfm.py \
  --fresh \
  --init_from_stage1uccrg ../ckpts/stage1_uccrg/DICAM_Stage1UCCRG_best_test.pt \
  --checkpoints_dir ../ckpts/CWM-Net \
  --decoder_warmup_epochs 0 \
  --uccp_lr_scale 0.5 \
  --batch_size 3 --end_epoch 30 --grad_clip 0.5
```

If there is no `*_best_test.pt` (eval is off by default), pass a numbered checkpoint such as `DICAM_Stage1UCCRG_30.pt`.

## Test

```bash
cd UIEB
python test.py \
  --checkpoint ../ckpts/CWM-Net/DICAM_UCCP_Stage1UCCRG_4.pt \
  --inp_dir ../Data/UIEB/hazy_test \
  --result_dir ../results/UIEB_CWM-Net
```

Enhanced images are written to `results/UIEB_CWM-Net/`.

Loss: L1 + SSIM + VGG19 `relu4_3` (equal weights), same as DICAM.
