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

**1) backbone **

```bash
cd UIEB
python train_uieb.py --checkpoints_dir ../ckpts/UIEB --end_epoch 60
```

**2) Stage-1**

```bash
python train_stage1_uccrg.py
```

**3) CWM-Net**

```bash
python train_stage1_uccrg_agfm.py
```


## Test

```bash
cd UIEB
python test.py
```
