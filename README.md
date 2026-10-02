# CWM-Net

## Setup

```bash
pip install -r requirements.txt
python prepare_data.py --uieb_root /path/to/UIEB
python prepare_euvp.py --euvp_root /path/to/EUVP
```

## Train

```bash
cd UIEB
python train_uieb.py
python train_stage1_uccrg.py
python train_cwmnet.py
```

EUVP:

```bash
cd EUVP
python train_euvp.py --arch base
python train_euvp.py --arch wam
python train_euvp.py --arch cwmnet
```

Fusion ablation:

```bash
cd UIEB
python train_fusion_ablation.py
```

## Test

Weights are under `ckpts/`.

```bash
cd UIEB
python test.py
```

EUVP:

```bash
cd EUVP
python test.py
```

Ablation variants: `base`, `ccrm`, `wam`, `cwmnet`.
