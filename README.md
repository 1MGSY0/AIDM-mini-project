# AIDM-mini-project

Project to test an algorithm to classify whether an image contains either a dog or a cat.

Data source: https://www.kaggle.com/c/dogs-vs-cats/data

## Contents

- [1) Clone the repo](#1-clone-the-repo)
- [2) Create and activate a Python environment](#2-create-and-activate-a-python-environment)
- [3) Data layout](#3-data-layout)
- [4) Where to change parameters](#4-where-to-change-parameters)
- [5) Run training](#5-run-training)
- [6) Outputs](#6-outputs)
- [7) GPU vs CPU](#7-gpu-vs-cpu)
- [8) Tips & troubleshooting](#8-tips--troubleshooting)
- [9) Re-running or customizing](#9-re-running-or-customizing)

This repo includes:
1. training script (`Resnet-finetune.py`) that trains, validates, and produces a `submission.csv`, along with a timestamped training report under report/.
...



## 1) Clone the repo

```powershell
# PowerShell
git clone https://github.com/1MGSY0/AIDM-mini-project.git
cd AIDM-mini-project
```

## 2) Create and activate a Python environment

```powershell
# Create a virtual environment (Windows)
python -m venv env

# Activate it (PowerShell)
./env/Scripts/Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Notes
- Requirements include CUDA-enabled PyTorch wheels (`cu121`) via an extra index URL in `requirements.txt`. 
    If CPU-only, remove the `--extra-index-url` line and install CPU wheels instead.
- Python 3.10+ recommended.

## 3) Data layout

Inside the project folder, the code expects a `datasets` folder:

```
AIDM-mini-project/
  datasets/
    train/
      cat/   ... images ...
      dog/   ... images ...
    val/
      cat/   ... images ...
      dog/   ... images ...
    test/
      1.jpg
      2.jpg
      ...
```

- Test is a flat folder of images (no labels). By default the code looks for `.jpg` files.

## 4) Where to change parameters

Open `Resnet-finetune.py` and look near the top for the key knobs:

```python
# Resnet-finetune.py
DATA_ROOT = "datasets"   # set your dataset root folder
IMG_SIZE = 224            # input resolution (ResNet-18 default)
BATCH = 64                # adjust for GPU memory
EPOCHS = 15               # training epochs
LR = 3e-4                 # AdamW learning rate
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # auto-pick

# Data Random subset
MAX_TRAIN_SAMPLES = 4000  # set None to disable cap
MAX_VAL_SAMPLES = 1000
```
- Data transforms are defined in `train_tfms` and `eval_tfms` if you want to change augmentation/normalization.


## 5) Run training

```powershell
# From the repo root with the venv activated
python Resnet-finetune.py
```
What happens
- Model: ResNet-18 (ImageNet-pretrained) with final `fc` swapped to 2 classes.
- Training: AdamW + CosineAnnealingLR; AMP used on GPU for speed.
- Logging: per-epoch metrics printed and saved to a timestamped report file under `reports/`.
- Checkpoint: best model weights (by val accuracy) saved to `best.pt` (state_dict).
- Inference: runs on the `datasets/test` folder to produce `submission.csv`.

## 6) Outputs

- Training reports: saved under `reports/` as `training_YYYYMMDD_HHMMSS.txt`.
  - Includes: device/GPU info, parameters, per-epoch `train_loss`, `val_loss`, `val_acc`, and when available precision/recall/F1 and a confusion matrix.
- Submission file: `submission.csv` at the repo root.
  - Structure (header + rows):

```
id,predicted
0001.jpg,1
0002.jpg,0
...
```

- Label mapping used by the script: `1 = dog`, `0 = cat`.

## 7) GPU vs CPU

- The script auto-selects `cuda` if available; otherwise uses `cpu`.
- To verify:

```powershell
python -c "import torch; print(torch.__version__, 'cuda?', torch.cuda.is_available());
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If CUDA is available, mixed precision is enabled and host→GPU transfers use pinned memory for speed.

## 8) Tips & troubleshooting

- Out-of-memory (GPU): lower `BATCH`, or reduce `IMG_SIZE` (e.g., 160), or turn off heavy augmentations.
- Slow data loading: increase `num_workers` (2–8), keep `pin_memory=True` on GPU. Consider keeping `persistent_workers=True` for faster epochs.
- Windows multiprocessing: if you hit worker/pickling issues, set `num_workers=0` for simplicity.
- Quick smoke test: set `EPOCHS = 1`, and smaller caps like `MAX_TRAIN_SAMPLES = 512`, `MAX_VAL_SAMPLES = 256`.

## 9) Re-running or customizing

- You can safely re-run `python Resnet-finetune.py`; a new report will be generated each time. The best model weights overwrite `best.pt` when validation accuracy improves.
- To add CLI flags (e.g., `--max-train`, `--max-val`, `--epochs`), we can wire `argparse` so teammates can tweak settings without editing the file.

---
