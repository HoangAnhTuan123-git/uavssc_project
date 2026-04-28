# UAVSSC RGB-only VoxFormer-Style Starter

This folder is a **RGB-only VoxFormer-style** training starter for UAVScenes SSC built around your exported `.npz` files.

It contains a lightweight **research starter** inspired by VoxFormer, plus a bootstrap script for the official repo.

## Expected NPZ root
Point `data.preprocess_root` to the folder that contains scene subfolders, for example:

```text
/path/to/rgb_npz_root/
  AMtown01/*.npz
  AMtown02/*.npz
```

## Quick start

```bash
python train.py --config configs/default.yaml
python eval.py  --config configs/default.yaml --checkpoint runs/default/checkpoints/best.pt
```

## What this package expects
NPZ type: **RGB exporter (`10_export_rgb_ssc.py`)**
