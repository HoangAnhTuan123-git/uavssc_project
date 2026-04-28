# UAVSSC RGB+LiDAR Fusion Gate3D Starter

This folder is a **RGB + LiDAR fusion** training starter for UAVScenes SSC built around your exported `.npz` files.

It contains a mid-level voxel fusion starter with a learned confidence gate. This package is a custom baseline, not an official external repo.

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
NPZ type: **Fusion exporter (`10_export_fusion_ssc.py`)**
