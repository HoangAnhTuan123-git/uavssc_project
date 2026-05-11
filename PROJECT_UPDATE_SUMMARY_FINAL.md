# UAVSSC project update summary

This folder is the old project refreshed with the interval5 / CAM-label / MonoScene / VoxFormer / LiDAR / fusion fixes developed during debugging.

## Main code updates

### Preprocessing

- Interval5 remains the default workflow.
- `preprocessing/scripts/10_export_lidar_ssc.py` replaced with the current LiDAR-only exporter.
- `preprocessing/scripts/10_export_fusion_ssc.py` replaced with the current RGB+LiDAR fusion exporter.
- Export smoke-test scripts were added so you can test 200 samples before paying for full export.

### MonoScene RGB baseline

- EfficientNet-B0 launcher defaults are stable for RTX 4090:
  - `input_image_hw=[224,320]`
  - `feature=64`
  - `lr=1e-5`
  - `precision=16`
  - `num_workers=8`
- MonoScene loss functions now guard against:
  - AMP/BCE unsafe autocast
  - NaN/inf scalar losses
  - empty semantic-scaling batches
  - division by zero in geo/semantic scaling losses
- MonoScene training supports old PyTorch Lightning resume via `resume_from_checkpoint` in `Trainer(...)`.
- `scripts/patch_lightning_torch26_checkpoint_loading.py` was added for PyTorch 2.6 checkpoint resume compatibility.

### VoxFormer-style RGB baseline

- Stable config added: `training/rgb_voxformer_style/configs/uav_interval5_4090_stable.yaml`.
- Loss/trainer code now prevents a single NaN batch from making `history.csv` / `history.json` become NaN.
- Visualization script added: `training/rgb_voxformer_style/visualize.py`.
- Orientation-grid helper added: `training/rgb_voxformer_style/tools/orientation_grid.py`.
- Root launchers added:
  - `scripts/train_rgb_voxformer_interval5_4090.sh`
  - `scripts/eval_rgb_voxformer_interval5_4090.sh`
  - `scripts/visualize_rgb_voxformer_interval5_4090.sh`

### LiDAR and fusion baselines

- Stable configs added for:
  - `training/lidar_lmscnet_style/configs/uav_interval5_4090.yaml`
  - `training/lidar_scpnet_style/configs/uav_interval5_4090.yaml`
  - `training/rgb_lidar_fusion_gate3d/configs/uav_interval5_4090.yaml`
- Root launchers added:
  - `scripts/train_lidar_lmscnet_interval5_4090.sh`
  - `scripts/eval_lidar_lmscnet_interval5_4090.sh`
  - `scripts/train_rgb_lidar_fusion_interval5_4090.sh`
  - `scripts/eval_rgb_lidar_fusion_interval5_4090.sh`

## Key commands

### MonoScene stable training

```bash
cd /root/Tuan/uavssc_project
CUDA_VISIBLE_DEVICES=0 \
LR=0.00001 \
PRECISION=16 \
INPUT_IMAGE_HW='[224,320]' \
FEATURE=64 \
NUM_WORKERS=8 \
MAX_EPOCHS=10 \
FP_LOSS=false \
RELATION_LOSS=false \
CONTEXT_PRIOR=false \
bash scripts/train_rgb_monoscene_interval5_4090_b0.sh
```

### Resume MonoScene after interruption

If PyTorch 2.6 refuses to load the Lightning checkpoint, patch Lightning once:

```bash
python scripts/patch_lightning_torch26_checkpoint_loading.py
```

Then:

```bash
cd /root/Tuan/uavssc_project
CUDA_VISIBLE_DEVICES=0 \
RESUME_CKPT=/absolute/path/to/last.ckpt \
LOG_ROOT=/same/log/folder/as/old/run \
LR=0.00001 \
PRECISION=16 \
INPUT_IMAGE_HW='[224,320]' \
FEATURE=64 \
NUM_WORKERS=8 \
MAX_EPOCHS=10 \
bash scripts/train_rgb_monoscene_interval5_4090_b0.sh
```

### VoxFormer-style training

```bash
cd /root/Tuan/uavssc_project
bash scripts/train_rgb_voxformer_interval5_4090.sh
```

### VoxFormer-style visualization

First generate an orientation grid on one sample and pick the matching target orientation:

```bash
cd /root/Tuan/uavssc_project/training/rgb_voxformer_style
NPZ=$(find /root/Tuan/uavssc_project/data/processed/interval5/rgb_ssc_npz/interval5_AMtown01 -name '*.npz' | head -1)
python tools/orientation_grid.py --npz "$NPZ" --out /root/Tuan/uavssc_project/results/qualitative/orientation_grid.png
```

Then visualize with the chosen orientation:

```bash
cd /root/Tuan/uavssc_project
ORIENTATION=fliplr bash scripts/visualize_rgb_voxformer_interval5_4090.sh
```

Try `identity`, `flipud`, `fliplr`, `rot180`, `transpose`, `transpose_flipud`, `transpose_fliplr`, or `transpose_rot180` until VoxFormer target top-down matches MonoScene target top-down.

### LiDAR export smoke test

```bash
cd /root/Tuan/uavssc_project
bash scripts/export_lidar_interval5_test200.sh
```

If it is fast enough:

```bash
bash scripts/export_lidar_interval5_full.sh
```

### LiDAR LMSCNet-style training

```bash
cd /root/Tuan/uavssc_project
bash scripts/train_lidar_lmscnet_interval5_4090.sh
```

### Fusion export smoke test

```bash
cd /root/Tuan/uavssc_project
bash scripts/export_fusion_interval5_test200.sh
```

If it is fast enough:

```bash
bash scripts/export_fusion_interval5_full.sh
```

### Fusion training

```bash
cd /root/Tuan/uavssc_project
bash scripts/train_rgb_lidar_fusion_interval5_4090.sh
```

## Practical model decision

Keep MonoScene B0/F64 as the strongest RGB baseline so far. VoxFormer-style is currently useful as a second RGB-only baseline but is weaker. If GPU time is limited, prioritize a quick LiDAR LMSCNet-style run and then RGB+LiDAR fusion if export speed is acceptable.
