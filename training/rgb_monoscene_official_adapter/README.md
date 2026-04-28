# UAVSSC MonoScene official-adapter package

This folder is based on the MonoScene repository you uploaded, with small patches for UAVScenes path resolution and config cleanup.

Contents:
- `MonoScene/` — patched repo copy
- `MonoScene/scripts/download_pretrained.sh` — download official MonoScene checkpoints
- `MonoScene/monoscene/config/uavscenes.yaml` — cleaned UAVScenes config

## Quick start

```bash
cd MonoScene
pip install -r requirements.txt
pip install -e .
bash scripts/download_pretrained.sh

python monoscene/scripts/train_uavscenes.py   uav_preprocess_root=/path/to/rgb_npz_root   uav_data_root=/path/to/UAVScenes   uav_logdir=./uav_logs   n_gpus=1 batch_size=1 max_epochs=30
```

## Notes
- This is the closest package here to an **official-code** route.
- It expects the **RGB-only NPZ exporter** output from your preprocessing.
- The patch removes the hard-coded Windows path rewrite and replaces it with a generic `uav_data_root` / `UAVSSC_DATA_ROOT` path resolver.
