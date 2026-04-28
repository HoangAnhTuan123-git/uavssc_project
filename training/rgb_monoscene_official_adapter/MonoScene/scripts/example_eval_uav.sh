#!/usr/bin/env bash
set -euo pipefail
python monoscene/scripts/eval_uavscenes.py   uav_preprocess_root=/path/to/rgb_npz_root   uav_data_root=/path/to/UAVScenes   eval_checkpoint_path=/path/to/checkpoint.ckpt   n_gpus=1
