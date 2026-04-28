#!/usr/bin/env bash
set -euo pipefail
python monoscene/scripts/train_uavscenes.py   uav_preprocess_root=/path/to/rgb_npz_root   uav_data_root=/path/to/UAVScenes   uav_logdir=./uav_logs   n_gpus=1 batch_size=1 max_epochs=30
