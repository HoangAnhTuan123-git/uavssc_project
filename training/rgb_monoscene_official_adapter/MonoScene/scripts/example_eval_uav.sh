#!/usr/bin/env bash
set -euo pipefail

# Example lightweight B0 evaluation command. The same settings must match training.
python -m monoscene.scripts.eval_uavscenes \
  uav_preprocess_root=/root/Tuan/uavssc_project/data/processed/interval5/rgb_ssc_npz \
  uav_data_root=/root/Tuan/uavssc_project/data/raw/uavscenes_official \
  eval_checkpoint_path=/path/to/checkpoint.ckpt \
  eval_split=val \
  input_image_hw='[320,384]' \
  rgb_backbone=tf_efficientnet_b0_ns \
  rgb_pretrained=false \
  feature=32 \
  precision=16 \
  n_gpus=1 \
  context_prior=false \
  relation_loss=false \
  fp_loss=false \
  project_1_2=false \
  project_1_4=false \
  project_1_8=false
