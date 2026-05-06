#!/usr/bin/env bash
set -euo pipefail

# Evaluate a UAVScenes MonoScene checkpoint with the same lightweight settings used for B0 training.
# Usage:
#   EVAL_CHECKPOINT=/absolute/path/to/last.ckpt bash scripts/eval_rgb_monoscene_interval5_4090_b0.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONOSCENE_ROOT="${PROJECT_ROOT}/training/rgb_monoscene_official_adapter/MonoScene"
NPZ_ROOT="${NPZ_ROOT:-${PROJECT_ROOT}/data/processed/interval5/rgb_ssc_npz}"
RAW_ROOT="${UAVSSC_DATA_ROOT:-${PROJECT_ROOT}/data/raw/uavscenes_official}"
RGB_BACKBONE="${RGB_BACKBONE:-tf_efficientnet_b0_ns}"
INPUT_IMAGE_HW="${INPUT_IMAGE_HW:-[320,384]}"
FEATURE="${FEATURE:-32}"
PRECISION="${PRECISION:-16}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"

if [[ -z "${EVAL_CHECKPOINT:-}" ]]; then
  echo "ERROR: set EVAL_CHECKPOINT=/path/to/checkpoint.ckpt" >&2
  echo "Example:" >&2
  echo "  EVAL_CHECKPOINT=${PROJECT_ROOT}/checkpoints/rgb_monoscene/interval5_${RGB_BACKBONE}_f${FEATURE}/uav_monoscene_uavscenes_1_sceneall_fs128_128_32_bs1_lr0.0001/checkpoints/last.ckpt bash $0" >&2
  exit 2
fi
if [[ ! -f "${EVAL_CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint not found: ${EVAL_CHECKPOINT}" >&2
  exit 2
fi

cd "${MONOSCENE_ROOT}"
export PYTHONPATH="${MONOSCENE_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:64}"

python -m monoscene.scripts.eval_uavscenes \
  eval_checkpoint_path="${EVAL_CHECKPOINT}" \
  eval_split="${EVAL_SPLIT}" \
  uav_preprocess_root="${NPZ_ROOT}" \
  uav_data_root="${RAW_ROOT}" \
  input_image_hw="${INPUT_IMAGE_HW}" \
  rgb_backbone="${RGB_BACKBONE}" \
  rgb_pretrained="${RGB_PRETRAINED:-false}" \
  freeze_rgb_encoder="${FREEZE_RGB_ENCODER:-false}" \
  feature="${FEATURE}" \
  n_gpus=1 \
  batch_size=1 \
  num_workers_per_gpu="${NUM_WORKERS:-2}" \
  precision="${PRECISION}" \
  load_pretrained=false \
  context_prior="${CONTEXT_PRIOR:-false}" \
  relation_loss="${RELATION_LOSS:-false}" \
  fp_loss="${FP_LOSS:-false}" \
  project_1_2="${PROJECT_1_2:-false}" \
  project_1_4="${PROJECT_1_4:-false}" \
  project_1_8="${PROJECT_1_8:-false}"
