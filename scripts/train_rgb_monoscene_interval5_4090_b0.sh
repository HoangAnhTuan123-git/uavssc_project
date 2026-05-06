#!/usr/bin/env bash
set -euo pipefail

# Memory-safe MonoScene RGB-only UAVScenes launcher for RTX 4090 24GB.
# This uses EfficientNet-B0 instead of original B7, disables the heavy context-prior path,
# and uses a smaller feature width. It is the recommended first successful training run.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONOSCENE_ROOT="${PROJECT_ROOT}/training/rgb_monoscene_official_adapter/MonoScene"
NPZ_ROOT="${NPZ_ROOT:-${PROJECT_ROOT}/data/processed/interval5/rgb_ssc_npz}"
RAW_ROOT="${UAVSSC_DATA_ROOT:-${PROJECT_ROOT}/data/raw/uavscenes_official}"
RGB_BACKBONE="${RGB_BACKBONE:-tf_efficientnet_b0_ns}"
INPUT_IMAGE_HW="${INPUT_IMAGE_HW:-[320,384]}"
FEATURE="${FEATURE:-32}"
PRECISION="${PRECISION:-16}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/checkpoints/rgb_monoscene/interval5_${RGB_BACKBONE}_f${FEATURE}}"

cd "${MONOSCENE_ROOT}"
export PYTHONPATH="${MONOSCENE_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:64}"

python - <<PY
import glob
import os
root = ${NPZ_ROOT@Q}
files = glob.glob(os.path.join(root, "*.npz")) + glob.glob(os.path.join(root, "*", "*.npz"))
if not files:
    raise SystemExit(f"ERROR: no NPZ files found under {root}. Run preprocessing or set NPZ_ROOT.")
print(f"Found {len(files)} NPZ files under {root}")
PY

python -c "import monoscene; print('monoscene import OK')"

python -m monoscene.scripts.train_uavscenes \
  uav_preprocess_root="${NPZ_ROOT}" \
  uav_data_root="${RAW_ROOT}" \
  uav_logdir="${LOG_ROOT}" \
  rgb_backbone="${RGB_BACKBONE}" \
  rgb_pretrained="${RGB_PRETRAINED:-true}" \
  freeze_rgb_encoder="${FREEZE_RGB_ENCODER:-false}" \
  input_image_hw="${INPUT_IMAGE_HW}" \
  feature="${FEATURE}" \
  n_gpus=1 \
  batch_size=1 \
  num_workers_per_gpu="${NUM_WORKERS:-2}" \
  max_epochs="${MAX_EPOCHS:-30}" \
  precision="${PRECISION}" \
  load_pretrained="${LOAD_PRETRAINED:-false}" \
  context_prior="${CONTEXT_PRIOR:-false}" \
  relation_loss="${RELATION_LOSS:-false}" \
  fp_loss="${FP_LOSS:-false}" \
  project_1_2="${PROJECT_1_2:-false}" \
  project_1_4="${PROJECT_1_4:-false}" \
  project_1_8="${PROJECT_1_8:-false}"
