#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOX_ROOT="$PROJECT_ROOT/training/rgb_voxformer_style"
CKPT="${CKPT:-$PROJECT_ROOT/checkpoints/rgb_voxformer/interval5_voxformer_style_stable_lr1e5/checkpoints/best.pt}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$VOX_ROOT/src:${PYTHONPATH:-}"
cd "$VOX_ROOT"
python eval.py --config "${CONFIG:-configs/uav_interval5_4090_stable.yaml}" --checkpoint "$CKPT"
