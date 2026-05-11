#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$PROJECT_ROOT/training/lidar_lmscnet_style"
CKPT="${CKPT:-$PROJECT_ROOT/checkpoints/lidar_lmscnet/interval5_lmscnet_style_debug/checkpoints/best.pt}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
cd "$ROOT"
python eval.py --config "${CONFIG:-configs/uav_interval5_4090.yaml}" --checkpoint "$CKPT"
