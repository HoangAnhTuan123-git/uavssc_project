#!/usr/bin/env bash
set -euo pipefail

# Rebuild only the RGB/fusion NPZ files with resized image projection metadata.
# This does NOT rerun expensive steps 00-09 if manifest and global_voxel_votes already exist.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_ROOT="${UAVSSC_DATA_ROOT:-${PROJECT_ROOT}/data/raw/uavscenes_official}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/preprocessing/configs/interval5.yaml}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/data/index/manifest_interval5.parquet}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/data/processed/interval5}"
EXT_MODE="${EXT_MODE:-cam_from_lidar}"
MAX_DEBUG_SAMPLES="${MAX_DEBUG_SAMPLES:-30}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: manifest not found: ${MANIFEST}" >&2
  echo "Run: bash scripts/run_preprocess_interval5.sh" >&2
  exit 2
fi

if [[ ! -d "${OUT_ROOT}/global_voxel_votes" ]]; then
  echo "ERROR: sparse votes folder not found: ${OUT_ROOT}/global_voxel_votes" >&2
  echo "Run: bash scripts/run_preprocess_interval5.sh" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}"

# Keep old NPZs as backup unless the caller disables it.
if [[ "${BACKUP_OLD_NPZ:-1}" == "1" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  if [[ -d "${OUT_ROOT}/rgb_ssc_npz" ]]; then
    mv "${OUT_ROOT}/rgb_ssc_npz" "${OUT_ROOT}/rgb_ssc_npz_fullres_backup_${TS}"
  fi
  if [[ -d "${OUT_ROOT}/fusion_ssc_npz" ]]; then
    mv "${OUT_ROOT}/fusion_ssc_npz" "${OUT_ROOT}/fusion_ssc_npz_fullres_backup_${TS}"
  fi
fi

mkdir -p "${OUT_ROOT}/rgb_ssc_npz" "${OUT_ROOT}/fusion_ssc_npz"

echo "Re-exporting RGB NPZ with resized image lattice from config: ${CONFIG}"
python preprocessing/scripts/10_export_rgb_ssc.py \
  --manifest "${MANIFEST}" \
  --sparse-votes "${OUT_ROOT}/global_voxel_votes" \
  --config "${CONFIG}" \
  --output "${OUT_ROOT}/rgb_ssc_npz"

echo "Re-exporting RGB+LiDAR fusion NPZ with resized image lattice"
python preprocessing/scripts/10_export_fusion_ssc.py \
  --manifest "${MANIFEST}" \
  --sparse-votes "${OUT_ROOT}/global_voxel_votes" \
  --data-root "${DATA_ROOT}" \
  --config "${CONFIG}" \
  --output "${OUT_ROOT}/fusion_ssc_npz" \
  --ext-mode "${EXT_MODE}"

echo "Regenerating resized projection overlays"
python preprocessing/scripts/11_overlay_npz_alignment.py \
  --input-root "${OUT_ROOT}/rgb_ssc_npz" \
  --output-dir "${OUT_ROOT}/rgb_overlays" \
  --max-files "${MAX_DEBUG_SAMPLES}"
python preprocessing/scripts/11_overlay_npz_alignment.py \
  --input-root "${OUT_ROOT}/fusion_ssc_npz" \
  --output-dir "${OUT_ROOT}/fusion_overlays" \
  --max-files "${MAX_DEBUG_SAMPLES}"

echo "Done. New resized NPZ roots:"
echo "  ${OUT_ROOT}/rgb_ssc_npz"
echo "  ${OUT_ROOT}/fusion_ssc_npz"
