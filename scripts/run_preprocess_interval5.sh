#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_ROOT="${UAVSSC_DATA_ROOT:-${PROJECT_ROOT}/data/raw/uavscenes_official}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/preprocessing/configs/interval5.yaml}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/data/index/manifest_interval5.parquet}"
TAXONOMY="${TAXONOMY:-${PROJECT_ROOT}/data/index/taxonomy_uavssc.json}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/data/processed/interval5}"
MAX_DEBUG_SAMPLES="${MAX_DEBUG_SAMPLES:-30}"
MAX_RAYS_PER_FRAME="${MAX_RAYS_PER_FRAME:-5000}"
EXT_MODE="${EXT_MODE:-cam_from_lidar}"

mkdir -p "${PROJECT_ROOT}/data/index" "${OUT_ROOT}"

echo "DATA_ROOT=${DATA_ROOT}"
echo "CONFIG=${CONFIG}"
echo "OUT_ROOT=${OUT_ROOT}"

echo "[1/11] Inspect dataset and CAM/LiDAR label pairing"
python preprocessing/scripts/00_inspect_dataset.py --data-root "${DATA_ROOT}" --config "${CONFIG}"

echo "[2/11] Build/edit taxonomy template"
python preprocessing/scripts/01_build_taxonomy.py --output "${TAXONOMY}"

echo "[3/11] Build manifest with interval5_CAM_label paths"
python preprocessing/scripts/02_build_manifest.py --data-root "${DATA_ROOT}" --config "${CONFIG}" --output "${MANIFEST}"

echo "[4/11] Validate LiDAR point/label row alignment"
python preprocessing/scripts/08_validate_label_pairing.py --manifest "${MANIFEST}"

echo "[5/11] Validate CAM label-id/color images"
python preprocessing/scripts/12_validate_cam_labels.py --manifest "${MANIFEST}" --output "${OUT_ROOT}/cam_label_report.csv"

echo "[6/11] Camera-LiDAR projection QC overlays"
python preprocessing/scripts/03_projection_debug.py --data-root "${DATA_ROOT}" --manifest "${MANIFEST}" --output-dir "${OUT_ROOT}/proj_debug" --max-samples "${MAX_DEBUG_SAMPLES}"

echo "[7/11] Scene box audit"
python preprocessing/scripts/09_build_scene_boxes.py --manifest "${MANIFEST}" --output "${OUT_ROOT}/scene_boxes.json"

echo "[8/11] Fuse global semantic/free-space sparse voxel votes"
python preprocessing/scripts/04_fuse_global_semantic_cloud.py --data-root "${DATA_ROOT}" --manifest "${MANIFEST}" --config "${CONFIG}" --output "${OUT_ROOT}/global_voxel_votes" --ext-mode "${EXT_MODE}" --max-rays-per-frame "${MAX_RAYS_PER_FRAME}"

echo "[9/11] Resolve scene-level sparse voxel maps"
python preprocessing/scripts/05_build_scene_voxel_grid.py --input "${OUT_ROOT}/global_voxel_votes" --config "${CONFIG}" --output "${OUT_ROOT}/scene_voxel_maps"

echo "[10/11] Export RGB-only, LiDAR-only, and RGB+LiDAR NPZ samples"
python preprocessing/scripts/10_export_rgb_ssc.py --manifest "${MANIFEST}" --sparse-votes "${OUT_ROOT}/global_voxel_votes" --config "${CONFIG}" --output "${OUT_ROOT}/rgb_ssc_npz"
python preprocessing/scripts/10_export_lidar_ssc.py --manifest "${MANIFEST}" --sparse-votes "${OUT_ROOT}/global_voxel_votes" --data-root "${DATA_ROOT}" --config "${CONFIG}" --output "${OUT_ROOT}/lidar_ssc_npz" --ext-mode "${EXT_MODE}"
python preprocessing/scripts/10_export_fusion_ssc.py --manifest "${MANIFEST}" --sparse-votes "${OUT_ROOT}/global_voxel_votes" --data-root "${DATA_ROOT}" --config "${CONFIG}" --output "${OUT_ROOT}/fusion_ssc_npz" --ext-mode "${EXT_MODE}"

echo "[11/11] NPZ alignment overlays: RGB panel plus CAM-label panel when available"
python preprocessing/scripts/11_overlay_npz_alignment.py --input-root "${OUT_ROOT}/rgb_ssc_npz" --output-dir "${OUT_ROOT}/rgb_overlays" --max-files "${MAX_DEBUG_SAMPLES}"
python preprocessing/scripts/11_overlay_npz_alignment.py --input-root "${OUT_ROOT}/fusion_ssc_npz" --output-dir "${OUT_ROOT}/fusion_overlays" --max-files "${MAX_DEBUG_SAMPLES}"

echo "Done. Key outputs:"
echo "  Manifest:             ${MANIFEST}"
echo "  Sparse votes:         ${OUT_ROOT}/global_voxel_votes"
echo "  RGB NPZ:              ${OUT_ROOT}/rgb_ssc_npz"
echo "  LiDAR NPZ:            ${OUT_ROOT}/lidar_ssc_npz"
echo "  Fusion NPZ:           ${OUT_ROOT}/fusion_ssc_npz"
echo "  Alignment overlays:   ${OUT_ROOT}/rgb_overlays and ${OUT_ROOT}/fusion_overlays"
