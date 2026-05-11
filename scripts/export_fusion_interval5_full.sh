#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
time python preprocessing/scripts/10_export_fusion_ssc.py \
  --manifest data/index/manifest_interval5.parquet \
  --sparse-votes data/processed/interval5/global_voxel_votes \
  --data-root "${UAVSSC_DATA_ROOT:-data/raw/uavscenes_official}" \
  --config preprocessing/configs/interval5.yaml \
  --output data/processed/interval5/fusion_ssc_npz
find data/processed/interval5/fusion_ssc_npz -name "*.npz" | wc -l
