#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
SCENE="${SCENE:-interval5_AMtown01}"
N="${N:-200}"
MANIFEST="data/index/manifest_interval5_${SCENE}_head${N}.parquet"
if [ ! -f "$MANIFEST" ]; then
  SCENE="$SCENE" N="$N" OUT="$MANIFEST" bash scripts/make_manifest_interval5_head200.sh
fi
time python preprocessing/scripts/10_export_fusion_ssc.py \
  --manifest "$MANIFEST" \
  --sparse-votes data/processed/interval5/global_voxel_votes \
  --data-root "${UAVSSC_DATA_ROOT:-data/raw/uavscenes_official}" \
  --config preprocessing/configs/interval5.yaml \
  --output data/processed/interval5/fusion_ssc_npz_test${N}
find data/processed/interval5/fusion_ssc_npz_test${N} -name "*.npz" | wc -l
