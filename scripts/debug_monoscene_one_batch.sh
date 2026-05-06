#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONOSCENE_ROOT="${PROJECT_ROOT}/training/rgb_monoscene_official_adapter/MonoScene"
NPZ_ROOT="${NPZ_ROOT:-${PROJECT_ROOT}/data/processed/interval5/rgb_ssc_npz}"
RAW_ROOT="${UAVSSC_DATA_ROOT:-${PROJECT_ROOT}/data/raw/uavscenes_official}"
INPUT_IMAGE_HW="${INPUT_IMAGE_HW:-[640,768]}"

cd "${MONOSCENE_ROOT}"
export PYTHONPATH="${MONOSCENE_ROOT}:${PYTHONPATH:-}"

python - <<PY
from monoscene.data.uavscenes.uav_dataset import UAVScenesDataset
root = ${NPZ_ROOT@Q}
raw = ${RAW_ROOT@Q}
hw_txt = ${INPUT_IMAGE_HW@Q}
ds = UAVScenesDataset(
    split='train',
    preprocess_root=root,
    data_root=raw,
    input_image_hw=hw_txt,
    split_ratios=(0.7, 0.15, 0.15),
)
item = ds[0]
print('sample:', item['frame_id'], item['sequence'])
print('img tensor:', tuple(item['img'].shape))
print('target:', item['target'].shape, item['target'].dtype)
print('projection shape hw:', item.get('projection_shape_hw'))
for k in sorted([x for x in item.keys() if x.startswith('projected_pix_')]):
    uv = item[k]
    print(k, uv.shape, 'x range', int(uv[:,0].min()), int(uv[:,0].max()), 'y range', int(uv[:,1].min()), int(uv[:,1].max()))
PY
