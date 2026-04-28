#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 METHOD PREPROCESS_ROOT [DATA_ROOT]"
  echo "METHOD one of: rgb_cgformer_style rgb_voxformer_style lidar_lmscnet_style lidar_scpnet_style rgb_lidar_fusion_gate3d"
  exit 1
fi

METHOD="$1"
PREPROCESS_ROOT="$2"
DATA_ROOT="${3:-data/raw/uavscenes_official}"

for FOLD in fold_A fold_B fold_C fold_D; do
  CFG="training/${METHOD}/configs/default.yaml"
  TMP="logs/stdout/${METHOD}_${FOLD}.yaml"
  python - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("${CFG}").read_text())
cfg["data"]["preprocess_root"] = "${PREPROCESS_ROOT}"
cfg["data"]["data_root"] = "${DATA_ROOT}"
cfg["data"]["split_files"] = {
    "train": f"data/splits/scene_strict_cv/${FOLD}/train_samples.txt",
    "val": f"data/splits/scene_strict_cv/${FOLD}/val_samples.txt",
    "test": f"data/splits/scene_strict_cv/${FOLD}/test_samples.txt",
}
cfg["output_dir"] = f"checkpoints/${METHOD}/${FOLD}"
Path("${TMP}").parent.mkdir(parents=True, exist_ok=True)
Path("${TMP}").write_text(yaml.safe_dump(cfg, sort_keys=False))
print("Wrote", "${TMP}")
PY
  python "training/${METHOD}/train.py" --config "${TMP}" | tee "logs/stdout/${METHOD}_${FOLD}.log"
done
