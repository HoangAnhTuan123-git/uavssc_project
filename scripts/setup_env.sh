#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[1/4] Python executable: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "[2/4] Upgrading build tooling"
"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel

echo "[3/4] Installing preprocessing package and runtime dependencies"
"${PYTHON_BIN}" -m pip install \
  -e preprocessing \
  numpy scipy pandas pyarrow pyyaml tqdm matplotlib \
  opencv-python-headless open3d trimesh imageio pillow scikit-learn \
  rich orjson gdown tensorboard

echo "[4/4] Checking PyTorch/CUDA from the selected container"
"${PYTHON_BIN}" - <<'PY'
try:
    import torch
    print('torch:', torch.__version__)
    print('cuda available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('cuda runtime:', torch.version.cuda)
        print('gpu:', torch.cuda.get_device_name(0))
except Exception as exc:
    print('Torch import/check failed:', repr(exc))
PY

echo "Done. Next: export UAVSSC_DATA_ROOT=${PROJECT_ROOT}/data/raw/uavscenes_official"
