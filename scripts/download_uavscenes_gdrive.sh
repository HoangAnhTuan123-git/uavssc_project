#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${1:-${PROJECT_ROOT}/data/raw/uavscenes_official}"
GDRIVE_URL="${2:-https://drive.google.com/drive/folders/1HSJWc5qmIKLdpaS8w8pqrWch4F9MHIeN}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${OUT_ROOT}" "${OUT_ROOT}/_downloads"

echo "Downloading UAVScenes from Google Drive folder: ${GDRIVE_URL}"
echo "Destination: ${OUT_ROOT}"

if ! command -v gdown >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pip install --upgrade gdown
fi

gdown --folder --continue --remaining-ok "${GDRIVE_URL}" -O "${OUT_ROOT}/_downloads"

shopt -s nullglob globstar
ZIP_COUNT=0
for z in "${OUT_ROOT}"/_downloads/*.zip "${OUT_ROOT}"/_downloads/**/*.zip; do
  ZIP_COUNT=$((ZIP_COUNT + 1))
  echo "Unzipping: ${z}"
  unzip -n "${z}" -d "${OUT_ROOT}"
done

if [ "${ZIP_COUNT}" -eq 0 ]; then
  echo "No zip files found under ${OUT_ROOT}/_downloads. If gdown downloaded extracted folders, keep them there or move them under ${OUT_ROOT}."
else
  echo "Extracted ${ZIP_COUNT} zip files into ${OUT_ROOT}."
fi

find "${OUT_ROOT}" -maxdepth 2 -type d | sort | sed -n '1,120p'
echo "Set: export UAVSSC_DATA_ROOT=${OUT_ROOT}"
