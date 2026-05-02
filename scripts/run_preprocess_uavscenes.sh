#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

INTERVAL="${INTERVAL:-5}"
if [[ "${INTERVAL}" != "1" && "${INTERVAL}" != "5" ]]; then
  echo "ERROR: INTERVAL must be 1 or 5. Got: ${INTERVAL}" >&2
  exit 2
fi

CONFIG="${CONFIG:-${PROJECT_ROOT}/preprocessing/configs/interval${INTERVAL}.yaml}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/data/index/manifest_interval${INTERVAL}.parquet}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/data/processed/interval${INTERVAL}}"
export CONFIG MANIFEST OUT_ROOT

bash "${PROJECT_ROOT}/scripts/run_preprocess_interval${INTERVAL}.sh"
