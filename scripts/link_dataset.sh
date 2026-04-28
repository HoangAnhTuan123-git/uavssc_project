#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /absolute/path/to/extracted/UAVScenes"
  exit 1
fi
SRC="$1"
DST="data/raw/uavscenes_official"
mkdir -p "$(dirname "$DST")"
rm -rf "$DST"
ln -s "$SRC" "$DST"
echo "Linked $SRC -> $DST"
