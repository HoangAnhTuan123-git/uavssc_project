#!/usr/bin/env bash
set -euo pipefail
mkdir -p external
git clone https://github.com/NVlabs/VoxFormer.git external/official_repo
echo "Cloned official repo into external/official_repo"
