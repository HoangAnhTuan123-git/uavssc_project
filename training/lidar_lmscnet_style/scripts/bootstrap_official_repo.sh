#!/usr/bin/env bash
set -euo pipefail
mkdir -p external
git clone https://github.com/astra-vision/LMSCNet.git external/official_repo
echo "Cloned official repo into external/official_repo"
