#!/usr/bin/env bash
set -euo pipefail
python eval.py --config configs/default.yaml --checkpoint runs/REPLACE/checkpoints/best.pt
