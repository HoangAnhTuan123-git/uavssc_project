#!/usr/bin/env bash
set -euo pipefail
mkdir -p trained_models
curl -L -o trained_models/monoscene_kitti.ckpt https://www.rocq.inria.fr/rits_files/computer-vision/monoscene/monoscene_kitti.ckpt
curl -L -o trained_models/monoscene_nyu.ckpt   https://www.rocq.inria.fr/rits_files/computer-vision/monoscene/monoscene_nyu.ckpt
echo "Downloaded pretrained checkpoints into trained_models/"
