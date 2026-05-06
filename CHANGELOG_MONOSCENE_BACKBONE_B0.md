# MonoScene EfficientNet backbone memory fix

## Problem

Training could still reach ~23GB CUDA memory even with `INPUT_IMAGE_HW='[160,224]'` because the original MonoScene RGB network was hard-coded to `tf_efficientnet_b7_ns`.

The failure occurred inside the EfficientNet encoder, not during preprocessing or NPZ loading. This means reducing image resolution alone is not enough.

## Changes

- Added configurable RGB backbone support to `monoscene/models/unet2d.py`.
- Added channel metadata for EfficientNet B0-B7 decoder skip connections.
- Changed the default RGB backbone to `tf_efficientnet_b0_ns`.
- Added `rgb_backbone`, `rgb_pretrained`, and `freeze_rgb_encoder` config keys.
- Added `scripts/train_rgb_monoscene_interval5_4090_b0.sh`.
- Updated `scripts/train_rgb_monoscene_interval5_4090.sh` to expose backbone and feature-width overrides.
- Changed safe training defaults:
  - `rgb_backbone=tf_efficientnet_b0_ns`
  - `feature=32`
  - `context_prior=false`
  - `relation_loss=false`
  - `fp_loss=false`
  - `project_1_2=false`
  - `project_1_4=false`
  - `project_1_8=false`

## Recommended first command

```bash
cd /root/Tuan/uavssc_project
bash scripts/train_rgb_monoscene_interval5_4090_b0.sh
```

## Try B4 only after B0 trains

```bash
cd /root/Tuan/uavssc_project
RGB_BACKBONE=tf_efficientnet_b4_ns INPUT_IMAGE_HW='[320,384]' FEATURE=32 bash scripts/train_rgb_monoscene_interval5_4090_b0.sh
```

## Emergency memory mode

```bash
cd /root/Tuan/uavssc_project
RGB_BACKBONE=tf_efficientnet_b0_ns INPUT_IMAGE_HW='[160,224]' FEATURE=16 FREEZE_RGB_ENCODER=true bash scripts/train_rgb_monoscene_interval5_4090_b0.sh
```
