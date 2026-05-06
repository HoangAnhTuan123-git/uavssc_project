# MonoScene AMP/BCE training fix

## Problem

Training failed during the validation sanity check with:

```text
RuntimeError: torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.
```

The launcher was still forcing `precision=16`, even if the YAML config was edited. The original MonoScene `sem_scal_loss` and `geo_scal_loss` call `binary_cross_entropy` on probabilities after softmax, which PyTorch AMP refuses to autocast.

## Fix

- Added `_bce_to_one_autocast_safe()` in `monoscene/loss/ssc_loss.py`.
- The scalar BCE terms now run in float32 with autocast disabled.
- Changed `scripts/train_rgb_monoscene_interval5_4090.sh` default from `PRECISION=16` to `PRECISION=32`.
- Changed `monoscene/config/uavscenes.yaml` default precision from `16` to `32`.

## Recommended command

```bash
cd /root/Tuan/uavssc_project
PRECISION=32 INPUT_IMAGE_HW='[512,640]' bash scripts/train_rgb_monoscene_interval5_4090.sh
```

If memory is safe and you want to try AMP later:

```bash
cd /root/Tuan/uavssc_project
PRECISION=16 INPUT_IMAGE_HW='[512,640]' bash scripts/train_rgb_monoscene_interval5_4090.sh
```
