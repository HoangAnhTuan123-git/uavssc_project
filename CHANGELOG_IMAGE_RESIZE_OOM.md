# Image resize / RTX 4090 OOM fix

## Problem

Original UAVScenes RGB frames are usually `2048 x 2448`. The first MonoScene adapter loaded those full-resolution images during training, which can cause CUDA out-of-memory on a 24GB RTX 4090.

## Fix

The project now supports a safe resized training image lattice. The default is:

```yaml
input_image_hw: [640, 768]
```

For MonoScene, this resize is applied online in the DataLoader. Existing full-resolution `.npz` files can still be used because the loader rescales `projected_pix_*` from the projection lattice to the resized training image.

For future preprocessing/export, `10_export_rgb_ssc.py` and `10_export_fusion_ssc.py` also support:

```yaml
image_export:
  enabled: true
  resize_hw: [640, 768]
```

This exports projection tensors directly on the resized image lattice.

## New scripts

```bash
scripts/train_rgb_monoscene_interval5_4090.sh
scripts/debug_monoscene_one_batch.sh
scripts/reexport_interval5_resized_rgb_npz.sh
```

## Do I need to rerun preprocessing?

No for training. Use `scripts/train_rgb_monoscene_interval5_4090.sh`; it resizes online and works with existing `.npz` files.

Optional cleanup: rerun only the export stage with `scripts/reexport_interval5_resized_rgb_npz.sh`. This uses the existing manifest and global voxel votes. It does not rerun the expensive fusion/global voxel steps.
