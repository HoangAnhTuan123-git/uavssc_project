# UAVSSC Training Suite

This bundle contains separate folders for:

- `rgb_monoscene_official_adapter/`
- `rgb_cgformer_style/`
- `rgb_voxformer_style/`
- `lidar_lmscnet_style/`
- `lidar_scpnet_style/`
- `rgb_lidar_fusion_gate3d/`

## Suggested order

1. `rgb_monoscene_official_adapter/` — quickest official-code baseline for RGB-only
2. `lidar_lmscnet_style/` — lightweight LiDAR sanity-check baseline
3. `lidar_scpnet_style/` — stronger LiDAR-only baseline
4. `rgb_cgformer_style/` — stronger RGB-only baseline
5. `rgb_voxformer_style/` — alternative RGB-only baseline
6. `rgb_lidar_fusion_gate3d/` — main multimodal baseline

## Important note

Only the MonoScene folder is an official-code adapter based on your uploaded repository.  
The CGFormer / VoxFormer / LMSCNet / SCPNet / Fusion folders are **starter implementations** designed to fit the `.npz` files you already exported. They are intended to get you training quickly and give you a clean code base to iterate on.


## Scene-strict split support

The starter method folders were patched to optionally use explicit split files via:

```yaml
data:
  split_files:
    train: data/splits/scene_strict_cv/fold_A/train_samples.txt
    val: data/splits/scene_strict_cv/fold_A/val_samples.txt
    test: data/splits/scene_strict_cv/fold_A/test_samples.txt
```

When `split_files` are provided, they override the older ratio-based file partitioning logic.
