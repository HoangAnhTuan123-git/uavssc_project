
# UAVScenes multimodal preprocessing starter

This bundle turns **UAVScenes** into three SSC-ready exports built on the same world-aligned target construction:

- **RGB-only** export for **MonoScene / CGFormer / VoxFormer-style** training
- **LiDAR-only** export for **LMSCNet / SCPNet-style** training
- **RGB+LiDAR fusion** export for a shared local voxel lattice

## Core design

The shared preprocessing backbone is:

1. inspect raw folders
2. freeze taxonomy
3. build a manifest
4. validate LiDAR ↔ label row pairing
5. audit camera–LiDAR projection
6. fuse a **scene-level sparse semantic/free-space vote map**
7. export **per-frame local voxel crops** in the format needed by each modality

The **ground-truth construction** is shared across all three methods.

The **final exported `.npz` schema** is different by modality.

## Important fixes already included

This bundle patches several issues that matter for UAV SSC:

- manifest building now respects `use_interval` and folder hints in `configs/default.yaml`
- label/image/LiDAR discovery is less likely to mix interval1 / interval5 or label images
- free-space ray-casting uses **all LiDAR returns**, not only semantically valid points
- local crop export uses a **camera-focus / local-ground** anchor instead of blindly centering under the drone
- a new `camera_forward_axis` option lets you override or auto-select the camera optical axis
- `11_overlay_npz_alignment.py` lets you check whether exported voxel targets align with the RGB image

## Recommended command order

```bash
python scripts/00_inspect_dataset.py --data-root /path/to/UAVScenes
python scripts/01_build_taxonomy.py --output artifacts/taxonomy_uavssc.json
python scripts/02_build_manifest.py --data-root /path/to/UAVScenes --config configs/default.yaml --output artifacts/manifest.parquet
python scripts/08_validate_label_pairing.py --manifest artifacts/manifest.parquet
python scripts/03_projection_debug.py --data-root /path/to/UAVScenes --manifest artifacts/manifest.parquet --output-dir artifacts/proj_debug --max-samples 20
python scripts/09_build_scene_boxes.py --manifest artifacts/manifest.parquet --output artifacts/scene_boxes.json
python scripts/04_fuse_global_semantic_cloud.py --data-root /path/to/UAVScenes --manifest artifacts/manifest.parquet --config configs/default.yaml --output artifacts/global_voxel_votes --ext-mode cam_from_lidar
python scripts/05_build_scene_voxel_grid.py --input artifacts/global_voxel_votes --config configs/default.yaml --output artifacts/scene_voxel_maps
```

## Exporters

### RGB-only
```bash
python scripts/10_export_rgb_ssc.py   --manifest artifacts/manifest.parquet   --sparse-votes artifacts/global_voxel_votes   --config configs/default.yaml   --output artifacts/rgb_ssc_npz
```

### LiDAR-only
```bash
python scripts/10_export_lidar_ssc.py   --manifest artifacts/manifest.parquet   --sparse-votes artifacts/global_voxel_votes   --data-root /path/to/UAVScenes   --config configs/default.yaml   --output artifacts/lidar_ssc_npz   --ext-mode cam_from_lidar
```

### RGB + LiDAR fusion
```bash
python scripts/10_export_fusion_ssc.py   --manifest artifacts/manifest.parquet   --sparse-votes artifacts/global_voxel_votes   --data-root /path/to/UAVScenes   --config configs/default.yaml   --output artifacts/fusion_ssc_npz   --ext-mode cam_from_lidar
```

## Alignment audit after export

For RGB-only or fusion exports, run:

```bash
python scripts/11_overlay_npz_alignment.py   --input-root artifacts/rgb_ssc_npz   --output-dir artifacts/rgb_overlays   --max-files 100
```

If overlays are consistently shifted, try changing:

```yaml
local_grid:
  camera_forward_axis: auto
```

to one of:

```yaml
+x
-x
+y
-y
+z
-z
```

and regenerate the exporter outputs.

## What each exporter stores

### `10_export_rgb_ssc.py`
Useful for MonoScene / CGFormer / VoxFormer-style models.

Stored keys include:

- `img_path`
- `cam_k`, `cam_E`, `T_world_cam`
- `vox_origin`, `voxel_size`, `scene_size_m`
- `target`, `target_1_8`
- `projected_pix_1`, `fov_mask_1`, `pix_z_1`
- `projected_pix_2`, `fov_mask_2`, `pix_z_2`
- `CP_mega_matrix`, `frustums_masks`, `frustums_class_dists`
- crop/debug metadata such as `look_axis_name` and `focus_shift_xy_m`

### `10_export_lidar_ssc.py`
Useful for dense LiDAR baselines and as a bridge toward LMSCNet / SCPNet.

Stored keys include:

- `lidar_path`
- `T_world_lidar`
- `input_occ_lidar`
- `input_density_lidar`
- `input_max_rel_height`
- `input_mean_rel_height`
- `lidar_sparse_coords`, `lidar_sparse_counts`
- `lidar_points_local_xyz`
- `target`, `occ_mask`, `free_mask`, `known_mask`, `sem_label`

### `10_export_fusion_ssc.py`
Useful for mid-level RGB–LiDAR fusion models.

It includes both RGB projection metadata and LiDAR voxel/point features on the **same local grid**.

## Practical notes

- These scripts are meant as a **strong preprocessing scaffold**, not a claim of plug-and-play SOTA training.
- I could not run them on your full UAVScenes tree inside the container because the dataset itself is not available here.
- Before training, always verify the exported overlays visually on random samples from every scene.
