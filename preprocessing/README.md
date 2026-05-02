# UAVScenes SSC preprocessing

Use `../MASTER_INSTRUCTIONS.md` for the complete command sequence. The current default workflow is now **interval5** because it is the practical keyframe subset for fast SSC iteration and lower storage use. Interval1 is still supported through `preprocessing/configs/interval1.yaml` and `scripts/run_preprocess_interval1.sh`.

Supported UAVScenes layouts:

```text
data/raw/uavscenes_official/
  interval5_AMtown01/
    interval5_CAM/
    interval5_LIDAR/
    sampleinfos_interpolated.json
  interval5_AMtown02/
  interval5_CAM_label/
  interval5_LIDAR_label/
  terra_3dmap_pointcloud_mesh/
  calibration_results.py
  cmap.py
```

or the official wrapper style:

```text
data/raw/uavscenes_official/
  interval5_CAM_LIDAR/
  interval5_CAM_label/
  interval5_LIDAR_label/
  terra_3dmap_pointcloud_mesh/
```

Main command:

```bash
export UAVSSC_DATA_ROOT=$PWD/data/raw/uavscenes_official
bash scripts/run_preprocess_interval5.sh
```

The manifest and RGB/fusion exporters support `interval5_CAM_label` folders with paired `label_id` and `label_color` PNG masks. Exported RGB/fusion NPZ files store `cam_label_id_path` and `cam_label_rgb_path` by default, and `preprocessing/scripts/12_validate_cam_labels.py` checks image/mask timestamp and shape alignment.

Important checks before training:

1. `data/index/manifest_interval5.parquet` should have non-empty `img_path`, `lidar_path`, `lidar_label_id_path`, and, when CAM labels are downloaded, `cam_label_id_path`.
2. Projection debug overlays in `data/processed/interval5/proj_debug/` should align LiDAR/voxel evidence with the RGB image.
3. NPZ overlays in `data/processed/interval5/rgb_overlays/` and `data/processed/interval5/fusion_overlays/` should project the generated 3D target back onto the correct 2D scene.

If overlays are shifted, do not fix it by manually translating voxels first. Check extrinsic direction, camera forward axis, timestamp pairing, and local-grid anchor mode first.
