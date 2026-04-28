Place the extracted UAVScenes dataset here.

Expected high-level structure after extraction:
- `interval1_CAM_LIDAR/`
- `interval1_CAM_label/`
- `interval1_LIDAR_label/`
- `interval5_CAM_LIDAR/`
- `interval5_CAM_label/`
- `interval5_LIDAR_label/`
- `terra_3dmap_pointcloud_mesh/`
- `cmap.py`
- `calibration_results.py`

Keep this folder read-only once extraction is complete.
All derived outputs should go under `data/processed/`.
