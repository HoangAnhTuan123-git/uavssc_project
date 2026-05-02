# Interval1 CAM-label update changelog

## Added

- `MASTER_INSTRUCTIONS.md`: full project command guide for VS Code/GPU container, Google Drive download, dataset placement, interval1 preprocessing, expected outputs, and training commands.
- `scripts/setup_env.sh`: installs preprocessing package and runtime dependencies without forcing a PyTorch upgrade.
- `scripts/download_uavscenes_gdrive.sh`: downloads the owner-shared UAVScenes Google Drive folder and extracts zip files.
- `scripts/run_preprocess_interval1.sh`: one-command interval1 preprocessing pipeline.
- `preprocessing/configs/interval1.yaml`: interval1 default configuration with CAM-label discovery/export settings.
- `preprocessing/configs/interval5.yaml`: retained interval5 configuration for debugging/keyframe experiments.
- `preprocessing/scripts/12_validate_cam_labels.py`: validates paired `cam_label_id_path` and `cam_label_rgb_path` against RGB images.

## Updated

- `preprocessing/src/uavssc/io.py`
  - Added CAM label PNG discovery for `interval1_CAM_label` / `interval5_CAM_label`.
  - Added CAM label-id reader and CAM label-color reader.
  - Added color-mask to raw-id fallback using the UAVScenes color map.
  - Improved scene discovery so label folders and Terra folders are not treated as scenes.
  - Improved calibration scene-prefix handling for names like `interval1_AMtown01` and `interval1_HKairport_GNSS01`.

- `preprocessing/src/uavssc/manifest.py`
  - Added `cam_label_id_path` and `cam_label_rgb_path` columns.
  - Supports sibling label folders instead of requiring labels inside each scene folder.
  - Matches CAM labels by timestamp to image frames.
  - Supports both dual timestamp LiDAR labels and single timestamp label filenames.

- `preprocessing/scripts/00_inspect_dataset.py`
  - Now accepts `--config` and reports manifest-level CAM/LiDAR label pairing counts.

- `preprocessing/scripts/10_export_rgb_ssc.py`
  - Stores CAM label paths in exported RGB-only NPZ files.
  - Can optionally embed CAM label-id/color arrays when enabled in config.

- `preprocessing/scripts/10_export_fusion_ssc.py`
  - Stores CAM label paths in exported RGB+LiDAR fusion NPZ files.
  - Can optionally embed CAM label-id/color arrays when enabled in config.

- `preprocessing/scripts/11_overlay_npz_alignment.py`
  - Adds optional side-by-side CAM-label panel for alignment debugging.

- `training/shared/uavssc_trainkit/.../data.py` and method-local copies
  - RGB and fusion datasets expose `cam_label_id_path`.
  - Optional `return_cam_label=True` loads the paired CAM label-id mask for 2D auxiliary supervision experiments.

- `scripts/make_scene_registry.py`
  - Supports both official wrapper layout and directly extracted `interval1_<scene>` folders.

- Training requirements
  - Relaxed starter requirement from `torch>=2.1` to `torch>=2.0` so the recommended PyTorch 2.0.1 + CUDA 11.7 template is usable.

## Important behavior

- CAM labels are **not** treated as the 3D SSC target. They are paired metadata/optional auxiliary supervision. The 3D SSC target is still the world-aligned local voxel grid produced from LiDAR labels, poses, calibration, free-space ray casting, and semantic voxel fusion.
- Exported `.npz` files must be checked visually. The projected occupied voxels should align with the paired RGB image and CAM label panel before training.

See also `CHANGELOG_INTERVAL5.md` for the later interval5-first workflow update.
