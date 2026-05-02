# UAVScenes SSC Master Instructions — interval1 CAM_label version

This project prepares UAVScenes for 3D Semantic Scene Completion (SSC) with three export modes:

1. RGB-only SSC NPZs
2. LiDAR-only SSC NPZs
3. RGB+LiDAR fusion SSC NPZs

The code in this folder is updated for the newer UAVScenes layout where `interval1_CAM_label` contains two camera semantic mask types:

- `label_id`: PNG mask with raw class id values per pixel
- `label_color`: PNG mask with the official class color per pixel

The file stem is the image timestamp, for example:

```text
1658137057.641204937.png
```

The manifest builder now pairs these CAM labels to RGB frames by timestamp.

---

## 0. Recommended GPU container template

Use this template first:

```text
PyTorch 2.0.1 + CUDA 11.7
```

Reason: it is the safer choice if you later adapt MonoScene/KITTI pretrained code or older 3D/SSC repositories. Those projects often depend on older PyTorch, CUDA, PyTorch-Lightning, OpenCV, or compiled CUDA extensions.

Use `PyTorch 2.9.1 + CUDA 12.8` only when you are running the starter models in this repository and not trying to reproduce old upstream MonoScene/CGFormer/VoxFormer code. The preprocessing code itself is mostly CPU/Python and can run in either container, but the training phase is less fragile on the PyTorch 2.0.1 + CUDA 11.7 template.

Avoid the plain CUDA base image unless you want to install PyTorch manually. Avoid the Kasm desktop image unless you specifically need a browser/desktop GUI inside the container.

---

## 1. Project location in VS Code / container

Recommended location:

```bash
cd /workspace
unzip uavssc_project_interval1_camlabel.zip -d /workspace
cd /workspace/uavssc_project
```

Expected output:

```text
README.md
MASTER_INSTRUCTIONS.md
preprocessing/
training/
scripts/
data/
configs/
```

---

## 2. Install environment

Run from the project root:

```bash
cd /workspace/uavssc_project
bash scripts/setup_env.sh
```

Expected output:

```text
torch: 2.x.x
cuda available: True
cuda runtime: 11.7   # or the CUDA version of your selected template
gpu: <your GPU name>
```

If `cuda available: False`, the container does not see the GPU. Fix the GPU/runtime selection before training.

---

## 3. Download UAVScenes from the shared Google Drive folder

Run:

```bash
cd /workspace/uavssc_project
bash scripts/download_uavscenes_gdrive.sh data/raw/uavscenes_official
export UAVSSC_DATA_ROOT=$PWD/data/raw/uavscenes_official
```

The script uses this Google Drive folder by default:

```text
https://drive.google.com/drive/folders/1HSJWc5qmIKLdpaS8w8pqrWch4F9MHIeN
```

Google Drive can fail because of quota/rate limits. If that happens, download manually and put the extracted folders under:

```text
data/raw/uavscenes_official/
```

---

## 4. Correct dataset placement

After extraction, the important folders should be direct children of:

```text
data/raw/uavscenes_official/
```

Acceptable layout A — official wrapper style:

```text
data/raw/uavscenes_official/
  interval1_CAM_LIDAR/
    AMtown01/
    AMtown02/
    ...
  interval1_CAM_label/
    AMtown01/
    AMtown02/
    ...
  interval1_LIDAR_label/
    AMtown01/
    AMtown02/
    ...
  terra_3dmap_pointcloud_mesh/
  calibration_results.py
  cmap.py
```

Acceptable layout B — extracted per-run folders like your screenshot:

```text
data/raw/uavscenes_official/
  interval1_AMtown01/
  interval1_AMtown02/
  interval1_AMtown03/
  interval1_AMvalley01/
  ...
  interval1_CAM_label/
  interval1_LIDAR_label/
  terra_3dmap_pointcloud_mesh/
  calibration_results.py
  cmap.py
```

Do not move `interval1_CAM_label` into each scene folder. The updated manifest code searches sibling label folders and pairs by timestamp.

If your download creates one extra wrapper such as:

```text
data/raw/uavscenes_official/UAVScenes/interval1_CAM_label
```

move the contents up one level:

```bash
mv data/raw/uavscenes_official/UAVScenes/* data/raw/uavscenes_official/
rmdir data/raw/uavscenes_official/UAVScenes
```

---

## 5. One-command interval1 preprocessing pipeline

Run:

```bash
cd /workspace/uavssc_project
export UAVSSC_DATA_ROOT=$PWD/data/raw/uavscenes_official
bash scripts/run_preprocess_interval1.sh
```

Default configuration:

```text
preprocessing/configs/interval1.yaml
```

Main expected outputs:

```text
data/index/manifest_interval1.parquet

data/processed/interval1/cam_label_report.csv
/data/processed/interval1/proj_debug/
/data/processed/interval1/global_voxel_votes/
/data/processed/interval1/scene_voxel_maps/
/data/processed/interval1/rgb_ssc_npz/
/data/processed/interval1/lidar_ssc_npz/
/data/processed/interval1/fusion_ssc_npz/
/data/processed/interval1/rgb_overlays/
/data/processed/interval1/fusion_overlays/
```

The script runs these steps:

```bash
python preprocessing/scripts/00_inspect_dataset.py \
  --data-root "$UAVSSC_DATA_ROOT" \
  --config preprocessing/configs/interval1.yaml

python preprocessing/scripts/01_build_taxonomy.py \
  --output data/index/taxonomy_uavssc.json

python preprocessing/scripts/02_build_manifest.py \
  --data-root "$UAVSSC_DATA_ROOT" \
  --config preprocessing/configs/interval1.yaml \
  --output data/index/manifest_interval1.parquet

python preprocessing/scripts/08_validate_label_pairing.py \
  --manifest data/index/manifest_interval1.parquet

python preprocessing/scripts/12_validate_cam_labels.py \
  --manifest data/index/manifest_interval1.parquet \
  --output data/processed/interval1/cam_label_report.csv

python preprocessing/scripts/03_projection_debug.py \
  --data-root "$UAVSSC_DATA_ROOT" \
  --manifest data/index/manifest_interval1.parquet \
  --output-dir data/processed/interval1/proj_debug \
  --max-samples 30

python preprocessing/scripts/04_fuse_global_semantic_cloud.py \
  --data-root "$UAVSSC_DATA_ROOT" \
  --manifest data/index/manifest_interval1.parquet \
  --config preprocessing/configs/interval1.yaml \
  --output data/processed/interval1/global_voxel_votes \
  --ext-mode cam_from_lidar \
  --max-rays-per-frame 5000

python preprocessing/scripts/10_export_rgb_ssc.py \
  --manifest data/index/manifest_interval1.parquet \
  --sparse-votes data/processed/interval1/global_voxel_votes \
  --config preprocessing/configs/interval1.yaml \
  --output data/processed/interval1/rgb_ssc_npz

python preprocessing/scripts/10_export_lidar_ssc.py \
  --manifest data/index/manifest_interval1.parquet \
  --sparse-votes data/processed/interval1/global_voxel_votes \
  --data-root "$UAVSSC_DATA_ROOT" \
  --config preprocessing/configs/interval1.yaml \
  --output data/processed/interval1/lidar_ssc_npz \
  --ext-mode cam_from_lidar

python preprocessing/scripts/10_export_fusion_ssc.py \
  --manifest data/index/manifest_interval1.parquet \
  --sparse-votes data/processed/interval1/global_voxel_votes \
  --data-root "$UAVSSC_DATA_ROOT" \
  --config preprocessing/configs/interval1.yaml \
  --output data/processed/interval1/fusion_ssc_npz \
  --ext-mode cam_from_lidar
```

---

## 6. What the manifest should contain

Open/check:

```bash
python - <<'PY'
import pandas as pd
m = pd.read_parquet('data/index/manifest_interval1.parquet')
print(m.columns.tolist())
print(m[['scene','timestamp','img_path','cam_label_id_path','cam_label_rgb_path','lidar_path','lidar_label_id_path']].head())
print('records:', len(m))
print('CAM label id paired:', m['cam_label_id_path'].notna().sum())
print('CAM label color paired:', m['cam_label_rgb_path'].notna().sum())
PY
```

Expected columns include:

```text
scene
timestamp
img_path
lidar_path
lidar_label_id_path
lidar_label_rgb_path
cam_label_id_path
cam_label_rgb_path
T_world_cam
T_world_lidar
K
dist
```

Expected result: most or all RGB frames should have a matching `cam_label_id_path` and `cam_label_rgb_path` when the downloaded `interval1_CAM_label` folder is complete.

---

## 7. What each `.npz` contains

### RGB-only NPZ

Location:

```text
data/processed/interval1/rgb_ssc_npz/<scene>/<timestamp>.npz
```

Important keys:

```text
img_path
cam_label_id_path
cam_label_rgb_path
cam_k
cam_E
T_world_cam
vox_origin
voxel_size
scene_size_m
grid_size_xyz
target
target_1_8
projected_pix_1
fov_mask_1
pix_z_1
CP_mega_matrix
frustums_masks
frustums_class_dists
```

### LiDAR-only NPZ

Location:

```text
data/processed/interval1/lidar_ssc_npz/<scene>/<timestamp>.npz
```

Important keys:

```text
lidar_path
T_world_lidar
input_occ_lidar
input_density_lidar
input_max_rel_height
input_mean_rel_height
lidar_sparse_coords
lidar_sparse_counts
lidar_points_local_xyz
target
occ_mask
free_mask
known_mask
sem_label
```

### RGB+LiDAR fusion NPZ

Location:

```text
data/processed/interval1/fusion_ssc_npz/<scene>/<timestamp>.npz
```

Important keys:

```text
img_path
lidar_path
cam_label_id_path
cam_label_rgb_path
RGB projection keys
LiDAR dense/sparse keys
target
occ_mask
free_mask
known_mask
sem_label
```

The 2D CAM labels are stored as paths by default to avoid making every NPZ very large. To embed the mask arrays inside every RGB/fusion NPZ, edit:

```yaml
cam_label_export:
  save_paths: true
  save_id_image: true
  save_rgb_image: true
```

in:

```text
preprocessing/configs/interval1.yaml
```

---

## 8. Alignment debugging

After export, inspect:

```text
data/processed/interval1/rgb_overlays/
data/processed/interval1/fusion_overlays/
```

The overlay script now creates a side-by-side view when CAM labels exist:

```text
left:  RGB image + projected occupied SSC voxels
right: CAM label image + projected occupied SSC voxels
```

The `.npz` target must visually match the scene geometry of the paired image. It does not need to look identical to a 2D segmentation mask because the `.npz` target is a local 3D voxel grid, but the projected occupied voxels should land on the same roads, roofs, fields, objects, and scene footprint.

If overlays are shifted, test these in order:

1. Check `data/processed/interval1/proj_debug/` and decide whether hypothesis A or B aligns better.
2. If hypothesis B is better, rerun with:

```bash
EXT_MODE=lidar_from_cam bash scripts/run_preprocess_interval1.sh
```

3. If the crop footprint is centered in the wrong area, edit:

```yaml
local_grid:
  camera_forward_axis: auto
```

Try one of:

```text
+x, -x, +y, -y, +z, -z
```

Then rerun the export steps.

Do not solve a systematic crop shift with only:

```text
shift_m = h / tan(theta)
```

That shortcut is fragile because pitch-angle conventions differ. A downward-looking camera near 90 degrees can make this formula either near zero or unstable depending on whether pitch is measured from horizontal or vertical. Use the full 6-DoF pose, the camera optical axis, and the local-ground anchor instead.

---

## 9. Training commands

Create scene-level splits after NPZ export:

```bash
python scripts/make_scene_registry.py \
  --raw-root data/raw/uavscenes_official \
  --out-csv data/index/scene_registry.csv \
  --intervals interval1

python scripts/make_splits_scene_strict.py \
  --registry data/index/scene_registry.csv \
  --out-root data/splits/scene_strict_cv
```

Populate sample lists for one modality, for example RGB-only:

```bash
for FOLD in fold_A fold_B fold_C fold_D; do
  python scripts/make_sample_lists_from_npz.py \
    --preprocess-root data/processed/interval1/rgb_ssc_npz \
    --split-root data/splits/scene_strict_cv/$FOLD
done
```

Train RGB-only starter model:

```bash
bash scripts/train_all_folds.sh \
  rgb_cgformer_style \
  data/processed/interval1/rgb_ssc_npz \
  data/raw/uavscenes_official
```

Train LiDAR-only starter model:

```bash
bash scripts/train_all_folds.sh \
  lidar_lmscnet_style \
  data/processed/interval1/lidar_ssc_npz \
  data/raw/uavscenes_official
```

Train RGB+LiDAR fusion starter model:

```bash
bash scripts/train_all_folds.sh \
  rgb_lidar_fusion_gate3d \
  data/processed/interval1/fusion_ssc_npz \
  data/raw/uavscenes_official
```

Expected outputs:

```text
checkpoints/<method>/<fold>/best.pt
checkpoints/<method>/<fold>/last.pt
logs/stdout/<method>_<fold>.log
```

---

## 10. Recommended research order

Start with this sequence:

1. Build manifest and validate CAM/LiDAR labels.
2. Verify projection overlays.
3. Generate a small interval1 subset by lowering `MAX_DEBUG_SAMPLES` and setting `MAX_RAYS_PER_FRAME=1000` for quick tests.
4. Export LiDAR-only NPZs and train the LiDAR baseline first.
5. Export RGB-only NPZs and test MonoScene/CGFormer-style transfer.
6. Train RGB+LiDAR fusion only after the single-modality pipelines produce plausible results.

The preprocessing phase is not optional. The `.npz` files are the actual training samples. If an `.npz` does not match the image when projected back into the image plane, the model will learn the wrong target regardless of architecture.

---

## 11. MonoScene/KITTI pretrained warning

You can test transfer learning from MonoScene KITTI pretrained weights, but treat it as a diagnostic baseline, not the main expected result.

Main risk: KITTI is ground-vehicle, forward-facing, street-scale data. UAVScenes is aerial/oblique or downward-looking, with different scale, viewpoint, class taxonomy, and altitude. A pretrained MonoScene model may help low-level image features, but the 3D geometry prior is mismatched.

Better baseline order:

```text
LiDAR-only sanity baseline -> RGB-only baseline -> RGB+LiDAR fusion model
```

---

## 12. Quick failure checklist

If the `.npz` visualization looks different from the corresponding RGB image, check these first:

1. Wrong `cam_from_lidar` vs `lidar_from_cam` extrinsic direction.
2. Wrong camera optical axis in `local_grid.camera_forward_axis`.
3. CAM/LiDAR timestamp pairing is wrong.
4. `interval5` labels are accidentally paired with `interval1` images.
5. Dataset folder has an extra wrapper level and labels are not found.
6. You treated all non-occupied voxels as free instead of unknown.
7. You used a drone-centered crop instead of a camera-footprint/local-ground crop.
8. The Terra/global voxel map is in a different frame from `T_world_cam`.

Run these diagnostics before training:

```bash
python preprocessing/scripts/00_inspect_dataset.py --data-root "$UAVSSC_DATA_ROOT" --config preprocessing/configs/interval1.yaml
python preprocessing/scripts/12_validate_cam_labels.py --manifest data/index/manifest_interval1.parquet
python preprocessing/scripts/03_projection_debug.py --data-root "$UAVSSC_DATA_ROOT" --manifest data/index/manifest_interval1.parquet --output-dir data/processed/interval1/proj_debug --max-samples 30
python preprocessing/scripts/11_overlay_npz_alignment.py --input-root data/processed/interval1/rgb_ssc_npz --output-dir data/processed/interval1/rgb_overlays --max-files 30
```
