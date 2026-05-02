# UAVScenes SSC Master Instructions — interval5 workflow

## Main Insight

Use **interval5 first**. It is the better engineering choice for this project stage because it is approximately one fifth of the frame count, uses much less storage, and is much faster for debugging the SSC ground-truth pipeline. Since adjacent UAVScenes frames are about 0.1 s apart in `sampleinfos_interpolated.json`, interval5 still gives roughly 0.5 s sampling, which is usually adequate for static aerial SSC target construction.

Interval1 should be used later for teacher-student expansion, pseudo-labeling, or final high-density experiments after the interval5 preprocessing overlays are correct.

---

## 1. Recommended GPU container

Use:

```text
PyTorch 2.0.1 + CUDA 11.7
```

Reason: this project includes MonoScene/KITTI-style code. Those older SSC repositories are usually more stable with PyTorch 1.x/2.0-era CUDA than with newest PyTorch/CUDA stacks. Use PyTorch 2.9.1 + CUDA 12.8 only for the lightweight starter models if you are not running MonoScene-compatible code.

Check GPU:

```bash
nvidia-smi
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda:', torch.version.cuda)
PY
```

Expected output: CUDA is available and the GPU is visible.

---

## 2. Project setup in VS Code / GPU container

From the container workspace:

```bash
cd /workspace
unzip uavssc_project_interval5_camlabel_updated.zip -d /workspace
cd /workspace/uavssc_project
```

Install preprocessing dependencies:

```bash
bash scripts/setup_env.sh
```

Expected output: editable preprocessing package installed and common packages such as numpy, pandas, pyarrow, OpenCV, Open3D, rich, tqdm, and gdown available.

---

## 3. Dataset placement

Put the extracted dataset here:

```text
uavssc_project/data/raw/uavscenes_official/
```

Recommended interval5 layout:

```text
data/raw/uavscenes_official/
  interval5_AMtown01/
    interval5_CAM/
    interval5_LIDAR/
    poses_viz.png
    rtk_positions_raw.xlsx or .csv
    rtk_positions_raw_viz.png
    sampleinfos_interpolated.json
  interval5_AMtown02/
  interval5_AMtown03/
  interval5_AMvalley01/
  ...
  interval5_CAM_label/
    ... label_id PNG masks ...
    ... label_color PNG masks ...
  interval5_LIDAR_label/
    ... label_id TXT files ...
    ... label_color TXT files ...
  terra_3dmap_pointcloud_mesh/
    AMtown/
      cloud_merged.ply
      Mesh.ply
      terra_ply/
    ...
  cmap.py
  calibration_results.py
```

Official-wrapper layout is also supported:

```text
data/raw/uavscenes_official/
  interval5_CAM_LIDAR/
  interval5_CAM_label/
  interval5_LIDAR_label/
  terra_3dmap_pointcloud_mesh/
  cmap.py
  calibration_results.py
```

Do **not** move `interval5_CAM_label` or `interval5_LIDAR_label` inside each scene folder. The code searches sibling label folders and pairs labels by timestamp.

Set the dataset root:

```bash
export UAVSSC_DATA_ROOT=$PWD/data/raw/uavscenes_official
```

Quick structure check:

```bash
find "$UAVSSC_DATA_ROOT" -maxdepth 2 -type d | sort | head -80
```

Expected output: `interval5_*` scene folders or `interval5_CAM_LIDAR`, plus `interval5_CAM_label`, `interval5_LIDAR_label`, and `terra_3dmap_pointcloud_mesh`.

---

## 4. Optional Google Drive download

Use the owner-shared Google Drive folder only when your container has internet access:

```bash
export UAVSSC_GDRIVE_URL="https://drive.google.com/drive/folders/1HSJWc5qmIKLdpaS8w8pqrWch4F9MHIeN"
bash scripts/download_uavscenes_gdrive.sh
```

Expected output:

```text
data/raw/uavscenes_official/
  interval5_...
  interval5_CAM_label/
  interval5_LIDAR_label/
  terra_3dmap_pointcloud_mesh/
```

If the downloaded folder contains extra wrapper folders, keep the official extracted structure and set `UAVSSC_DATA_ROOT` to the folder that contains the interval5 folders.

---

## 5. One-command interval5 preprocessing

Run:

```bash
export UAVSSC_DATA_ROOT=$PWD/data/raw/uavscenes_official
bash scripts/run_preprocess_interval5.sh
```

Equivalent generic command:

```bash
INTERVAL=5 bash scripts/run_preprocess_uavscenes.sh
```

Expected outputs:

```text
data/index/manifest_interval5.parquet
data/index/taxonomy_uavssc.json
data/processed/interval5/cam_label_report.csv
data/processed/interval5/proj_debug/
data/processed/interval5/scene_boxes.json
data/processed/interval5/global_voxel_votes/
data/processed/interval5/scene_voxel_maps/
data/processed/interval5/rgb_ssc_npz/
data/processed/interval5/lidar_ssc_npz/
data/processed/interval5/fusion_ssc_npz/
data/processed/interval5/rgb_overlays/
data/processed/interval5/fusion_overlays/
```

---

## 6. Manual preprocessing commands

Use these when debugging one stage at a time.

```bash
python preprocessing/scripts/00_inspect_dataset.py \
  --data-root "$UAVSSC_DATA_ROOT" \
  --config preprocessing/configs/interval5.yaml
```

Expected output: scene count, interval=5, global calibration/Terra files, per-scene image/LiDAR counts, and manifest pairing counts.

```bash
python preprocessing/scripts/01_build_taxonomy.py \
  --output data/index/taxonomy_uavssc.json
```

Expected output: `data/index/taxonomy_uavssc.json`.

```bash
python preprocessing/scripts/02_build_manifest.py \
  --data-root "$UAVSSC_DATA_ROOT" \
  --config preprocessing/configs/interval5.yaml \
  --output data/index/manifest_interval5.parquet
```

Expected output: manifest rows with these important columns:

```text
scene, timestamp, img_path, lidar_path,
lidar_label_id_path, lidar_label_rgb_path,
cam_label_id_path, cam_label_rgb_path,
T_world_cam, T_world_lidar, K, dist
```

Validate LiDAR labels:

```bash
python preprocessing/scripts/08_validate_label_pairing.py \
  --manifest data/index/manifest_interval5.parquet
```

Validate camera labels:

```bash
python preprocessing/scripts/12_validate_cam_labels.py \
  --manifest data/index/manifest_interval5.parquet \
  --output data/processed/interval5/cam_label_report.csv
```

Expected result: CAM label masks have the same timestamp stem and image size as the paired RGB frame when the complete `interval5_CAM_label` folder is present.

Projection QC:

```bash
python preprocessing/scripts/03_projection_debug.py \
  --data-root "$UAVSSC_DATA_ROOT" \
  --manifest data/index/manifest_interval5.parquet \
  --output-dir data/processed/interval5/proj_debug \
  --max-samples 30
```

Expected result: projected LiDAR/geometry should align with RGB structures. If it is mirrored or strongly shifted, fix calibration/extrinsic interpretation before training.

Fuse global sparse semantic/free-space votes:

```bash
python preprocessing/scripts/04_fuse_global_semantic_cloud.py \
  --data-root "$UAVSSC_DATA_ROOT" \
  --manifest data/index/manifest_interval5.parquet \
  --config preprocessing/configs/interval5.yaml \
  --output data/processed/interval5/global_voxel_votes \
  --ext-mode cam_from_lidar \
  --max-rays-per-frame 5000
```

Resolve scene-level voxel maps:

```bash
python preprocessing/scripts/05_build_scene_voxel_grid.py \
  --input data/processed/interval5/global_voxel_votes \
  --config preprocessing/configs/interval5.yaml \
  --output data/processed/interval5/scene_voxel_maps
```

Export all model-input variants:

```bash
python preprocessing/scripts/10_export_rgb_ssc.py \
  --manifest data/index/manifest_interval5.parquet \
  --sparse-votes data/processed/interval5/global_voxel_votes \
  --config preprocessing/configs/interval5.yaml \
  --output data/processed/interval5/rgb_ssc_npz

python preprocessing/scripts/10_export_lidar_ssc.py \
  --manifest data/index/manifest_interval5.parquet \
  --sparse-votes data/processed/interval5/global_voxel_votes \
  --data-root "$UAVSSC_DATA_ROOT" \
  --config preprocessing/configs/interval5.yaml \
  --output data/processed/interval5/lidar_ssc_npz \
  --ext-mode cam_from_lidar

python preprocessing/scripts/10_export_fusion_ssc.py \
  --manifest data/index/manifest_interval5.parquet \
  --sparse-votes data/processed/interval5/global_voxel_votes \
  --data-root "$UAVSSC_DATA_ROOT" \
  --config preprocessing/configs/interval5.yaml \
  --output data/processed/interval5/fusion_ssc_npz \
  --ext-mode cam_from_lidar
```

Make overlay audits:

```bash
python preprocessing/scripts/11_overlay_npz_alignment.py \
  --input-root data/processed/interval5/rgb_ssc_npz \
  --output-dir data/processed/interval5/rgb_overlays \
  --max-files 30

python preprocessing/scripts/11_overlay_npz_alignment.py \
  --input-root data/processed/interval5/fusion_ssc_npz \
  --output-dir data/processed/interval5/fusion_overlays \
  --max-files 30
```

Expected result: the projected 3D target should correspond to the same road/roof/field layout visible in the paired image and CAM label.

---

## 7. How to inspect one NPZ sample

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
root = Path('data/processed/interval5/fusion_ssc_npz')
f = next(root.rglob('*.npz'))
z = np.load(f, allow_pickle=True)
print('file:', f)
print('keys:', sorted(z.files))
for k in z.files:
    v = z[k]
    if hasattr(v, 'shape'):
        print(k, v.shape, v.dtype)
    else:
        print(k, v)
PY
```

Important keys usually include:

```text
target_semantic, target_free, target_valid,
origin, voxel_size, K, T_world_cam,
img_path, cam_label_id_path, cam_label_rgb_path
```

For RGB-only, LiDAR-only, and fusion training, the **target grid should be the same benchmark target**. The input modality changes; the ground-truth SSC target should not.

---

## 8. Training commands

Make sample-list files:

```bash
python scripts/make_sample_lists_from_npz.py \
  data/processed/interval5/rgb_ssc_npz \
  --output-dir data/splits/interval5_rgb

python scripts/make_sample_lists_from_npz.py \
  data/processed/interval5/lidar_ssc_npz \
  --output-dir data/splits/interval5_lidar

python scripts/make_sample_lists_from_npz.py \
  data/processed/interval5/fusion_ssc_npz \
  --output-dir data/splits/interval5_fusion
```

RGB-only starter model:

```bash
cd training/rgb_cgformer_style
python train.py \
  --config configs/default.yaml \
  --preprocess-root ../../data/processed/interval5/rgb_ssc_npz \
  --output-dir ../../runs/interval5_rgb_cgformer_style
```

LiDAR-only starter model:

```bash
cd training/lidar_lmscnet_style
python train.py \
  --config configs/default.yaml \
  --preprocess-root ../../data/processed/interval5/lidar_ssc_npz \
  --output-dir ../../runs/interval5_lidar_lmscnet_style
```

RGB+LiDAR fusion starter model:

```bash
cd training/rgb_lidar_fusion_gate3d
python train.py \
  --config configs/default.yaml \
  --preprocess-root ../../data/processed/interval5/fusion_ssc_npz \
  --output-dir ../../runs/interval5_fusion_gate3d
```

MonoScene adapter:

```bash
cd training/rgb_monoscene_official_adapter/MonoScene
bash scripts/download_pretrained.sh
python monoscene/scripts/train_uavscenes.py \
  --config monoscene/config/uavscenes.yaml \
  --preprocess-root ../../../data/processed/interval5/rgb_ssc_npz
```

Treat MonoScene KITTI pretraining as an experiment, not as the main expected winner. UAV top-down geometry, scale, camera pitch, and class priors differ strongly from ground-vehicle KITTI.

---

## 9. Alignment troubleshooting

### Symptom

The `.npz` visualization does not look like the paired 2D image.

### Most likely causes, in order

1. Wrong camera/LiDAR extrinsic direction.
2. Wrong camera forward-axis convention.
3. Wrong timestamp pairing between RGB, LiDAR, label, and sampleinfos.
4. Local grid centered at the wrong altitude or wrong ground anchor.
5. Interval1 labels accidentally paired with interval5 images, or the reverse.
6. Treating unknown space as free space.

### What to avoid

Do not start with the formula:

```text
shift_m ≈ h / tan(theta)
```

That is a 2D nadir-view approximation. It does not solve 6-DoF projection, yaw/roll, lens intrinsics, map-frame transforms, or LiDAR-camera extrinsic direction. A constant manual shift can make one frame look better and make other frames worse. Use it only as a diagnostic after projection overlays prove the transform is otherwise correct.

### Useful tests

```bash
python preprocessing/scripts/00_inspect_dataset.py --data-root "$UAVSSC_DATA_ROOT" --config preprocessing/configs/interval5.yaml
python preprocessing/scripts/03_projection_debug.py --data-root "$UAVSSC_DATA_ROOT" --manifest data/index/manifest_interval5.parquet --output-dir data/processed/interval5/proj_debug --max-samples 30
python preprocessing/scripts/11_overlay_npz_alignment.py --input-root data/processed/interval5/fusion_ssc_npz --output-dir data/processed/interval5/fusion_overlays --max-files 30
```

If `cam_from_lidar` looks wrong, test the inverse assumption:

```bash
EXT_MODE=lidar_from_cam bash scripts/run_preprocess_interval5.sh
```

Compare `proj_debug` and `fusion_overlays` before choosing.

---

## 10. Interval1 expansion later

After interval5 works:

```bash
INTERVAL=1 bash scripts/run_preprocess_uavscenes.sh
```

Expected interval1 outputs:

```text
data/index/manifest_interval1.parquet
data/processed/interval1/rgb_ssc_npz/
data/processed/interval1/lidar_ssc_npz/
data/processed/interval1/fusion_ssc_npz/
```

Best use of interval1: train a teacher on interval5, infer high-confidence pseudo-labels on interval1, then fine-tune with interval5 ground truth plus filtered interval1 pseudo-labels.
