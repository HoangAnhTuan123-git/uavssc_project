# UAV SSC Master Project Structure

**Start here:** read `MASTER_INSTRUCTIONS.md`. That file contains the interval1 CAM-label workflow, GPU-template recommendation, dataset placement, setup/download scripts, preprocessing commands, expected outputs, and training commands.


This repository skeleton combines the preprocessing and training phases for UAVScenes-based 3D Semantic Scene Completion.

What is already included:
- preprocessing code bundle with RGB-only, LiDAR-only, and RGB+LiDAR fusion exporters
- training folders for MonoScene, CGFormer-style, VoxFormer-style, LMSCNet-style, SCPNet-style, and RGB+LiDAR fusion
- project-level docs, split utilities, result folders, and dataset placeholders
- scene-level split utilities to reduce geographic leakage across runs of the same physical scene

What you still need to fill:
- `data/raw/uavscenes_official/` with the extracted UAVScenes dataset
- `data/external/pretrained/` with downloaded pretrained weights
- method-specific paths in config files
- link for dataset: https://drive.google.com/drive/folders/1HSJWc5qmIKLdpaS8w8pqrWch4F9MHIeN

## Recommended workflow

1. Fill `data/raw/uavscenes_official/` with the extracted dataset.
2. Run `scripts/make_scene_registry.py` to inventory scenes and runs.
3. Run `scripts/make_splits_scene_strict.py` to create cross-scene folds.
4. Run preprocessing in `preprocessing/` to build manifests, fuse sparse votes, and export NPZ files.
5. Point training configs to the exported NPZ roots and the split files under `data/splits/`.
6. Train baselines in this order:
   - `training/rgb_monoscene_official_adapter`
   - `training/lidar_lmscnet_style`
   - `training/lidar_scpnet_style`
   - `training/rgb_cgformer_style`
   - `training/rgb_voxformer_style`
   - `training/rgb_lidar_fusion_gate3d`

## Important split rule

Do not mix different runs of the same physical location across train / val / test.
Example: `AMtown01`, `AMtown02`, and `AMtown03` should stay inside the same outer split in the main cross-scene benchmark.

## Key folders

- `data/raw/` — put the extracted dataset here
- `data/index/` — manifests, registry, taxonomy, checksums
- `data/splits/` — scene-level split definitions
- `data/processed/` — sparse votes, scene voxel grids, exported NPZs, QC overlays
- `preprocessing/` — preprocessing code
- `training/` — model-specific training folders
- `configs/` — project-level dataset and split configs
- `checkpoints/`, `logs/`, `results/` — outputs

## Notes

- This project skeleton does not ship the UAVScenes dataset.
- Some training folders are research starters rather than official upstream reproductions.
- MonoScene is included as an adapted baseline path with support for UAVScenes NPZ exports.
- The starter training folders were patched to optionally read explicit split files instead of using random scene-internal ratios.

See:
- `docs/02_scene_grouping_and_splits.md`
- `docs/05_preprocessing_spec.md`
- `docs/06_training_protocols.md`
- `docs/08_hardware_budget.md`
