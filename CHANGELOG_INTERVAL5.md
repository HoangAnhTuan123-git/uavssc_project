# Changelog — interval5 update

## Added

- `MASTER_INSTRUCTIONS.md` is now the interval5-first workflow.
- `MASTER_INSTRUCTIONS_INTERVAL1.md` preserves the previous interval1 workflow.
- `scripts/run_preprocess_interval5.sh` runs the full interval5 preprocessing pipeline.
- `scripts/run_preprocess_uavscenes.sh` lets you switch with `INTERVAL=5` or `INTERVAL=1`.
- `preprocessing/configs/default.yaml` now defaults to `use_interval: 5`.
- Interval configs now include explicit sibling label-root hints:
  - `interval5_CAM_label`
  - `interval5_LIDAR_label`
  - `interval1_CAM_label`
  - `interval1_LIDAR_label`

## Kept

- Interval1 support remains available through `preprocessing/configs/interval1.yaml` and `scripts/run_preprocess_interval1.sh`.
- CAM-label path support remains in the manifest and RGB/fusion NPZ exporters.

## Practical effect

The default project behavior now matches the recommended first-stage workflow: build and validate the SSC benchmark on interval5 before scaling to interval1.
