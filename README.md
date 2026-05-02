# UAVScenes SSC project

**Start here:** read `MASTER_INSTRUCTIONS.md`. The primary workflow is now **interval5** because it is faster, smaller, and better for debugging the SSC benchmark construction. Interval1 remains supported as an optional full-rate expansion.

Main commands:

```bash
bash scripts/setup_env.sh
export UAVSSC_DATA_ROOT=$PWD/data/raw/uavscenes_official
bash scripts/run_preprocess_interval5.sh
```

Key generated outputs:

```text
data/index/manifest_interval5.parquet
data/processed/interval5/rgb_ssc_npz/
data/processed/interval5/lidar_ssc_npz/
data/processed/interval5/fusion_ssc_npz/
data/processed/interval5/rgb_overlays/
data/processed/interval5/fusion_overlays/
```

For the previous full-rate workflow, see `MASTER_INSTRUCTIONS_INTERVAL1.md` and run `bash scripts/run_preprocess_interval1.sh`.
