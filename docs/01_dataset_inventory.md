# Dataset inventory

Fill this document after placing the extracted dataset into `data/raw/uavscenes_official/`.

Checklist:
- verify interval1 and interval5 folders
- verify Terra point cloud / mesh folders
- verify `cmap.py`
- verify `calibration_results.py`
- verify per-scene `sampleinfos_interpolated.json`
- note the total extracted size on disk
- note missing or corrupted scenes, if any

Suggested command sequence:
1. run `preprocessing/scripts/00_inspect_dataset.py`
2. run `scripts/make_scene_registry.py`
3. save a brief inventory summary here
