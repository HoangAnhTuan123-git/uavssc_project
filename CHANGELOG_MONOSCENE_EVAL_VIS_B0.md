# MonoScene Evaluation and Visualization Patch for Configurable Backbones

This patch updates the MonoScene UAVScenes evaluation and visualization path so it matches the newer training changes:

- configurable RGB backbone, e.g. `tf_efficientnet_b0_ns`, `tf_efficientnet_b4_ns`, `tf_efficientnet_b7_ns`
- online resized RGB input through `input_image_hw`
- smaller feature width through `feature`
- optional disabled context prior / relation / frustum projection branches
- AMP-safe evaluation/prediction configuration

## Updated files

```text
training/rgb_monoscene_official_adapter/MonoScene/monoscene/scripts/eval_uavscenes.py
training/rgb_monoscene_official_adapter/MonoScene/monoscene/scripts/predict_uavscenes.py
training/rgb_monoscene_official_adapter/MonoScene/monoscene/scripts/visualization/uavscenes_vis_pred.py
training/rgb_monoscene_official_adapter/MonoScene/monoscene/data/uavscenes/collate.py
training/rgb_monoscene_official_adapter/MonoScene/monoscene/models/monoscene.py
training/rgb_monoscene_official_adapter/MonoScene/monoscene/config/uavscenes.yaml
training/rgb_monoscene_official_adapter/MonoScene/scripts/example_eval_uav.sh
scripts/eval_rgb_monoscene_interval5_4090_b0.sh
scripts/visualize_rgb_monoscene_interval5_4090_b0.sh
```

## Why this was needed

Training was changed from the original MonoScene EfficientNet-B7 setup to a lightweight B0/B4 configurable setup. Evaluation and visualization must instantiate the same architecture as training. If evaluation silently builds B7 or uses full-resolution images, it can OOM or fail with checkpoint shape mismatches.

## Evaluation command

```bash
cd /root/Tuan/uavssc_project
EVAL_CHECKPOINT=/absolute/path/to/checkpoint.ckpt \
  bash scripts/eval_rgb_monoscene_interval5_4090_b0.sh
```

Optional overrides:

```bash
RGB_BACKBONE=tf_efficientnet_b4_ns \
INPUT_IMAGE_HW='[320,384]' \
FEATURE=32 \
EVAL_CHECKPOINT=/absolute/path/to/checkpoint.ckpt \
  bash scripts/eval_rgb_monoscene_interval5_4090_b0.sh
```

## Visualization command

```bash
cd /root/Tuan/uavssc_project
EVAL_CHECKPOINT=/absolute/path/to/checkpoint.ckpt \
MAX_PREDICT_SAMPLES=50 \
  bash scripts/visualize_rgb_monoscene_interval5_4090_b0.sh
```

Outputs:

```text
results/qualitative/rgb_monoscene_predictions_*/
results/qualitative/rgb_monoscene_visualizations_*/
```

Each visualization panel contains:

- RGB input
- predicted top-down semantic SSC map
- target top-down semantic SSC map
- predicted height/occupancy summary

## Important rule

Use the same `RGB_BACKBONE`, `FEATURE`, `INPUT_IMAGE_HW`, `context_prior`, `relation_loss`, `fp_loss`, and `project_1_*` settings for training, evaluation, and visualization.
