# Taxonomy mapping

Freeze the training taxonomy early.

Suggested columns in `taxonomy_raw_to_train.csv`:
- `raw_id`
- `raw_name`
- `train_id`
- `train_name`
- `is_valid`
- `is_dynamic`
- `rgb_r`
- `rgb_g`
- `rgb_b`

Also record:
- which IDs are ignored
- which IDs are grouped
- any dynamic classes that should be treated carefully in long-term fusion
