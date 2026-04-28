import numpy as np

# Dense train IDs exported by uavssc_monoscene_prep:
# 0 = empty
# 1..16 = static semantic classes
uavscenes_class_names = [
    "empty",
    "roof",
    "dirt_motor_road",
    "paved_motor_road",
    "river",
    "pool",
    "bridge",
    "container",
    "airstrip",
    "traffic_barrier",
    "green_field",
    "wild_field",
    "solar_board",
    "umbrella",
    "transparent_roof",
    "car_park",
    "paved_walk",
]

# Fallback frequencies used only if a training-set frequency cache has not been computed yet.
# The train script will normally replace these with dataset-specific frequencies computed
# from the exported UAVScenes targets.
# Keep all values positive to avoid divide-by-zero in the weight formula.
uavscenes_default_class_frequencies = np.array([
    1.0,  # empty, overwritten at runtime
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
], dtype=np.float64)
