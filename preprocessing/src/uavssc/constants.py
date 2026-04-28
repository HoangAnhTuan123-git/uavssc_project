from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

IGNORE_SEMANTIC_ID = 255
UNKNOWN_VOXEL = 255

# Raw UAVScenes cmap mirrored from official repo at time of writing.
# Use script 01 to materialize/edit your own training taxonomy.
RAW_CMAP: Dict[int, dict] = {
    0: {'name': 'background', 'RGB': [0, 0, 0]},
    1: {'name': 'roof', 'RGB': [119, 11, 32]},
    2: {'name': 'dirt_motor_road', 'RGB': [180, 165, 180]},
    3: {'name': 'paved_motor_road', 'RGB': [128, 64, 128]},
    4: {'name': 'river', 'RGB': [173, 216, 230]},
    5: {'name': 'pool', 'RGB': [0, 80, 100]},
    6: {'name': 'bridge', 'RGB': [150, 100, 100]},
    7: {'name': '', 'RGB': [150, 120, 90]},
    8: {'name': '', 'RGB': [70, 70, 70]},
    9: {'name': 'container', 'RGB': [250, 170, 30]},
    10: {'name': 'airstrip', 'RGB': [81, 0, 81]},
    11: {'name': 'traffic_barrier', 'RGB': [102, 102, 156]},
    12: {'name': '', 'RGB': [190, 153, 153]},
    13: {'name': 'green_field', 'RGB': [107, 142, 35]},
    14: {'name': 'wild_field', 'RGB': [210, 180, 140]},
    15: {'name': 'solar_board', 'RGB': [220, 220, 0]},
    16: {'name': 'umbrella', 'RGB': [153, 153, 153]},
    17: {'name': 'transparent_roof', 'RGB': [0, 0, 90]},
    18: {'name': 'car_park', 'RGB': [250, 170, 160]},
    19: {'name': 'paved_walk', 'RGB': [244, 35, 232]},
    20: {'name': 'sedan', 'RGB': [0, 0, 142]},
    21: {'name': '', 'RGB': [224, 224, 192]},
    22: {'name': '', 'RGB': [220, 20, 60]},
    23: {'name': '', 'RGB': [192, 64, 128]},
    24: {'name': 'truck', 'RGB': [0, 0, 70]},
    25: {'name': '', 'RGB': [0, 60, 100]},
}

RGB_TO_RAW_ID = {tuple(v['RGB']): k for k, v in RAW_CMAP.items()}

DEFAULT_IGNORE_RAW_IDS: List[int] = [0, 7, 8, 12, 21, 22, 23, 25]
DEFAULT_DYNAMIC_RAW_IDS: List[int] = [20, 24]
