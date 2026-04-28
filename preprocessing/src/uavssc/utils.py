from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any
import re

import numpy as np
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files_recursive(root: str | Path, suffixes: tuple[str, ...] | None = None) -> list[Path]:
    root = Path(root)
    files = [p for p in root.rglob('*') if p.is_file()]
    if suffixes is not None:
        files = [p for p in files if p.suffix.lower() in suffixes]
    return sorted(files)


TS_PAT = re.compile(r'(?:^|[^0-9])((?:1[0-9]{9})(?:\.[0-9]+)?)(?:$|[^0-9])')


def timestamp_from_stem(path: str | Path) -> float | None:
    """Parse timestamps from filenames robustly.

    Handles all of these common cases:
    - 1658137057.624840774.txt
    - lidar1658137057.624840774.txt
    - scan_1658137057.624840774.txt
    - image1658137057.641204937_lidar1658137057.624840774.txt

    Preference order:
    1) whole stem as float
    2) token after 'lidar'
    3) token after 'image'
    4) first epoch-like floating number in the stem
    """
    stem = Path(path).stem
    try:
        return float(stem)
    except Exception:
        pass

    for key in ['lidar', 'image']:
        m = re.search(rf'{key}(?P<ts>(?:1[0-9]{{9}})(?:\.[0-9]+)?)', stem)
        if m:
            try:
                return float(m.group('ts'))
            except Exception:
                pass

    m = TS_PAT.search(stem)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return None


def pair_by_nearest_timestamp(
    anchors: list[float],
    queries: list[float],
    max_abs_delta: float | None = None,
) -> list[int | None]:
    """Return, for every anchor timestamp, the index of the nearest query timestamp."""
    if not queries:
        return [None] * len(anchors)

    q = np.asarray(queries, dtype=np.float64)
    out: list[int | None] = []
    for a in anchors:
        idx = int(np.argmin(np.abs(q - a)))
        if max_abs_delta is not None and abs(q[idx] - a) > max_abs_delta:
            out.append(None)
        else:
            out.append(idx)
    return out


def as_float_array(x: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if shape is not None and arr.shape != shape:
        raise ValueError(f'Expected shape {shape}, got {arr.shape}')
    return arr


def normalize_quaternion_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError('Zero quaternion norm.')
    return q / n
