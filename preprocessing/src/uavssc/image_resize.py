from __future__ import annotations

from typing import Any

import numpy as np


def _as_hw(value: Any) -> tuple[int, int] | None:
    """Parse a config/NPZ image size value as (height, width)."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size < 2:
            return None
        flat = value.reshape(-1)
        return int(flat[0]), int(flat[1])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    if isinstance(value, str):
        txt = value.strip().replace("[", "").replace("]", "")
        if not txt:
            return None
        parts = [p.strip() for p in txt.replace("x", ",").split(",") if p.strip()]
        if len(parts) >= 2:
            return int(float(parts[0])), int(float(parts[1]))
    return None


def resolve_image_export_shape(orig_h: int, orig_w: int, cfg: dict) -> tuple[int, int, float, float, str]:
    """Return (new_h, new_w, sy, sx, mode) for preprocessing exports.

    Config examples:
      image_export:
        resize_hw: [640, 768]   # fixed H,W

      image_export:
        max_size_hw: [640, 768] # preserve aspect inside this box

      image_export:
        max_long_edge: 768      # preserve aspect
    """
    image_cfg = (cfg or {}).get("image_export", {}) or {}
    enabled = bool(image_cfg.get("enabled", True))
    if not enabled:
        return int(orig_h), int(orig_w), 1.0, 1.0, "disabled"

    fixed = _as_hw(
        image_cfg.get("resize_hw")
        or image_cfg.get("input_size_hw")
        or image_cfg.get("target_hw")
    )
    if fixed is not None:
        new_h, new_w = fixed
        if new_h <= 0 or new_w <= 0:
            raise ValueError(f"Invalid image_export resize_hw: {fixed}")
        return new_h, new_w, new_h / float(orig_h), new_w / float(orig_w), "fixed_hw"

    max_hw = _as_hw(image_cfg.get("max_size_hw"))
    if max_hw is not None:
        max_h, max_w = max_hw
        scale = min(max_h / float(orig_h), max_w / float(orig_w), 1.0)
        new_h = max(1, int(round(orig_h * scale)))
        new_w = max(1, int(round(orig_w * scale)))
        return new_h, new_w, new_h / float(orig_h), new_w / float(orig_w), "max_size_hw"

    max_long = image_cfg.get("max_long_edge", None)
    if max_long is not None:
        max_long = int(max_long)
        if max_long <= 0:
            raise ValueError(f"Invalid image_export max_long_edge: {max_long}")
        scale = min(max_long / float(max(orig_h, orig_w)), 1.0)
        new_h = max(1, int(round(orig_h * scale)))
        new_w = max(1, int(round(orig_w * scale)))
        return new_h, new_w, new_h / float(orig_h), new_w / float(orig_w), "max_long_edge"

    return int(orig_h), int(orig_w), 1.0, 1.0, "original"


def scale_camera_matrix(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Scale a 3x3 camera intrinsic matrix for a resized image."""
    K2 = np.asarray(K, dtype=np.float32).copy()
    K2[0, 0] *= float(sx)
    K2[0, 2] *= float(sx)
    K2[1, 1] *= float(sy)
    K2[1, 2] *= float(sy)
    return K2
