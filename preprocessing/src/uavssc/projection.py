from __future__ import annotations

import numpy as np
import cv2

from .transforms import apply_transform



def undistort_image(img: np.ndarray, K: np.ndarray, dist: np.ndarray | None) -> np.ndarray:
    if dist is None or np.allclose(dist, 0):
        return img
    return cv2.undistort(img, K, dist)



def project_points_world_to_image(
    points_world: np.ndarray,
    T_world_cam: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points into image.

    Returns:
        uv: (N,2) float pixels
        depth: (N,) positive camera-frame z
        valid: (N,) bool, point is in front of camera
    """
    T_cam_world = np.linalg.inv(T_world_cam)
    pts_cam = apply_transform(points_world, T_cam_world)
    z = pts_cam[:, 2]
    valid = z > 1e-6
    uv = np.full((points_world.shape[0], 2), np.nan, dtype=np.float64)
    if valid.any():
        x = pts_cam[valid, 0] / z[valid]
        y = pts_cam[valid, 1] / z[valid]
        uv_valid = (K @ np.stack([x, y, np.ones_like(x)], axis=0)).T
        uv[valid] = uv_valid[:, :2]
    return uv, z, valid



def draw_projected_points(
    img: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    radius: int = 2,
) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for (u, v), ok in zip(uv, valid):
        if not ok or not np.isfinite(u) or not np.isfinite(v):
            continue
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < w and 0 <= vi < h:
            cv2.circle(out, (ui, vi), radius, color, -1)
    return out
