from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .utils import normalize_quaternion_xyzw


@dataclass
class Pose:
    """Rigid transform as 4x4 homogeneous matrix.

    Convention used in this repo:
    - `T_a_b` means transform a point from frame `b` into frame `a`.
    - If `p_b` is 3D point in frame b, then `p_a = T_a_b @ [p_b, 1]`.
    """

    matrix: np.ndarray

    def inverse(self) -> 'Pose':
        R = self.matrix[:3, :3]
        t = self.matrix[:3, 3]
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = R.T
        out[:3, 3] = -(R.T @ t)
        return Pose(out)

    def __matmul__(self, other: 'Pose') -> 'Pose':
        return Pose(self.matrix @ other.matrix)



def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T



def apply_transform(points_xyz: np.ndarray, T_dst_src: np.ndarray) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f'Expected (N,3), got {points_xyz.shape}')
    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float64)
    homog = np.concatenate([points_xyz, ones], axis=1)
    out = (T_dst_src @ homog.T).T
    return out[:, :3]



def quaternion_xyzw_to_rotmat(q_xyzw: Iterable[float]) -> np.ndarray:
    x, y, z, w = normalize_quaternion_xyzw(np.asarray(list(q_xyzw), dtype=np.float64))
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=np.float64)
    return R



def rotmat_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
    """Simple robust conversion. Returns x,y,z,w."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q
