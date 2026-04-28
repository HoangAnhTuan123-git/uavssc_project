from __future__ import annotations

import numpy as np


IGNORE_LABEL = 255



def vox2world(vox_origin: np.ndarray, vox_coords: np.ndarray, voxel_size: float, offsets=(0.5, 0.5, 0.5)) -> np.ndarray:
    vox_origin = np.asarray(vox_origin, dtype=np.float32).reshape(3)
    vox_coords = np.asarray(vox_coords, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.float32).reshape(3)
    return vox_origin[None, :] + voxel_size * (vox_coords + offsets[None, :])



def rigid_transform(points_xyz: np.ndarray, T_dst_src: np.ndarray) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    T = np.asarray(T_dst_src, dtype=np.float32).reshape(4, 4)
    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([points_xyz, ones], axis=1)
    out = (T @ homo.T).T
    return out[:, :3]



def cam2pix(cam_pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    cam_pts = np.asarray(cam_pts, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32).reshape(3, 3)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    pix = np.empty((cam_pts.shape[0], 2), dtype=np.int32)
    pix[:, 0] = np.round((cam_pts[:, 0] * fx / cam_pts[:, 2]) + cx).astype(np.int32)
    pix[:, 1] = np.round((cam_pts[:, 1] * fy / cam_pts[:, 2]) + cy).astype(np.int32)
    return pix



def vox2pix(cam_E: np.ndarray, cam_k: np.ndarray, vox_origin: np.ndarray, voxel_size: float, img_W: int, img_H: int, scene_size: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adapted from MonoScene helper logic, but standalone.

    cam_E: transform from voxel/world frame to camera frame.
    vox_origin: coordinates of voxel (0,0,0) in voxel/world frame.
    """
    scene_size = np.asarray(scene_size, dtype=np.float32).reshape(3)
    vol_dim = np.ceil(scene_size / float(voxel_size)).astype(np.int32)
    xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
    vox_coords = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)
    world_pts = vox2world(vox_origin, vox_coords, voxel_size)
    cam_pts = rigid_transform(world_pts, cam_E)
    projected_pix = cam2pix(cam_pts, cam_k)
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]
    pix_z = cam_pts[:, 2]
    fov_mask = (
        (pix_x >= 0) & (pix_x < img_W) &
        (pix_y >= 0) & (pix_y < img_H) &
        (pix_z > 0)
    )
    return projected_pix, fov_mask, pix_z



def majority_pooling(grid: np.ndarray, k_size: int = 2, empty_label: int = 0, ignore_label: int = IGNORE_LABEL) -> np.ndarray:
    grid = np.asarray(grid)
    out_shape = (grid.shape[0] // k_size, grid.shape[1] // k_size, grid.shape[2] // k_size)
    result = np.zeros(out_shape, dtype=grid.dtype)
    for xx in range(out_shape[0]):
        for yy in range(out_shape[1]):
            for zz in range(out_shape[2]):
                sub = grid[
                    xx * k_size:(xx + 1) * k_size,
                    yy * k_size:(yy + 1) * k_size,
                    zz * k_size:(zz + 1) * k_size,
                ]
                unique, counts = np.unique(sub, return_counts=True)
                mask_sem = (unique != empty_label) & (unique != ignore_label)
                if np.any(mask_sem):
                    unique = unique[mask_sem]
                    counts = counts[mask_sem]
                else:
                    mask_not_ignore = unique != ignore_label
                    unique = unique[mask_not_ignore]
                    counts = counts[mask_not_ignore]
                result[xx, yy, zz] = unique[np.argmax(counts)] if unique.size > 0 else ignore_label
    return result



def downsample_label(label_1_1: np.ndarray, factor: int = 8) -> np.ndarray:
    label = np.asarray(label_1_1)
    out = label.copy()
    step = 2
    cur = 1
    while cur < factor:
        out = majority_pooling(out, k_size=step)
        cur *= step
    return out



def compute_CP_mega_matrix(target: np.ndarray, is_binary: bool = False, ignore_label: int = IGNORE_LABEL) -> np.ndarray:
    target = np.asarray(target)
    label_row = target.reshape(-1)
    N = label_row.shape[0]
    super_voxel_size = [i // 2 for i in target.shape]
    C = 2 if is_binary else 4
    matrix = np.zeros((C, N, super_voxel_size[0] * super_voxel_size[1] * super_voxel_size[2]), dtype=np.uint8)
    for xx in range(super_voxel_size[0]):
        for yy in range(super_voxel_size[1]):
            for zz in range(super_voxel_size[2]):
                col_idx = xx * (super_voxel_size[1] * super_voxel_size[2]) + yy * super_voxel_size[2] + zz
                vals = np.array([
                    target[xx * 2, yy * 2, zz * 2],
                    target[xx * 2 + 1, yy * 2, zz * 2],
                    target[xx * 2, yy * 2 + 1, zz * 2],
                    target[xx * 2, yy * 2, zz * 2 + 1],
                    target[xx * 2 + 1, yy * 2 + 1, zz * 2],
                    target[xx * 2 + 1, yy * 2, zz * 2 + 1],
                    target[xx * 2, yy * 2 + 1, zz * 2 + 1],
                    target[xx * 2 + 1, yy * 2 + 1, zz * 2 + 1],
                ])
                vals = vals[vals != ignore_label]
                for label_col_mega in vals:
                    label_col = np.full(N, label_col_mega, dtype=label_row.dtype)
                    if not is_binary:
                        matrix[0, (label_row != ignore_label) & (label_col == label_row) & (label_col != 0), col_idx] = 1
                        matrix[1, (label_row != ignore_label) & (label_col != label_row) & (label_col != 0) & (label_row != 0), col_idx] = 1
                        matrix[2, (label_row != ignore_label) & (label_row == label_col) & (label_col == 0), col_idx] = 1
                        matrix[3, (label_row != ignore_label) & (label_row != label_col) & ((label_row == 0) | (label_col == 0)), col_idx] = 1
                    else:
                        matrix[0, (label_row != ignore_label) & (label_col != label_row), col_idx] = 1
                        matrix[1, (label_row != ignore_label) & (label_col == label_row), col_idx] = 1
    return matrix



def compute_local_frustums(projected_pix: np.ndarray, pix_z: np.ndarray, target: np.ndarray, img_W: int, img_H: int, n_classes: int, size: int = 4, ignore_label: int = IGNORE_LABEL) -> tuple[np.ndarray, np.ndarray]:
    H, W, D = target.shape
    ranges = [(i / size, (i + 1) / size) for i in range(size)]
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]
    masks = []
    dists = []
    for y in ranges:
        for x in ranges:
            start_x, end_x = x[0] * img_W, x[1] * img_W
            start_y, end_y = y[0] * img_H, y[1] * img_H
            local_frustum = (
                (pix_x >= start_x) & (pix_x < end_x) &
                (pix_y >= start_y) & (pix_y < end_y) &
                (pix_z > 0)
            )
            mask = (target != ignore_label) & local_frustum.reshape(H, W, D)
            masks.append(mask)
            classes, cnts = np.unique(target[mask], return_counts=True)
            class_counts = np.zeros(n_classes, dtype=np.int32)
            classes = classes.astype(np.int32)
            good = (classes >= 0) & (classes < n_classes)
            class_counts[classes[good]] = cnts[good]
            dists.append(class_counts)
    return np.asarray(masks), np.asarray(dists)
