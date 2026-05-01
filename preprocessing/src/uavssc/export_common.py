
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .constants import IGNORE_SEMANTIC_ID


Index3 = Tuple[int, int, int]


def parse_matrix_cell(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, str):
        return np.asarray(ast.literal_eval(x), dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def group_occ_votes(occ_idx: np.ndarray, occ_cls: np.ndarray, occ_cnt: np.ndarray):
    if occ_idx.size == 0:
        return (
            np.empty((0, 3), dtype=np.int32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int16),
        )

    order = np.lexsort((occ_cnt, occ_idx[:, 2], occ_idx[:, 1], occ_idx[:, 0]))
    occ_idx = occ_idx[order]
    occ_cls = occ_cls[order]
    occ_cnt = occ_cnt[order]

    change = np.any(np.diff(occ_idx, axis=0) != 0, axis=1)
    starts = np.r_[0, np.nonzero(change)[0] + 1]
    ends = np.r_[starts[1:], len(occ_idx)]

    uniq_idx = occ_idx[starts]
    occ_total = np.add.reduceat(occ_cnt.astype(np.int64), starts)
    occ_winner = occ_cls[ends - 1]
    return uniq_idx, occ_total, occ_winner


def sort_by_x(idx: np.ndarray, *arrays):
    if len(idx) == 0:
        return (idx,) + arrays
    order = np.lexsort((idx[:, 2], idx[:, 1], idx[:, 0]))
    out = [idx[order]]
    out.extend(arr[order] for arr in arrays)
    return tuple(out)


def query_sorted_x(idx_sorted: np.ndarray, x_min: int, x_max: int) -> np.ndarray:
    if len(idx_sorted) == 0:
        return np.empty((0,), dtype=np.int64)
    xs = idx_sorted[:, 0]
    l = np.searchsorted(xs, x_min, side='left')
    r = np.searchsorted(xs, x_max, side='left')
    return np.arange(l, r, dtype=np.int64)


def select_local_occ(
    occ_idx_sorted: np.ndarray,
    occ_total_sorted: np.ndarray,
    occ_winner_sorted: np.ndarray,
    idx_min: np.ndarray,
    idx_max: np.ndarray,
):
    occ_slice = query_sorted_x(occ_idx_sorted, int(idx_min[0]), int(idx_max[0]))
    if len(occ_slice) == 0:
        return (
            np.empty((0, 3), dtype=np.int32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int16),
        )
    occ_idx = occ_idx_sorted[occ_slice]
    mask = (
        (occ_idx[:, 1] >= idx_min[1]) & (occ_idx[:, 1] < idx_max[1]) &
        (occ_idx[:, 2] >= idx_min[2]) & (occ_idx[:, 2] < idx_max[2])
    )
    return occ_idx[mask], occ_total_sorted[occ_slice][mask], occ_winner_sorted[occ_slice][mask]


def select_local_free(
    free_idx_sorted: np.ndarray,
    free_cnt_sorted: np.ndarray,
    idx_min: np.ndarray,
    idx_max: np.ndarray,
):
    free_slice = query_sorted_x(free_idx_sorted, int(idx_min[0]), int(idx_max[0]))
    if len(free_slice) == 0:
        return np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.int64)
    free_idx = free_idx_sorted[free_slice]
    mask = (
        (free_idx[:, 1] >= idx_min[1]) & (free_idx[:, 1] < idx_max[1]) &
        (free_idx[:, 2] >= idx_min[2]) & (free_idx[:, 2] < idx_max[2])
    )
    return free_idx[mask], free_cnt_sorted[free_slice][mask]


def build_local_target(
    occ_idx_sorted: np.ndarray,
    occ_total_sorted: np.ndarray,
    occ_winner_sorted: np.ndarray,
    free_idx_sorted: np.ndarray,
    free_cnt_sorted: np.ndarray,
    idx_min: np.ndarray,
    idx_max: np.ndarray,
    min_occ_votes: int,
    min_free_votes: int,
    occ_free_ratio: float,
    grid_shape: Tuple[int, int, int],
) -> np.ndarray:
    nx, ny, nz = grid_shape
    target = np.full((nx, ny, nz), IGNORE_SEMANTIC_ID, dtype=np.uint8)

    occ_idx, occ_total, occ_winner = select_local_occ(
        occ_idx_sorted, occ_total_sorted, occ_winner_sorted, idx_min, idx_max
    )
    free_idx, free_cnt = select_local_free(
        free_idx_sorted, free_cnt_sorted, idx_min, idx_max
    )

    local: Dict[Index3, list] = {}
    for idx, ov, sem in zip(occ_idx, occ_total, occ_winner):
        key = (int(idx[0]), int(idx[1]), int(idx[2]))
        local[key] = [int(ov), int(sem), 0]

    for idx, fv in zip(free_idx, free_cnt):
        key = (int(idx[0]), int(idx[1]), int(idx[2]))
        if key in local:
            local[key][2] = int(fv)
        else:
            local[key] = [0, IGNORE_SEMANTIC_ID, int(fv)]

    for key, (occ_votes, sem, free_votes) in local.items():
        lx = key[0] - int(idx_min[0])
        ly = key[1] - int(idx_min[1])
        lz = key[2] - int(idx_min[2])
        if not (0 <= lx < nx and 0 <= ly < ny and 0 <= lz < nz):
            continue
        if occ_votes >= min_occ_votes and occ_votes >= free_votes * occ_free_ratio:
            target[lx, ly, lz] = np.uint8(sem)
        elif free_votes >= min_free_votes:
            target[lx, ly, lz] = 0
    return target


def target_to_masks(target: np.ndarray):
    target = np.asarray(target)
    occ_mask = (target != 255) & (target != 0)
    free_mask = target == 0
    known_mask = target != 255
    sem_label = target.astype(np.uint8, copy=True)
    return (
        occ_mask.astype(np.uint8),
        free_mask.astype(np.uint8),
        known_mask.astype(np.uint8),
        sem_label,
    )


def voxel_indices_to_world_centers(idx_xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    return (idx_xyz.astype(np.float64) + 0.5) * voxel_size


def estimate_local_ground_z(
    occ_idx_sorted: np.ndarray,
    occ_winner_sorted: np.ndarray,
    cam_pos: np.ndarray,
    voxel_size: float,
    search_radius_m: float,
    ground_like_train_ids: list[int],
    min_points: int,
    percentile_if_ground_higher_than_cam: float,
    percentile_if_ground_lower_than_cam: float,
):
    rad_idx = int(np.ceil(search_radius_m / voxel_size))
    cx_idx = int(np.floor(cam_pos[0] / voxel_size))
    cy_idx = int(np.floor(cam_pos[1] / voxel_size))

    occ_slice = query_sorted_x(occ_idx_sorted, cx_idx - rad_idx, cx_idx + rad_idx + 1)
    if len(occ_slice) == 0:
        return None, None, {'reason': 'no_occ_in_x_slice'}

    occ_idx = occ_idx_sorted[occ_slice]
    occ_sem = occ_winner_sorted[occ_slice]
    dx = occ_idx[:, 0] - cx_idx
    dy = occ_idx[:, 1] - cy_idx
    mask_xy = (dx * dx + dy * dy) <= (rad_idx * rad_idx)
    occ_idx = occ_idx[mask_xy]
    occ_sem = occ_sem[mask_xy]
    if len(occ_idx) == 0:
        return None, None, {'reason': 'no_occ_in_xy_radius'}

    z_world = voxel_indices_to_world_centers(occ_idx, voxel_size)[:, 2]
    cam_z = float(cam_pos[2])

    ground_like = np.isin(occ_sem.astype(np.int32), np.asarray(ground_like_train_ids, dtype=np.int32))
    use_ground_like = int(ground_like.sum()) >= int(min_points)
    z_use = z_world[ground_like] if use_ground_like else z_world

    if len(z_use) < max(5, min_points // 2):
        return None, None, {'reason': 'too_few_points', 'n_points': int(len(z_use))}

    median_z = float(np.median(z_use))
    ground_dir = 1 if median_z > cam_z else -1
    percentile = percentile_if_ground_higher_than_cam if ground_dir > 0 else percentile_if_ground_lower_than_cam
    ground_z = float(np.percentile(z_use, percentile))
    return ground_z, ground_dir, {
        'reason': 'ok',
        'n_points': int(len(z_use)),
        'used_ground_like': bool(use_ground_like),
        'cam_z': cam_z,
        'median_z': median_z,
        'ground_z': ground_z,
        'ground_dir': int(ground_dir),
    }


def compute_ground_anchored_origin(
    cam_pos: np.ndarray,
    size_xy_m: np.ndarray,
    z_above_ground_m: float,
    z_below_ground_m: float,
    ground_z: float,
    ground_dir: int,
    center_offset_xy_m: np.ndarray,
):
    origin_xy = cam_pos[:2] + center_offset_xy_m - 0.5 * size_xy_m
    if ground_dir > 0:
        z_min = ground_z - z_above_ground_m
        z_max = ground_z + z_below_ground_m
    else:
        z_min = ground_z - z_below_ground_m
        z_max = ground_z + z_above_ground_m
    size_z = float(z_max - z_min)
    origin_world = np.array([origin_xy[0], origin_xy[1], z_min], dtype=np.float64)
    size_m = np.array([size_xy_m[0], size_xy_m[1], size_z], dtype=np.float64)
    return origin_world, size_m


def compute_fallback_origin(cam_pos: np.ndarray, size_m: np.ndarray, center_offset_m: np.ndarray) -> np.ndarray:
    center_world = cam_pos + center_offset_m
    return center_world - 0.5 * size_m


def prepare_sparse_votes_for_scene(sparse_path: str | Path):
    data = np.load(str(sparse_path), allow_pickle=False)
    occ_u_idx, occ_total, occ_winner = group_occ_votes(
        data['occ_idx'].astype(np.int32, copy=False),
        data['occ_cls'].astype(np.int16, copy=False),
        data['occ_cnt'].astype(np.int32, copy=False),
    )
    free_idx = data['free_idx'].astype(np.int32, copy=False)
    free_cnt = data['free_cnt'].astype(np.int32, copy=False)
    occ_u_idx, occ_total, occ_winner = sort_by_x(occ_u_idx, occ_total, occ_winner)
    free_idx, free_cnt = sort_by_x(free_idx, free_cnt)
    return occ_u_idx, occ_total, occ_winner, free_idx, free_cnt


def _camera_axes_world(T_world_cam: np.ndarray) -> dict[str, np.ndarray]:
    R = T_world_cam[:3, :3]
    return {
        '+x': R[:, 0],
        '-x': -R[:, 0],
        '+y': R[:, 1],
        '-y': -R[:, 1],
        '+z': R[:, 2],
        '-z': -R[:, 2],
    }


def choose_camera_forward_axis(T_world_cam: np.ndarray, axis_mode: str, z_ground_estimate: float):
    axes = _camera_axes_world(T_world_cam)
    if axis_mode in axes:
        vec = axes[axis_mode]
        return vec / (np.linalg.norm(vec) + 1e-12), axis_mode

    cam_z = float(T_world_cam[2, 3])
    toward_ground_sign = 1.0 if z_ground_estimate > cam_z else -1.0
    best_name = '+z'
    best_vec = axes['+z']
    best_score = -1e18
    for name, vec in axes.items():
        score = toward_ground_sign * float(vec[2])
        if score > best_score:
            best_score = score
            best_name = name
            best_vec = vec
    best_vec = best_vec / (np.linalg.norm(best_vec) + 1e-12)
    return best_vec, best_name


def project_focus_to_ground(cam_pos: np.ndarray, look_dir: np.ndarray, z_ground_estimate: float):
    if abs(float(look_dir[2])) > 1e-4:
        t = (float(z_ground_estimate) - float(cam_pos[2])) / float(look_dir[2])
        if t > 0:
            return cam_pos[:2] + (t * look_dir[:2]), float(t)
    return cam_pos[:2].copy(), 0.0


def compute_local_box(
    T_world_cam: np.ndarray,
    occ_u_idx: np.ndarray,
    occ_winner: np.ndarray,
    voxel_size: float,
    local_grid_cfg: dict,
):
    size_m_cfg = np.asarray(local_grid_cfg['size_m'], dtype=np.float64)
    size_xy_m = size_m_cfg[:2]
    fallback_center_offset = np.asarray(
        local_grid_cfg.get('fallback_center_offset_m', local_grid_cfg.get('center_offset_m', [0.0, 0.0, 0.0])),
        dtype=np.float64,
    )
    center_offset_xy = fallback_center_offset[:2]
    anchor_mode = str(local_grid_cfg.get('anchor_mode', 'local_ground'))
    search_radius_m = float(local_grid_cfg.get('ground_search_radius_m', 18.0))
    ground_min_points = int(local_grid_cfg.get('ground_min_points', 30))
    ground_like_train_ids = [int(x) for x in local_grid_cfg.get('ground_like_train_ids', [2, 3, 8, 10, 11, 15, 16])]
    pct_high = float(local_grid_cfg.get('ground_percentile_if_ground_higher_than_cam', 80.0))
    pct_low = float(local_grid_cfg.get('ground_percentile_if_ground_lower_than_cam', 20.0))
    z_above_ground_m = float(local_grid_cfg.get('z_above_ground_m', max(1.0, size_m_cfg[2] - 2.0)))
    z_below_ground_m = float(local_grid_cfg.get('z_below_ground_m', 2.0))
    axis_mode = str(local_grid_cfg.get('camera_forward_axis', 'auto')).lower()

    cam_pos = T_world_cam[:3, 3]
    if len(occ_u_idx) > 0:
        z_ground_estimate = float(np.median((occ_u_idx[:, 2].astype(np.float64) + 0.5) * voxel_size))
    else:
        z_ground_estimate = float(cam_pos[2])

    look_dir, look_axis_name = choose_camera_forward_axis(T_world_cam, axis_mode, z_ground_estimate)
    focus_xy, focus_t = project_focus_to_ground(cam_pos, look_dir, z_ground_estimate)
    focus_pos = np.array([focus_xy[0], focus_xy[1], cam_pos[2]], dtype=np.float64)

    ground_dbg = {
        'reason': 'disabled',
        'look_axis_name': look_axis_name,
        'focus_t': float(focus_t),
        'focus_shift_xy_m': float(np.linalg.norm(focus_xy - cam_pos[:2])),
        'global_ground_estimate_z': float(z_ground_estimate),
    }

    if anchor_mode == 'local_ground':
        ground_z, ground_dir, gdbg = estimate_local_ground_z(
            occ_idx_sorted=occ_u_idx,
            occ_winner_sorted=occ_winner,
            cam_pos=focus_pos,
            voxel_size=voxel_size,
            search_radius_m=search_radius_m,
            ground_like_train_ids=ground_like_train_ids,
            min_points=ground_min_points,
            percentile_if_ground_higher_than_cam=pct_high,
            percentile_if_ground_lower_than_cam=pct_low,
        )
        ground_dbg.update(gdbg)
        if ground_z is not None and ground_dir is not None:
            origin_world, size_m = compute_ground_anchored_origin(
                cam_pos=focus_pos,
                size_xy_m=size_xy_m,
                z_above_ground_m=z_above_ground_m,
                z_below_ground_m=z_below_ground_m,
                ground_z=ground_z,
                ground_dir=ground_dir,
                center_offset_xy_m=center_offset_xy,
            )
            used_local_ground = True
        else:
            size_m = size_m_cfg.copy()
            origin_world = compute_fallback_origin(focus_pos, size_m, fallback_center_offset)
            used_local_ground = False
    else:
        size_m = size_m_cfg.copy()
        origin_world = compute_fallback_origin(focus_pos, size_m, fallback_center_offset)
        used_local_ground = False

    nxyz = np.round(size_m / voxel_size).astype(np.int32)
    idx_min = np.floor(origin_world / voxel_size).astype(np.int32)
    idx_max = idx_min + nxyz
    return {
        'cam_pos': cam_pos.astype(np.float64),
        'focus_pos': focus_pos.astype(np.float64),
        'origin_world': origin_world.astype(np.float64),
        'size_m': size_m.astype(np.float64),
        'grid_size_xyz': nxyz.astype(np.int32),
        'idx_min': idx_min.astype(np.int32),
        'idx_max': idx_max.astype(np.int32),
        'ground_debug': ground_dbg,
        'used_local_ground': bool(used_local_ground),
        'look_axis_name': look_axis_name,
        'look_dir': look_dir.astype(np.float64),
    }


def build_dense_lidar_inputs(
    pts_world: np.ndarray,
    origin_world: np.ndarray,
    voxel_size: float,
    grid_shape: tuple[int, int, int],
    ground_z: float | None = None,
):
    nx, ny, nz = [int(x) for x in grid_shape]
    input_occ_lidar = np.zeros((nx, ny, nz), dtype=np.uint8)
    input_density_lidar = np.zeros((nx, ny, nz), dtype=np.uint16)
    input_max_rel_height = np.zeros((nx, ny, nz), dtype=np.float32)
    input_mean_rel_height = np.zeros((nx, ny, nz), dtype=np.float32)

    if pts_world.size == 0:
        return {
            'input_occ_lidar': input_occ_lidar,
            'input_density_lidar': input_density_lidar,
            'input_max_rel_height': input_max_rel_height,
            'input_mean_rel_height': input_mean_rel_height,
            'lidar_sparse_coords': np.empty((0, 3), dtype=np.int32),
            'lidar_sparse_counts': np.empty((0,), dtype=np.int32),
            'lidar_points_local_xyz': np.empty((0, 3), dtype=np.float32),
        }

    local_xyz_m = pts_world.astype(np.float64) - origin_world[None, :]
    local_idx = np.floor(local_xyz_m / float(voxel_size)).astype(np.int32)
    ok = (
        (local_idx[:, 0] >= 0) & (local_idx[:, 0] < nx) &
        (local_idx[:, 1] >= 0) & (local_idx[:, 1] < ny) &
        (local_idx[:, 2] >= 0) & (local_idx[:, 2] < nz)
    )
    local_idx = local_idx[ok]
    local_xyz_m = local_xyz_m[ok]

    if len(local_idx) == 0:
        return {
            'input_occ_lidar': input_occ_lidar,
            'input_density_lidar': input_density_lidar,
            'input_max_rel_height': input_max_rel_height,
            'input_mean_rel_height': input_mean_rel_height,
            'lidar_sparse_coords': np.empty((0, 3), dtype=np.int32),
            'lidar_sparse_counts': np.empty((0,), dtype=np.int32),
            'lidar_points_local_xyz': np.empty((0, 3), dtype=np.float32),
        }

    uniq, inv, counts = np.unique(local_idx, axis=0, return_inverse=True, return_counts=True)
    ground_ref = float(ground_z) if ground_z is not None else float(origin_world[2])
    rel_height = local_xyz_m[:, 2] + float(origin_world[2]) - ground_ref

    sum_h = np.zeros(len(uniq), dtype=np.float64)
    max_h = np.full(len(uniq), -np.inf, dtype=np.float64)
    np.add.at(sum_h, inv, rel_height)
    np.maximum.at(max_h, inv, rel_height)
    mean_h = sum_h / np.maximum(counts.astype(np.float64), 1.0)

    for i, (lx, ly, lz) in enumerate(uniq):
        input_occ_lidar[lx, ly, lz] = 1
        input_density_lidar[lx, ly, lz] = min(int(counts[i]), np.iinfo(np.uint16).max)
        input_max_rel_height[lx, ly, lz] = float(max_h[i]) if np.isfinite(max_h[i]) else 0.0
        input_mean_rel_height[lx, ly, lz] = float(mean_h[i]) if np.isfinite(mean_h[i]) else 0.0

    return {
        'input_occ_lidar': input_occ_lidar,
        'input_density_lidar': input_density_lidar,
        'input_max_rel_height': input_max_rel_height.astype(np.float32),
        'input_mean_rel_height': input_mean_rel_height.astype(np.float32),
        'lidar_sparse_coords': uniq.astype(np.int32),
        'lidar_sparse_counts': counts.astype(np.int32),
        'lidar_points_local_xyz': local_xyz_m.astype(np.float32),
    }


def _cell_to_path_string(value) -> str | None:
    """Return a clean path string from a pandas cell, or None for empty/NaN."""
    if value is None:
        return None
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
    except Exception:
        pass
    s = str(value)
    if not s or s.lower() in {'nan', 'none', '<na>'}:
        return None
    return s


def add_cam_label_metadata(sample: dict, row, cfg: dict, img_shape_hw: tuple[int, int] | None = None) -> dict:
    """Attach UAVScenes CAM_label metadata to an exported NPZ sample.

    The new UAVScenes layout stores 2D semantic masks in sibling folders such as
    interval1_CAM_label/label_id and interval1_CAM_label/label_color.  These masks
    are not the 3D SSC target; they are paired by timestamp and are useful for
    QC overlays or optional 2D auxiliary supervision.
    """
    cam_cfg = cfg.get('cam_label_export', {}) if isinstance(cfg, dict) else {}
    save_paths = bool(cam_cfg.get('save_paths', True))
    save_id_image = bool(cam_cfg.get('save_id_image', False))
    save_rgb_image = bool(cam_cfg.get('save_rgb_image', False))

    id_path = _cell_to_path_string(row.get('cam_label_id_path', None)) if hasattr(row, 'get') else None
    rgb_path = _cell_to_path_string(row.get('cam_label_rgb_path', None)) if hasattr(row, 'get') else None

    if save_paths:
        sample['cam_label_id_path'] = np.array([id_path or ''])
        sample['cam_label_rgb_path'] = np.array([rgb_path or ''])

    if img_shape_hw is not None:
        sample['image_shape_hw'] = np.asarray(img_shape_hw, dtype=np.int32)

    if id_path and save_id_image and Path(id_path).exists():
        from .io import read_cam_label_id_image
        mask = read_cam_label_id_image(id_path)
        sample['cam_label_id'] = mask.astype(np.uint16, copy=False)
        sample['cam_label_shape_hw'] = np.asarray(mask.shape[:2], dtype=np.int32)

    if rgb_path and save_rgb_image and Path(rgb_path).exists():
        from .io import read_cam_label_rgb_image
        rgb = read_cam_label_rgb_image(rgb_path)
        sample['cam_label_rgb'] = rgb.astype(np.uint8, copy=False)
        sample['cam_label_rgb_shape_hw'] = np.asarray(rgb.shape[:2], dtype=np.int32)

    return sample
