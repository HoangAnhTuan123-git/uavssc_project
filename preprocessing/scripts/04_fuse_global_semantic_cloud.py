
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from rich import print
from tqdm import tqdm

from uavssc.io import (
    calibration_dict_to_matrices,
    find_first_existing,
    infer_scene_calibration_dict,
    parse_calibration_results_py,
    read_label_id_txt,
    read_lidar_txt,
)
from uavssc.transforms import apply_transform
from uavssc.utils import load_yaml
from uavssc.voxel import SparseVoxelVotes, point_to_index, ray_voxel_indices, unique_rows_with_majority_label


def parse_matrix_cell(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, str):
        return np.asarray(ast.literal_eval(x), dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def map_raw_to_train(raw_ids: np.ndarray, cfg: dict) -> np.ndarray:
    mapping = {int(k): int(v) for k, v in cfg['training_taxonomy']['map_raw_to_train'].items()}
    ignore = set(int(x) for x in cfg['training_taxonomy']['ignore_raw_ids'])
    out = np.full(raw_ids.shape, 255, dtype=np.uint8)
    for rid, tid in mapping.items():
        out[raw_ids == rid] = tid
    for rid in ignore:
        out[raw_ids == rid] = 255
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=str, required=True)
    ap.add_argument('--manifest', type=str, required=True)
    ap.add_argument('--config', type=str, default='configs/default.yaml')
    ap.add_argument('--output', type=str, required=True)
    ap.add_argument('--ext-mode', type=str, default='cam_from_lidar', choices=['cam_from_lidar', 'lidar_from_cam'])
    ap.add_argument('--max-rays-per-frame', type=int, default=5000)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    voxel_size = float(cfg['voxel']['voxel_size_m'])
    ray_step_ratio = float(cfg['voxel']['ray_step_ratio'])
    data_root = Path(args.data_root)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.manifest) if args.manifest.endswith('.parquet') else pd.read_csv(args.manifest)
    df = df[df['lidar_path'].notna() & df['lidar_label_id_path'].notna() & df['T_world_cam'].notna()].copy()

    for scene, sdf in df.groupby('scene'):
        print(f'\n[bold cyan]Scene {scene}[/bold cyan] with {len(sdf)} samples')
        scene_root = data_root / scene
        calib_py = find_first_existing(scene_root, ['calibration_results.py'])
        if calib_py is None:
            calib_py = find_first_existing(data_root, ['calibration_results.py'])
        if calib_py is None:
            print(f'[yellow]Missing calibration_results.py for {scene}, skipping.[/yellow]')
            continue
        calib_vars = parse_calibration_results_py(calib_py)
        calib_dict = infer_scene_calibration_dict(calib_vars, scene)
        _, _, T_ext = calibration_dict_to_matrices(calib_dict)
        if T_ext is None:
            print(f'[yellow]Missing camera-LiDAR extrinsic for {scene}, skipping.[/yellow]')
            continue

        votes = SparseVoxelVotes(voxel_size=voxel_size)
        n_occ_pts = 0
        n_free_vox = 0

        for _, row in tqdm(sdf.iterrows(), total=len(sdf), desc=f'fuse {scene}'):
            T_world_cam = parse_matrix_cell(row['T_world_cam'])
            if T_world_cam is None:
                continue
            T_cam_lidar = T_ext if args.ext_mode == 'cam_from_lidar' else np.linalg.inv(T_ext)
            T_world_lidar = T_world_cam @ T_cam_lidar

            pts_lidar_all = read_lidar_txt(row['lidar_path'])
            raw_ids = read_label_id_txt(row['lidar_label_id_path'])
            if pts_lidar_all.shape[0] != raw_ids.shape[0]:
                print(f'[red]Row mismatch[/red] {row["lidar_path"]}')
                continue

            train_ids = map_raw_to_train(raw_ids, cfg)
            pts_world_all = apply_transform(pts_lidar_all, T_world_lidar)

            valid_sem = train_ids != 255
            if valid_sem.any():
                pts_world_sem = pts_world_all[valid_sem]
                cls = train_ids[valid_sem].astype(np.int32)
                occ_idx = point_to_index(pts_world_sem, voxel_size)
                occ_idx, cls = unique_rows_with_majority_label(occ_idx, cls)
                votes.add_occupied(occ_idx, cls)
                n_occ_pts += len(cls)

            sensor_origin_world = T_world_lidar[:3, 3]
            if pts_world_all.shape[0] > 0:
                if args.max_rays_per_frame > 0:
                    n = min(args.max_rays_per_frame, pts_world_all.shape[0])
                    choose = np.linspace(0, pts_world_all.shape[0] - 1, n).astype(np.int64)
                    pts_for_rays = pts_world_all[choose]
                else:
                    pts_for_rays = pts_world_all
                for hit in pts_for_rays:
                    free_idx = ray_voxel_indices(
                        sensor_origin_world,
                        hit,
                        voxel_size=voxel_size,
                        step_ratio=ray_step_ratio,
                        include_endpoint=False,
                    )
                    if free_idx.size > 0:
                        votes.add_free(free_idx)
                        n_free_vox += free_idx.shape[0]

        out_path = out_root / f'{scene}_sparse_votes.npz'
        votes.save_npz(out_path)
        print(f'[green]Saved[/green] {out_path}')
        print(f'  occupied vote points: {n_occ_pts}')
        print(f'  free-space voxel updates: {n_free_vox}')
        print(f'  global index bounds: min={votes.min_idx.tolist()}, max={votes.max_idx.tolist()}')


if __name__ == '__main__':
    main()
