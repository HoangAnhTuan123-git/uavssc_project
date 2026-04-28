
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rich import print
from tqdm import tqdm

from uavssc.export_common import (
    build_dense_lidar_inputs,
    build_local_target,
    compute_local_box,
    parse_matrix_cell,
    prepare_sparse_votes_for_scene,
    target_to_masks,
)
from uavssc.io import (
    calibration_dict_to_matrices,
    find_first_existing,
    infer_scene_calibration_dict,
    parse_calibration_results_py,
    read_lidar_txt,
)
from uavssc.transforms import apply_transform
from uavssc.utils import ensure_dir, load_yaml


def main() -> None:
    ap = argparse.ArgumentParser(description='Export LiDAR-only SSC samples for LMSCNet/SCPNet-style training.')
    ap.add_argument('--manifest', type=str, required=True)
    ap.add_argument('--sparse-votes', type=str, required=True)
    ap.add_argument('--data-root', type=str, required=True)
    ap.add_argument('--config', type=str, default='configs/default.yaml')
    ap.add_argument('--output', type=str, required=True)
    ap.add_argument('--ext-mode', type=str, default='cam_from_lidar', choices=['cam_from_lidar', 'lidar_from_cam'])
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    voxel_size = float(cfg['voxel']['voxel_size_m'])
    min_occ_votes = int(cfg['voxel']['min_occ_votes'])
    min_free_votes = int(cfg['voxel']['min_free_votes'])
    occ_free_ratio = float(cfg['voxel']['occ_free_ratio'])
    sample_every = int(cfg['local_grid']['sample_every_nth_frame'])

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    df = pd.read_parquet(args.manifest) if args.manifest.endswith('.parquet') else pd.read_csv(args.manifest)
    df = df[df['T_world_cam'].notna() & df['lidar_path'].notna()].copy().reset_index(drop=True)

    for scene, sdf in df.groupby('scene'):
        sparse_path = Path(args.sparse_votes) / f'{scene}_sparse_votes.npz'
        if not sparse_path.exists():
            print(f'[yellow]Missing sparse vote file for {scene}: {sparse_path}[/yellow]')
            continue

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
        T_cam_lidar = T_ext if args.ext_mode == 'cam_from_lidar' else np.linalg.inv(T_ext)

        occ_u_idx, occ_total, occ_winner, free_idx, free_cnt = prepare_sparse_votes_for_scene(sparse_path)
        scene_out = ensure_dir(out_root / scene)
        print(f'\n[bold cyan]Scene {scene}[/bold cyan] occ_vox={len(occ_u_idx):,} free_vox={len(free_idx):,} samples={len(sdf):,}')

        written = 0
        for i, (_, row) in enumerate(tqdm(sdf.iterrows(), total=len(sdf), desc=f'export {scene}')):
            if i % sample_every != 0:
                continue

            T_world_cam = parse_matrix_cell(row['T_world_cam'])
            if T_world_cam is None:
                continue

            box = compute_local_box(T_world_cam, occ_u_idx, occ_winner, voxel_size, cfg['local_grid'])
            nx, ny, nz = [int(v) for v in box['grid_size_xyz']]
            target = build_local_target(
                occ_idx_sorted=occ_u_idx,
                occ_total_sorted=occ_total,
                occ_winner_sorted=occ_winner,
                free_idx_sorted=free_idx,
                free_cnt_sorted=free_cnt,
                idx_min=box['idx_min'],
                idx_max=box['idx_max'],
                min_occ_votes=min_occ_votes,
                min_free_votes=min_free_votes,
                occ_free_ratio=occ_free_ratio,
                grid_shape=(nx, ny, nz),
            )
            occ_mask, free_mask, known_mask, sem_label = target_to_masks(target)

            T_world_lidar = T_world_cam @ T_cam_lidar
            pts_lidar = read_lidar_txt(row['lidar_path'])
            pts_world = apply_transform(pts_lidar, T_world_lidar)
            dense_inputs = build_dense_lidar_inputs(
                pts_world=pts_world,
                origin_world=box['origin_world'],
                voxel_size=voxel_size,
                grid_shape=(nx, ny, nz),
                ground_z=float(box['ground_debug'].get('ground_z', box['origin_world'][2])) if box['ground_debug'].get('reason') == 'ok' else None,
            )

            sensor_origin_local = (T_world_lidar[:3, 3] - box['origin_world']).astype(np.float32)

            sample = {
                'scene': np.array([scene]),
                'timestamp': np.array([float(row['timestamp'])], dtype=np.float64),
                'img_path': np.array([str(row['img_path'])]) if isinstance(row.get('img_path', None), str) else np.array(['']),
                'lidar_path': np.array([str(row['lidar_path'])]),
                'T_world_cam': T_world_cam.astype(np.float32),
                'T_world_lidar': T_world_lidar.astype(np.float32),
                'vox_origin': box['origin_world'].astype(np.float32),
                'voxel_size': np.array([voxel_size], dtype=np.float32),
                'scene_size_m': box['size_m'].astype(np.float32),
                'grid_size_xyz': box['grid_size_xyz'].astype(np.int32),
                'target': target,
                'occ_mask': occ_mask,
                'free_mask': free_mask,
                'known_mask': known_mask,
                'sem_label': sem_label,
                'focus_pos': box['focus_pos'].astype(np.float32),
                'look_dir': box['look_dir'].astype(np.float32),
                'look_axis_name': np.array([box['look_axis_name']]),
                'sensor_origin_local_m': sensor_origin_local,
                'ground_debug_reason': np.array([box['ground_debug'].get('reason', 'unknown')]),
                'ground_debug_ground_z': np.array([float(box['ground_debug'].get('ground_z', np.nan))], dtype=np.float32),
                'focus_shift_xy_m': np.array([float(box['ground_debug'].get('focus_shift_xy_m', 0.0))], dtype=np.float32),
                **dense_inputs,
            }
            out_path = scene_out / f"{Path(row['lidar_path']).stem}.npz"
            np.savez_compressed(out_path, **sample)
            written += 1

        print(f'[green]Wrote[/green] {written:,} LiDAR SSC samples to {scene_out}')


if __name__ == '__main__':
    main()
