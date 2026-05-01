
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rich import print
from tqdm import tqdm

from uavssc.export_common import (
    add_cam_label_metadata,
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
    read_image,
    read_lidar_txt,
)
from uavssc.monoscene_utils import downsample_label, vox2pix
from uavssc.transforms import apply_transform
from uavssc.utils import ensure_dir, load_yaml


def main() -> None:
    ap = argparse.ArgumentParser(description='Export RGB+LiDAR fusion SSC samples.')
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
    project_scales = [int(x) for x in cfg['monoscene']['project_scales']]

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    df = pd.read_parquet(args.manifest) if args.manifest.endswith('.parquet') else pd.read_csv(args.manifest)
    df = df[df['T_world_cam'].notna() & df['lidar_path'].notna() & df['img_path'].notna() & df['K'].notna()].copy().reset_index(drop=True)

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
            K = parse_matrix_cell(row['K'])
            if T_world_cam is None or K is None:
                continue

            img = read_image(row['img_path'])
            img_H, img_W = img.shape[:2]
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

            T_cam_world = np.linalg.inv(T_world_cam)
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

            sample = {
                'scene': np.array([scene]),
                'timestamp': np.array([float(row['timestamp'])], dtype=np.float64),
                'img_path': np.array([str(row['img_path'])]),
                'lidar_path': np.array([str(row['lidar_path'])]),
                'cam_k': K.astype(np.float32),
                'cam_E': T_cam_world.astype(np.float32),
                'T_world_cam': T_world_cam.astype(np.float32),
                'T_world_lidar': T_world_lidar.astype(np.float32),
                'vox_origin': box['origin_world'].astype(np.float32),
                'voxel_size': np.array([voxel_size], dtype=np.float32),
                'scene_size_m': box['size_m'].astype(np.float32),
                'grid_size_xyz': box['grid_size_xyz'].astype(np.int32),
                'target': target,
                'target_1_8': downsample_label(target, factor=8).astype(np.uint8),
                'occ_mask': occ_mask,
                'free_mask': free_mask,
                'known_mask': known_mask,
                'sem_label': sem_label,
                'focus_pos': box['focus_pos'].astype(np.float32),
                'look_dir': box['look_dir'].astype(np.float32),
                'look_axis_name': np.array([box['look_axis_name']]),
                'sensor_origin_local_m': (T_world_lidar[:3, 3] - box['origin_world']).astype(np.float32),
                'ground_debug_reason': np.array([box['ground_debug'].get('reason', 'unknown')]),
                'ground_debug_ground_z': np.array([float(box['ground_debug'].get('ground_z', np.nan))], dtype=np.float32),
                **dense_inputs,
            }

            for scale in project_scales:
                projected_pix, fov_mask, pix_z = vox2pix(
                    cam_E=T_cam_world,
                    cam_k=K,
                    vox_origin=box['origin_world'],
                    voxel_size=voxel_size * scale,
                    img_W=img_W,
                    img_H=img_H,
                    scene_size=tuple(box['size_m'].tolist()),
                )
                sample[f'projected_pix_{scale}'] = projected_pix.astype(np.int32)
                sample[f'fov_mask_{scale}'] = fov_mask.astype(bool)
                sample[f'pix_z_{scale}'] = pix_z.astype(np.float32)

            add_cam_label_metadata(sample, row, cfg, img_shape_hw=(img_H, img_W))

            out_path = scene_out / f"{Path(row['img_path']).stem}.npz"
            np.savez_compressed(out_path, **sample)
            written += 1

        print(f'[green]Wrote[/green] {written:,} fusion SSC samples to {scene_out}')


if __name__ == '__main__':
    main()
