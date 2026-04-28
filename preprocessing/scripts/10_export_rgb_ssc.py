
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rich import print
from tqdm import tqdm

from uavssc.export_common import (
    build_local_target,
    compute_local_box,
    parse_matrix_cell,
    prepare_sparse_votes_for_scene,
)
from uavssc.io import read_image
from uavssc.monoscene_utils import compute_CP_mega_matrix, compute_local_frustums, downsample_label, vox2pix
from uavssc.utils import ensure_dir, load_yaml


def main() -> None:
    ap = argparse.ArgumentParser(description='Export RGB-only SSC samples for MonoScene/CGFormer/VoxFormer-style training.')
    ap.add_argument('--manifest', type=str, required=True)
    ap.add_argument('--sparse-votes', type=str, required=True)
    ap.add_argument('--config', type=str, default='configs/default.yaml')
    ap.add_argument('--output', type=str, required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    voxel_size = float(cfg['voxel']['voxel_size_m'])
    min_occ_votes = int(cfg['voxel']['min_occ_votes'])
    min_free_votes = int(cfg['voxel']['min_free_votes'])
    occ_free_ratio = float(cfg['voxel']['occ_free_ratio'])
    sample_every = int(cfg['local_grid']['sample_every_nth_frame'])

    monoscene_cfg = cfg['monoscene']
    n_classes = int(monoscene_cfg['n_classes'])
    project_scales = [int(x) for x in monoscene_cfg['project_scales']]
    frustum_size = int(monoscene_cfg['frustum_size'])
    build_cp_matrix = bool(monoscene_cfg['build_cp_matrix'])
    build_frustums = bool(monoscene_cfg['build_frustums'])

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.manifest) if args.manifest.endswith('.parquet') else pd.read_csv(args.manifest)
    df = df[df['T_world_cam'].notna() & df['img_path'].notna() & df['K'].notna()].copy().reset_index(drop=True)

    for scene, sdf in df.groupby('scene'):
        sparse_path = Path(args.sparse_votes) / f'{scene}_sparse_votes.npz'
        if not sparse_path.exists():
            print(f'[yellow]Missing sparse vote file for {scene}: {sparse_path}[/yellow]')
            continue

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

            T_cam_world = np.linalg.inv(T_world_cam)
            sample = {
                'scene': np.array([scene]),
                'timestamp': np.array([float(row['timestamp'])], dtype=np.float64),
                'img_path': np.array([str(row['img_path'])]),
                'cam_k': K.astype(np.float32),
                'cam_E': T_cam_world.astype(np.float32),
                'T_world_cam': T_world_cam.astype(np.float32),
                'vox_origin': box['origin_world'].astype(np.float32),
                'voxel_size': np.array([voxel_size], dtype=np.float32),
                'scene_size_m': box['size_m'].astype(np.float32),
                'grid_size_xyz': box['grid_size_xyz'].astype(np.int32),
                'target': target,
                'focus_pos': box['focus_pos'].astype(np.float32),
                'look_dir': box['look_dir'].astype(np.float32),
                'look_axis_name': np.array([box['look_axis_name']]),
                'used_local_ground': np.array([box['used_local_ground']]),
                'ground_debug_reason': np.array([box['ground_debug'].get('reason', 'unknown')]),
                'ground_debug_cam_z': np.array([float(box['ground_debug'].get('cam_z', np.nan))], dtype=np.float32),
                'ground_debug_median_z': np.array([float(box['ground_debug'].get('median_z', np.nan))], dtype=np.float32),
                'ground_debug_ground_z': np.array([float(box['ground_debug'].get('ground_z', np.nan))], dtype=np.float32),
                'ground_debug_ground_dir': np.array([int(box['ground_debug'].get('ground_dir', 0))], dtype=np.int8),
                'ground_debug_n_points': np.array([int(box['ground_debug'].get('n_points', 0))], dtype=np.int32),
                'focus_t': np.array([float(box['ground_debug'].get('focus_t', 0.0))], dtype=np.float32),
                'focus_shift_xy_m': np.array([float(box['ground_debug'].get('focus_shift_xy_m', 0.0))], dtype=np.float32),
                'global_ground_estimate_z': np.array([float(box['ground_debug'].get('global_ground_estimate_z', np.nan))], dtype=np.float32),
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

            target_1_8 = downsample_label(target, factor=8).astype(np.uint8)
            sample['target_1_8'] = target_1_8
            if build_cp_matrix:
                sample['CP_mega_matrix'] = compute_CP_mega_matrix(target_1_8).astype(np.uint8)
            if build_frustums:
                frustums_masks, frustums_class_dists = compute_local_frustums(
                    projected_pix=sample['projected_pix_1'],
                    pix_z=sample['pix_z_1'],
                    target=target,
                    img_W=img_W,
                    img_H=img_H,
                    n_classes=n_classes,
                    size=frustum_size,
                )
                sample['frustums_masks'] = frustums_masks.astype(bool)
                sample['frustums_class_dists'] = frustums_class_dists.astype(np.int32)

            out_path = scene_out / f"{Path(row['img_path']).stem}.npz"
            np.savez_compressed(out_path, **sample)
            written += 1

        print(f'[green]Wrote[/green] {written:,} RGB SSC samples to {scene_out}')


if __name__ == '__main__':
    main()
