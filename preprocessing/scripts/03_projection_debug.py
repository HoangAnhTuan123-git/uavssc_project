from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from rich import print

from uavssc.io import (
    calibration_dict_to_matrices,
    find_first_existing,
    infer_scene_calibration_dict,
    parse_calibration_results_py,
    read_image,
    read_lidar_txt,
)
from uavssc.projection import draw_projected_points, undistort_image
from uavssc.transforms import apply_transform
from uavssc.visualization import save_rgb



def parse_matrix_cell(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, str):
        return np.asarray(ast.literal_eval(x), dtype=np.float64)
    return np.asarray(x, dtype=np.float64)



def project_lidar_to_image(points_lidar: np.ndarray, T_cam_lidar: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts_cam = apply_transform(points_lidar, T_cam_lidar)
    z = pts_cam[:, 2]
    valid = z > 1e-6
    uv = np.full((pts_cam.shape[0], 2), np.nan, dtype=np.float64)
    if valid.any():
        x = pts_cam[valid, 0] / z[valid]
        y = pts_cam[valid, 1] / z[valid]
        uv_valid = (K @ np.stack([x, y, np.ones_like(x)], axis=0)).T
        uv[valid] = uv_valid[:, :2]
    return uv, valid



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=str, required=True)
    ap.add_argument('--manifest', type=str, required=True)
    ap.add_argument('--output-dir', type=str, required=True)
    ap.add_argument('--max-samples', type=int, default=20)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.manifest) if args.manifest.endswith('.parquet') else pd.read_csv(args.manifest)
    df = df[df['img_path'].notna() & df['lidar_path'].notna()].copy()
    print(f'[green]Projection debug samples:[/green] {len(df)}')

    by_scene = dict(tuple(df.groupby('scene')))
    saved = 0

    for scene, sdf in by_scene.items():
        scene_root = data_root / scene
        calib_py = find_first_existing(scene_root, ['calibration_results.py'])
        if calib_py is None:
            calib_py = find_first_existing(data_root, ['calibration_results.py'])
        if calib_py is None:
            print(f'[yellow]No calibration_results.py found for {scene}, skipping.[/yellow]')
            continue
        calib_vars = parse_calibration_results_py(calib_py)
        calib_dict = infer_scene_calibration_dict(calib_vars, scene)
        if calib_dict is None:
            print(f'[yellow]No scene calibration dict found for {scene}, skipping.[/yellow]')
            continue
        K_scene, D_scene, T_ext = calibration_dict_to_matrices(calib_dict)
        if T_ext is None or K_scene is None:
            print(f'[yellow]Incomplete scene calibration for {scene}, skipping.[/yellow]')
            continue

        for _, row in sdf.head(args.max_samples).iterrows():
            img = read_image(row['img_path'])
            pts = read_lidar_txt(row['lidar_path'])
            K = parse_matrix_cell(row.get('K', None))
            if K is None:
                K = K_scene
            D = parse_matrix_cell(row.get('dist', None))
            if D is None:
                D = D_scene
            img_und = undistort_image(img, K, D)

            # Hypothesis A: calibration ext is T_cam_lidar (lidar -> camera)
            uv_a, valid_a = project_lidar_to_image(pts, T_ext, K)
            overlay_a = draw_projected_points(img_und, uv_a, valid_a, color=(0, 255, 0), radius=2)

            # Hypothesis B: calibration ext is T_lidar_cam (camera -> lidar), so invert it.
            uv_b, valid_b = project_lidar_to_image(pts, np.linalg.inv(T_ext), K)
            overlay_b = draw_projected_points(img_und, uv_b, valid_b, color=(255, 0, 0), radius=2)

            ts = Path(row['img_path']).stem
            scene_dir = out_dir / scene
            scene_dir.mkdir(parents=True, exist_ok=True)
            save_rgb(scene_dir / f'{ts}_hypA_lidar_to_cam.png', overlay_a)
            save_rgb(scene_dir / f'{ts}_hypB_inv_ext.png', overlay_b)
            saved += 2

    print(f'[bold cyan]Saved {saved} overlay images to[/bold cyan] {out_dir}')
    print('Interpretation: whichever hypothesis yields tighter alignment is the correct extrinsic direction.')


if __name__ == '__main__':
    main()
