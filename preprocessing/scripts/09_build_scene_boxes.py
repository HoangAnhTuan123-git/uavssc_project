from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from rich import print

from uavssc.utils import save_json



def parse_matrix_cell(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, str):
        return np.asarray(ast.literal_eval(x), dtype=np.float64)
    return np.asarray(x, dtype=np.float64)



def main() -> None:
    ap = argparse.ArgumentParser(description='Build per-scene map-frame bounding boxes from camera trajectories.')
    ap.add_argument('--manifest', type=str, required=True)
    ap.add_argument('--output', type=str, required=True)
    ap.add_argument('--margin-xy', type=float, default=60.0)
    ap.add_argument('--margin-z', type=float, default=20.0)
    args = ap.parse_args()

    df = pd.read_parquet(args.manifest) if args.manifest.endswith('.parquet') else pd.read_csv(args.manifest)
    df = df[df['T_world_cam'].notna()].copy()
    results = {}

    for scene, sdf in df.groupby('scene'):
        centers = []
        for x in sdf['T_world_cam']:
            T = parse_matrix_cell(x)
            if T is not None:
                centers.append(T[:3, 3])
        if not centers:
            continue
        xyz = np.asarray(centers, dtype=np.float64)
        xyz_min = xyz.min(axis=0)
        xyz_max = xyz.max(axis=0)
        xyz_min[:2] -= args.margin_xy
        xyz_max[:2] += args.margin_xy
        xyz_min[2] -= args.margin_z
        xyz_max[2] += args.margin_z
        results[scene] = {
            'xyz_min': xyz_min.round(6).tolist(),
            'xyz_max': xyz_max.round(6).tolist(),
            'n_frames': int(len(xyz)),
        }
        print(f'[cyan]{scene}[/cyan] min={results[scene]["xyz_min"]} max={results[scene]["xyz_max"]}')

    save_json(results, args.output, indent=2)
    print(f'[green]Saved[/green] {args.output}')


if __name__ == '__main__':
    main()
