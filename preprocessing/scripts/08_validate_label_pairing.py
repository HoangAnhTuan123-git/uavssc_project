from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rich import print
from tqdm import tqdm

from uavssc.io import read_label_id_txt, read_label_rgb_txt, read_lidar_txt



def main() -> None:
    ap = argparse.ArgumentParser(description='Check that LiDAR points and label txt files align row-by-row.')
    ap.add_argument('--manifest', type=str, required=True)
    ap.add_argument('--max-samples', type=int, default=0, help='0 means all samples')
    args = ap.parse_args()

    df = pd.read_parquet(args.manifest) if args.manifest.endswith('.parquet') else pd.read_csv(args.manifest)
    df = df[df['lidar_path'].notna() & df['lidar_label_id_path'].notna() & df['lidar_label_rgb_path'].notna()].copy()
    if args.max_samples > 0:
        df = df.iloc[:args.max_samples].copy()

    n_ok = 0
    n_bad = 0
    bad_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='validate'):
        lidar_path = Path(row['lidar_path'])
        id_path = Path(row['lidar_label_id_path'])
        rgb_path = Path(row['lidar_label_rgb_path'])
        try:
            n_pts = read_lidar_txt(lidar_path).shape[0]
            n_id = read_label_id_txt(id_path).shape[0]
            n_rgb = read_label_rgb_txt(rgb_path).shape[0]
            if n_pts == n_id == n_rgb:
                n_ok += 1
            else:
                n_bad += 1
                bad_rows.append({
                    'scene': row['scene'],
                    'timestamp': row['timestamp'],
                    'lidar': str(lidar_path),
                    'label_id': str(id_path),
                    'label_rgb': str(rgb_path),
                    'n_pts': n_pts,
                    'n_id': n_id,
                    'n_rgb': n_rgb,
                })
        except Exception as e:
            n_bad += 1
            bad_rows.append({
                'scene': row['scene'],
                'timestamp': row['timestamp'],
                'lidar': str(lidar_path),
                'label_id': str(id_path),
                'label_rgb': str(rgb_path),
                'error': repr(e),
            })

    print(f'[green]OK samples[/green]: {n_ok}')
    print(f'[red]Bad samples[/red]: {n_bad}')
    if bad_rows:
        out = Path('artifacts') / 'label_pairing_report.csv'
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(bad_rows).to_csv(out, index=False)
        print(f'[yellow]Wrote mismatch report[/yellow]: {out}')
        print(pd.DataFrame(bad_rows).head(20))


if __name__ == '__main__':
    main()
