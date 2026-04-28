
from __future__ import annotations

import argparse
from pathlib import Path

from rich import print

from uavssc.io import discover_scene_dirs
from uavssc.manifest import build_manifest_for_scene, manifest_to_dataframe
from uavssc.utils import load_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=str, required=True)
    ap.add_argument('--output', type=str, required=True)
    ap.add_argument('--config', type=str, default='configs/default.yaml')
    args = ap.parse_args()

    cfg = load_yaml(args.config) if Path(args.config).exists() else {}
    data_root = Path(args.data_root)
    records = []
    for scene_root in discover_scene_dirs(data_root):
        recs = build_manifest_for_scene(scene_root, cfg=cfg)
        print(f'[green]{scene_root.name}[/green]: {len(recs)} records')
        records.extend(recs)

    df = manifest_to_dataframe(records)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == '.parquet':
        df.to_parquet(out, index=False)
    elif out.suffix == '.csv':
        df.to_csv(out, index=False)
    else:
        raise ValueError('Output must be .parquet or .csv')
    print(f'[bold cyan]Saved manifest[/bold cyan]: {out}')
    print(df.head())


if __name__ == '__main__':
    main()
