from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rich import print

from uavssc.io import (
    discover_camera_files,
    discover_scene_dirs,
    discover_label_files,
    discover_lidar_files,
    find_first_existing,
    parse_calibration_results_py,
    read_ply_header_counts,
    read_sampleinfos_json,
)
from uavssc.manifest import build_manifest_for_scene, manifest_to_dataframe
from uavssc.utils import load_yaml


def _count_notna(df, col: str) -> int:
    return int(df[col].notna().sum()) if col in df.columns else 0


def main() -> None:
    ap = argparse.ArgumentParser(description='Inspect UAVScenes folder layout and interval1/interval5 label availability.')
    ap.add_argument('--data-root', type=str, required=True)
    ap.add_argument('--config', type=str, default='configs/default.yaml')
    ap.add_argument('--max-scenes', type=int, default=0, help='0 means all scenes')
    args = ap.parse_args()

    cfg = load_yaml(args.config) if Path(args.config).exists() else {}
    dcfg = cfg.get('dataset', {}) if isinstance(cfg, dict) else {}
    interval = dcfg.get('use_interval', None)

    data_root = Path(args.data_root)
    scene_dirs = discover_scene_dirs(data_root)
    if args.max_scenes > 0:
        scene_dirs = scene_dirs[:args.max_scenes]

    print(f'[bold green]Found {len(scene_dirs)} candidate scene directories under[/bold green] {data_root}')
    print(f'[bold green]Configured interval:[/bold green] {interval}')

    global_calib = find_first_existing(data_root, ['calibration_results.py'])
    global_cloud = find_first_existing(data_root, ['cloud_merged.ply'])
    global_mesh = find_first_existing(data_root, ['Mesh.ply'])
    print('\n[bold magenta]Global/shared files[/bold magenta]')
    print(f'  calibration py: {global_calib}')
    print(f'  terra cloud:    {global_cloud}')
    print(f'  terra mesh:     {global_mesh}')
    if global_cloud is not None:
        print(f'  cloud header counts: {read_ply_header_counts(global_cloud)}')
    if global_mesh is not None:
        print(f'  mesh header counts:  {read_ply_header_counts(global_mesh)}')
    if global_calib is not None:
        mod = parse_calibration_results_py(global_calib)
        top_keys = [k for k in mod.keys() if not callable(mod[k])]
        print(f'  calibration top-level keys: {top_keys}')

    all_manifest_rows = []
    for scene_root in sorted(scene_dirs):
        print(f'\n[bold cyan]Scene:[/bold cyan] {scene_root.name}')
        cams = discover_camera_files(scene_root, interval=interval, camera_folder_hint=dcfg.get('camera_folder_hint', 'CAM'))
        lidars = discover_lidar_files(scene_root, interval=interval, lidar_folder_hint=dcfg.get('lidar_folder_hint', 'LiDAR'))
        label_ids = discover_label_files(
            scene_root,
            mode='id',
            interval=interval,
            label_id_hint=dcfg.get('lidar_label_id_hint', 'label_id'),
            label_rgb_hint=dcfg.get('lidar_label_rgb_hint', 'label_color'),
            lidar_folder_hint=dcfg.get('lidar_folder_hint', 'LiDAR'),
        )
        label_rgbs = discover_label_files(
            scene_root,
            mode='rgb',
            interval=interval,
            label_id_hint=dcfg.get('lidar_label_id_hint', 'label_id'),
            label_rgb_hint=dcfg.get('lidar_label_rgb_hint', 'label_color'),
            lidar_folder_hint=dcfg.get('lidar_folder_hint', 'LiDAR'),
        )
        sampleinfos = find_first_existing(scene_root, [dcfg.get('sampleinfos_json_name', 'sampleinfos_interpolated.json')])

        records = build_manifest_for_scene(scene_root, cfg=cfg)
        mdf = manifest_to_dataframe(records)
        all_manifest_rows.append(mdf)

        print(f'  raw camera files discovered inside scene: {len(cams)}')
        print(f'  raw lidar files discovered inside scene:  {len(lidars)}')
        print(f'  raw lidar label-id inside scene:          {len(label_ids)}')
        print(f'  raw lidar label-rgb inside scene:         {len(label_rgbs)}')
        print(f'  sampleinfos json:                         {sampleinfos}')
        print(f'  manifest records:                         {len(mdf)}')
        print(f'  manifest with img/lidar:                  {_count_notna(mdf, "img_path")}/{_count_notna(mdf, "lidar_path")}')
        print(f'  manifest with LiDAR label id/rgb:         {_count_notna(mdf, "lidar_label_id_path")}/{_count_notna(mdf, "lidar_label_rgb_path")}')
        print(f'  manifest with CAM label id/rgb:           {_count_notna(mdf, "cam_label_id_path")}/{_count_notna(mdf, "cam_label_rgb_path")}')
        print(f'  manifest with pose/K:                     {_count_notna(mdf, "T_world_cam")}/{_count_notna(mdf, "K")}')

        if sampleinfos is not None:
            data = read_sampleinfos_json(sampleinfos)
            print(f'  sampleinfos records:                      {len(data)}')
            if len(data) > 0:
                print(f'  first record keys:                        {sorted(list(data[0].keys()))}')

    if all_manifest_rows:
        df = pd.concat(all_manifest_rows, ignore_index=True)
        print('\n[bold magenta]Manifest-level summary[/bold magenta]')
        print(f'  total records:               {len(df)}')
        print(f'  CAM label-id paired:         {_count_notna(df, "cam_label_id_path")}')
        print(f'  CAM label-color paired:      {_count_notna(df, "cam_label_rgb_path")}')
        print(f'  LiDAR label-id paired:       {_count_notna(df, "lidar_label_id_path")}')
        print(f'  LiDAR label-color paired:    {_count_notna(df, "lidar_label_rgb_path")}')


if __name__ == '__main__':
    main()
