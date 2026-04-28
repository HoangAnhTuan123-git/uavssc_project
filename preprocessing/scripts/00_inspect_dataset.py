from __future__ import annotations

import argparse
from pathlib import Path

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



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=str, required=True)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    scene_dirs = discover_scene_dirs(data_root)
    print(f'[bold green]Found {len(scene_dirs)} candidate scene directories under[/bold green] {data_root}')

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

    for scene_root in sorted(scene_dirs):
        print(f'\n[bold cyan]Scene:[/bold cyan] {scene_root.name}')
        cams = discover_camera_files(scene_root)
        lidars = discover_lidar_files(scene_root)
        label_ids = discover_label_files(scene_root, mode='id')
        label_rgbs = discover_label_files(scene_root, mode='rgb')
        sampleinfos = find_first_existing(scene_root, ['sampleinfos_interpolated.json'])

        print(f'  camera files:      {len(cams)}')
        print(f'  lidar files:       {len(lidars)}')
        print(f'  label id files:    {len(label_ids)}')
        print(f'  label rgb files:   {len(label_rgbs)}')
        print(f'  sampleinfos json:  {sampleinfos}')

        if sampleinfos is not None:
            data = read_sampleinfos_json(sampleinfos)
            print(f'  sampleinfos records: {len(data)}')
            if len(data) > 0:
                print(f'  first record keys: {sorted(list(data[0].keys()))}')
                print(f'  first record sample: {data[0]}')


if __name__ == '__main__':
    main()
