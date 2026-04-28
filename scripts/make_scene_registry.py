#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_scene_run(name: str) -> Tuple[str, str]:
    m = re.match(r"^(.*?)(\d+)$", name)
    if m:
        return m.group(1), m.group(2)
    return name, ""


def find_scene_dirs(interval_cam_lidar_root: Path) -> List[Path]:
    if not interval_cam_lidar_root.exists():
        return []
    return sorted([p for p in interval_cam_lidar_root.iterdir() if p.is_dir()])


def find_child_folder(scene_dir: Path, keywords: List[str]) -> Optional[Path]:
    candidates = []
    for p in scene_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name.lower()
        if all(k.lower() in name for k in keywords):
            candidates.append(p)
    if candidates:
        return sorted(candidates)[0]
    return None


def find_label_scene_root(raw_root: Path, interval: str, kind: str, scene_name: str) -> Optional[Path]:
    roots = [
        raw_root / f"{interval}_{kind}",
        raw_root / f"{interval}_{kind.upper()}",
        raw_root / kind,
    ]
    for root in roots:
        if root.exists():
            candidate = root / scene_name
            if candidate.exists():
                return candidate
    return None


def find_terra_root(raw_root: Path, physical_scene: str) -> Optional[Path]:
    terra_root = raw_root / "terra_3dmap_pointcloud_mesh"
    if not terra_root.exists():
        return None
    for p in terra_root.iterdir():
        if p.is_dir() and p.name.lower().startswith(physical_scene.lower()):
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a scene-level registry for UAVScenes.")
    ap.add_argument("--raw-root", type=Path, default=Path("data/raw/uavscenes_official"))
    ap.add_argument("--out-csv", type=Path, default=Path("data/index/scene_registry.csv"))
    ap.add_argument("--intervals", nargs="*", default=["interval5", "interval1"])
    args = ap.parse_args()

    rows: List[Dict[str, str]] = []
    for interval in args.intervals:
        cam_lidar_root = args.raw_root / f"{interval}_CAM_LIDAR"
        for scene_dir in find_scene_dirs(cam_lidar_root):
            scene_run = scene_dir.name
            physical_scene, run_id = parse_scene_run(scene_run)

            cam_root = find_child_folder(scene_dir, ["cam"])
            lidar_root = find_child_folder(scene_dir, ["lidar"])

            cam_label_root = find_label_scene_root(args.raw_root, interval, "CAM_label", scene_run)
            lidar_label_root = find_label_scene_root(args.raw_root, interval, "LIDAR_label", scene_run)
            terra_root = find_terra_root(args.raw_root, physical_scene)

            rows.append(
                {
                    "scene_run": scene_run,
                    "physical_scene": physical_scene,
                    "run_id": run_id,
                    "interval": interval,
                    "scene_root": str(scene_dir),
                    "cam_root": str(cam_root) if cam_root else "",
                    "lidar_root": str(lidar_root) if lidar_root else "",
                    "cam_label_root": str(cam_label_root) if cam_label_root else "",
                    "lidar_label_root": str(lidar_label_root) if lidar_label_root else "",
                    "terra_root": str(terra_root) if terra_root else "",
                }
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene_run",
                "physical_scene",
                "run_id",
                "interval",
                "scene_root",
                "cam_root",
                "lidar_root",
                "cam_label_root",
                "lidar_label_root",
                "terra_root",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} scene rows to {args.out_csv}")


if __name__ == "__main__":
    main()
