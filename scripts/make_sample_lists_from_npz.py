#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List


def collect_scene_npz(preprocess_root: Path, scene_runs: List[str]) -> List[str]:
    out: List[str] = []
    for scene_run in scene_runs:
        scene_dir = preprocess_root / scene_run
        if not scene_dir.exists():
            print(f"[WARN] missing NPZ scene directory: {scene_dir}")
            continue
        for p in sorted(scene_dir.glob("*.npz")):
            out.append(str(Path(scene_run) / p.name))
    return out


def read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")]


def main() -> None:
    ap = argparse.ArgumentParser(description="Populate train/val/test sample lists from exported NPZ roots and split run files.")
    ap.add_argument("--preprocess-root", type=Path, required=True, help="Root that contains scene_run folders of exported NPZ files.")
    ap.add_argument("--split-root", type=Path, required=True, help="Fold directory, e.g. data/splits/scene_strict_cv/fold_A")
    args = ap.parse_args()

    for subset in ["train", "val", "test"]:
        runs = read_lines(args.split_root / f"{subset}_runs.txt")
        samples = collect_scene_npz(args.preprocess_root, runs)
        (args.split_root / f"{subset}_samples.txt").write_text("\n".join(samples) + ("\n" if samples else ""), encoding="utf-8")
        print(f"{subset}: wrote {len(samples)} samples")


if __name__ == "__main__":
    main()
