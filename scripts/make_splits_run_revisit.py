#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List
import yaml


def load_registry(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description="Create same-scene unseen-run split templates.")
    ap.add_argument("--registry", type=Path, default=Path("data/index/scene_registry.csv"))
    ap.add_argument("--out-root", type=Path, default=Path("data/splits/run_revisit"))
    args = ap.parse_args()

    rows = load_registry(args.registry)
    grouped: Dict[str, List[str]] = {}
    for r in rows:
        grouped.setdefault(r["physical_scene"], [])
        if r["scene_run"] not in grouped[r["physical_scene"]]:
            grouped[r["physical_scene"]].append(r["scene_run"])

    for physical_scene, runs in sorted(grouped.items()):
        runs = sorted(runs)
        out_dir = args.out_root / physical_scene
        out_dir.mkdir(parents=True, exist_ok=True)

        if len(runs) >= 3:
            spec = {"train": runs[:-2], "val": [runs[-2]], "test": [runs[-1]]}
        elif len(runs) == 2:
            spec = {"train": [runs[0]], "val": [], "test": [runs[1]]}
        elif len(runs) == 1:
            spec = {"train": [runs[0]], "val": [], "test": []}
        else:
            spec = {"train": [], "val": [], "test": []}

        (out_dir / "split.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        for subset in ["train", "val", "test"]:
            (out_dir / f"{subset}_runs.txt").write_text("\n".join(spec[subset]) + ("\n" if spec[subset] else ""), encoding="utf-8")
            (out_dir / f"{subset}_samples.txt").write_text("# generated later\n", encoding="utf-8")

        print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
