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
    ap = argparse.ArgumentParser(description="Create cross-scene split folders from scene_registry.csv.")
    ap.add_argument("--registry", type=Path, default=Path("data/index/scene_registry.csv"))
    ap.add_argument("--out-root", type=Path, default=Path("data/splits/scene_strict_cv"))
    ap.add_argument("--scene-order", nargs="*", default=["AMtown", "AMvalley", "HKairport", "HKisland"])
    args = ap.parse_args()

    rows = load_registry(args.registry)
    all_physical = sorted({r["physical_scene"] for r in rows})
    if len(all_physical) == 0:
        raise RuntimeError("No physical scenes found in registry.")
    for scene in args.scene_order:
        if scene not in all_physical:
            print(f"[WARN] scene_order contains {scene} but it was not found in registry.")

    folds: Dict[str, Dict[str, List[str]]] = {
        "fold_A": {"train": ["AMtown", "AMvalley"], "val": ["HKairport"], "test": ["HKisland"]},
        "fold_B": {"train": ["AMvalley", "HKairport"], "val": ["HKisland"], "test": ["AMtown"]},
        "fold_C": {"train": ["HKairport", "HKisland"], "val": ["AMtown"], "test": ["AMvalley"]},
        "fold_D": {"train": ["HKisland", "AMtown"], "val": ["AMvalley"], "test": ["HKairport"]},
    }

    scene_to_runs: Dict[str, List[str]] = {}
    for r in rows:
        scene_to_runs.setdefault(r["physical_scene"], [])
        if r["scene_run"] not in scene_to_runs[r["physical_scene"]]:
            scene_to_runs[r["physical_scene"]].append(r["scene_run"])

    for fold, spec in folds.items():
        fold_dir = args.out_root / fold
        fold_dir.mkdir(parents=True, exist_ok=True)

        (fold_dir / "split.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        for subset in ["train", "val", "test"]:
            scenes = spec[subset]
            runs: List[str] = []
            for s in scenes:
                runs.extend(sorted(scene_to_runs.get(s, [])))

            (fold_dir / f"{subset}_scenes.txt").write_text("\n".join(scenes) + "\n", encoding="utf-8")
            (fold_dir / f"{subset}_runs.txt").write_text("\n".join(runs) + "\n", encoding="utf-8")
            # sample lists are generated later, once NPZ export roots are known
            (fold_dir / f"{subset}_samples.txt").write_text("# generated later\n", encoding="utf-8")

        print(f"Wrote {fold_dir}")


if __name__ == "__main__":
    main()
