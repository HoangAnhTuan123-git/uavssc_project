#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Colors for dense train IDs exported by this project.
# 0 = empty. 255 = unknown/ignore.
UAV_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (70, 70, 70),       # roof
    2: (128, 64, 128),     # dirt_motor_road
    3: (128, 64, 128),     # paved_motor_road
    4: (0, 0, 142),        # river
    5: (0, 0, 180),        # pool
    6: (150, 100, 100),    # bridge
    7: (153, 153, 153),    # container
    8: (81, 0, 81),        # airstrip
    9: (190, 153, 153),    # traffic_barrier
    10: (107, 142, 35),    # green_field
    11: (152, 251, 152),   # wild_field
    12: (220, 220, 0),     # solar_board
    13: (255, 128, 0),     # umbrella
    14: (102, 102, 156),   # transparent_roof
    15: (250, 170, 160),   # car_park
    16: (244, 35, 232),    # paved_walk
    255: (255, 255, 255),
}


def colorize(label: np.ndarray) -> np.ndarray:
    label = label.astype(np.int32)
    out = np.zeros(label.shape + (3,), dtype=np.uint8)
    for k, v in UAV_COLORS.items():
        out[label == int(k)] = v
    missing = ~np.isin(label, list(UAV_COLORS.keys()))
    out[missing] = (255, 0, 255)
    return out


def topdown(label_3d: np.ndarray) -> np.ndarray:
    """Project a 3D semantic grid to a 2D top-down semantic map.

    For each (x, y), select the top/highest occupied semantic voxel. Empty columns remain 0.
    Assumes label_3d shape [X, Y, Z].
    """
    label = label_3d.astype(np.int32)
    occ = (label > 0) & (label < 255)
    # Reverse Z so first hit is highest voxel.
    rev_occ = occ[:, :, ::-1]
    has = rev_occ.any(axis=2)
    rev_idx = rev_occ.argmax(axis=2)
    z_idx = label.shape[2] - 1 - rev_idx
    out = np.zeros(label.shape[:2], dtype=np.uint8)
    xs, ys = np.where(has)
    out[xs, ys] = label[xs, ys, z_idx[xs, ys]].astype(np.uint8)
    # Rotate for a more natural image-like top-down view.
    return np.flipud(out.T)


def occupancy_height(label_3d: np.ndarray) -> np.ndarray:
    label = label_3d.astype(np.int32)
    occ = (label > 0) & (label < 255)
    h = np.zeros(label.shape[:2], dtype=np.uint8)
    if occ.any():
        rev = occ[:, :, ::-1]
        has = rev.any(axis=2)
        z = label.shape[2] - 1 - rev.argmax(axis=2)
        h[has] = np.round(255.0 * z[has] / max(1, label.shape[2] - 1)).astype(np.uint8)
    return np.flipud(h.T)


def load_rgb(img_path: str, size: Tuple[int, int]) -> Image.Image:
    if img_path and Path(img_path).exists():
        img = Image.open(img_path).convert("RGB")
    else:
        img = Image.new("RGB", size, (30, 30, 30))
    img.thumbnail(size, Image.BILINEAR)
    canvas = Image.new("RGB", size, (0, 0, 0))
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def resize_to(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return img.resize(size, Image.NEAREST)


def add_title(img: Image.Image, title: str) -> Image.Image:
    bar_h = 28
    out = Image.new("RGB", (img.width, img.height + bar_h), (255, 255, 255))
    out.paste(img, (0, bar_h))
    d = ImageDraw.Draw(out)
    d.text((6, 6), title, fill=(0, 0, 0))
    return out


def make_panel(pred_npz: Path, out_dir: Path, panel_size: Tuple[int, int]) -> Path:
    arr = np.load(pred_npz, allow_pickle=True)
    pred = arr["pred"]
    target = arr["target"] if "target" in arr.files else None
    img_path = str(arr["img_path"].item()) if "img_path" in arr.files and arr["img_path"].shape == () else ""
    seq = str(arr["sequence"].item()) if "sequence" in arr.files and arr["sequence"].shape == () else pred_npz.parent.name
    frame = str(arr["frame_id"].item()) if "frame_id" in arr.files and arr["frame_id"].shape == () else pred_npz.stem

    rgb = add_title(load_rgb(img_path, panel_size), "RGB input")
    pred_td = Image.fromarray(colorize(topdown(pred)))
    pred_td = add_title(resize_to(pred_td, panel_size), "Prediction top-down")

    if target is not None:
        tgt_td = Image.fromarray(colorize(topdown(target)))
        tgt_td = add_title(resize_to(tgt_td, panel_size), "Target top-down")
    else:
        tgt_td = add_title(Image.new("RGB", panel_size, (30, 30, 30)), "Target unavailable")

    pred_h = Image.fromarray(occupancy_height(pred)).convert("RGB")
    pred_h = add_title(resize_to(pred_h, panel_size), "Prediction height")

    margin = 10
    header_h = 34
    w = panel_size[0] * 2 + margin
    h = (panel_size[1] + 28) * 2 + margin + header_h
    canvas = Image.new("RGB", (w, h), (245, 245, 245))
    d = ImageDraw.Draw(canvas)
    d.text((8, 8), f"{seq} / {frame}", fill=(0, 0, 0))
    canvas.paste(rgb, (0, header_h))
    canvas.paste(pred_td, (panel_size[0] + margin, header_h))
    canvas.paste(tgt_td, (0, header_h + panel_size[1] + 28 + margin))
    canvas.paste(pred_h, (panel_size[0] + margin, header_h + panel_size[1] + 28 + margin))

    rel_dir = out_dir / seq
    rel_dir.mkdir(parents=True, exist_ok=True)
    out_path = rel_dir / f"{frame}_panel.png"
    canvas.save(out_path)
    return out_path


def iter_npz(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*.npz"))
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        yield from sorted(d.glob("*.npz"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize UAVScenes MonoScene prediction NPZ files.")
    ap.add_argument("--prediction-root", type=Path, required=True, help="Root created by predict_uavscenes.py")
    ap.add_argument("--output-dir", type=Path, required=True, help="Where PNG panels will be written")
    ap.add_argument("--max-files", type=int, default=50, help="Maximum panels to render; <=0 means all")
    ap.add_argument("--panel-width", type=int, default=384)
    ap.add_argument("--panel-height", type=int, default=320)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for p in iter_npz(args.prediction_root):
        if args.max_files > 0 and count >= args.max_files:
            break
        out = make_panel(p, args.output_dir, (args.panel_width, args.panel_height))
        count += 1
        if count == 1 or count % 10 == 0:
            print(f"Rendered {count}: {out}")
    print(f"Done. Rendered {count} visualization panels to {args.output_dir}")


if __name__ == "__main__":
    main()
