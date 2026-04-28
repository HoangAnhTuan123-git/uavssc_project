
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from rich import print
from tqdm import tqdm


def overlay_one(npz_path: Path, out_path: Path, max_points: int = 30000) -> bool:
    data = np.load(npz_path, allow_pickle=False)
    if 'img_path' not in data or 'projected_pix_1' not in data or 'fov_mask_1' not in data or 'target' not in data:
        return False
    img_path = str(data['img_path'][0])
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return False

    target = data['target'].reshape(-1)
    uv = data['projected_pix_1']
    fov = data['fov_mask_1'].reshape(-1)
    pix_z = data['pix_z_1'].reshape(-1) if 'pix_z_1' in data else np.ones_like(fov, dtype=np.float32)

    good = (target != 255) & (target != 0) & fov & (pix_z > 0)
    idx = np.where(good)[0]
    if len(idx) == 0:
        return False
    if len(idx) > max_points:
        step = max(1, len(idx) // max_points)
        idx = idx[::step]

    for flat_idx in idx:
        u, v = uv[flat_idx]
        cls = int(target[flat_idx])
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            color = (
                int((37 * cls) % 255),
                int((97 * cls) % 255),
                int((173 * cls) % 255),
            )
            cv2.circle(img, (int(u), int(v)), 1, color, -1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-root', type=str, required=True)
    ap.add_argument('--output-dir', type=str, required=True)
    ap.add_argument('--max-files', type=int, default=200)
    args = ap.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    files = sorted(input_root.rglob('*.npz'))[:args.max_files]
    saved = 0
    for path in tqdm(files, desc='overlay'):
        rel = path.relative_to(input_root)
        out_path = output_dir / rel.with_suffix('.png')
        ok = overlay_one(path, out_path)
        saved += int(ok)
    print(f'[green]Saved[/green] {saved} overlays to {output_dir}')


if __name__ == '__main__':
    main()
