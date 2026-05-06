from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from rich import print
from tqdm import tqdm


def _npz_string(arr: np.ndarray) -> str:
    try:
        return str(arr[0])
    except Exception:
        return str(arr)


def _read_optional_path(data, key: str) -> str | None:
    if key not in data:
        return None
    s = _npz_string(data[key])
    if not s or s.lower() in {'nan', 'none', '<na>'}:
        return None
    return s




def _get_hw(data, key: str) -> tuple[int, int] | None:
    if key not in data:
        return None
    arr = np.asarray(data[key]).reshape(-1)
    if arr.size < 2:
        return None
    return int(arr[0]), int(arr[1])


def _resize_canvas_to_npz_projection(canvas: np.ndarray, data) -> np.ndarray:
    """Resize RGB/CAM-label panels to the image size used by projected_pix_*.

    Old NPZ files may store full-resolution projections. New resized NPZ files
    store projection_image_shape_hw/image_shape_hw. The overlay must draw on the
    same 2D lattice used by projected_pix_*; otherwise the visualization looks
    shifted/scaled even when geometry is correct.
    """
    hw = _get_hw(data, 'projection_image_shape_hw') or _get_hw(data, 'image_shape_hw')
    if hw is None:
        return canvas
    h, w = hw
    if h > 0 and w > 0 and canvas.shape[:2] != (h, w):
        return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
    return canvas

def _draw_projected_voxels(canvas: np.ndarray, target: np.ndarray, uv: np.ndarray, fov: np.ndarray, pix_z: np.ndarray, max_points: int) -> int:
    good = (target != 255) & (target != 0) & fov & (pix_z > 0)
    idx = np.where(good)[0]
    if len(idx) == 0:
        return 0
    if len(idx) > max_points:
        step = max(1, len(idx) // max_points)
        idx = idx[::step]

    h, w = canvas.shape[:2]
    for flat_idx in idx:
        u, v = uv[flat_idx]
        cls = int(target[flat_idx])
        if 0 <= u < w and 0 <= v < h:
            color = (
                int((37 * cls) % 255),
                int((97 * cls) % 255),
                int((173 * cls) % 255),
            )
            cv2.circle(canvas, (int(u), int(v)), 1, color, -1)
    return len(idx)


def overlay_one(npz_path: Path, out_path: Path, max_points: int = 30000, side_by_side_cam_label: bool = True) -> bool:
    data = np.load(npz_path, allow_pickle=False)
    if 'img_path' not in data or 'projected_pix_1' not in data or 'fov_mask_1' not in data or 'target' not in data:
        return False
    img_path = _npz_string(data['img_path'])
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return False
    img = _resize_canvas_to_npz_projection(img, data)

    target = data['target'].reshape(-1)
    uv = data['projected_pix_1']
    fov = data['fov_mask_1'].reshape(-1)
    pix_z = data['pix_z_1'].reshape(-1) if 'pix_z_1' in data else np.ones_like(fov, dtype=np.float32)

    overlay_img = img.copy()
    n_drawn = _draw_projected_voxels(overlay_img, target, uv, fov, pix_z, max_points=max_points)
    if n_drawn == 0:
        return False

    cv2.putText(overlay_img, 'RGB + projected occupied SSC voxels', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    final = overlay_img
    if side_by_side_cam_label:
        cam_label_path = _read_optional_path(data, 'cam_label_rgb_path') or _read_optional_path(data, 'cam_label_id_path')
        if cam_label_path:
            label = cv2.imread(cam_label_path, cv2.IMREAD_COLOR)
            if label is not None:
                label = _resize_canvas_to_npz_projection(label, data)
                if label.shape[:2] != overlay_img.shape[:2]:
                    label = cv2.resize(label, (overlay_img.shape[1], overlay_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                label_overlay = label.copy()
                _draw_projected_voxels(label_overlay, target, uv, fov, pix_z, max_points=max_points)
                cv2.putText(label_overlay, 'CAM label + projected occupied SSC voxels', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                final = np.concatenate([overlay_img, label_overlay], axis=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), final)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description='Overlay exported SSC voxel projections on matching RGB and optional CAM_label masks.')
    ap.add_argument('--input-root', type=str, required=True)
    ap.add_argument('--output-dir', type=str, required=True)
    ap.add_argument('--max-files', type=int, default=200)
    ap.add_argument('--max-points', type=int, default=30000)
    ap.add_argument('--no-cam-label-panel', action='store_true')
    args = ap.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    files = sorted(input_root.rglob('*.npz'))[:args.max_files]
    saved = 0
    for path in tqdm(files, desc='overlay'):
        rel = path.relative_to(input_root)
        out_path = output_dir / rel.with_suffix('.png')
        ok = overlay_one(path, out_path, max_points=args.max_points, side_by_side_cam_label=not args.no_cam_label_panel)
        saved += int(ok)
    print(f'[green]Saved[/green] {saved} overlays to {output_dir}')


if __name__ == '__main__':
    main()
