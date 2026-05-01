from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rich import print
from tqdm import tqdm

from uavssc.constants import RAW_CMAP
from uavssc.io import read_cam_label_id_image, read_cam_label_rgb_image, read_image
from uavssc.utils import timestamp_from_stem


def _path_or_none(value) -> Path | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
    except Exception:
        pass
    s = str(value)
    if not s or s.lower() in {'nan', 'none', '<na>'}:
        return None
    return Path(s)


def _exact_rgb_agreement(id_mask: np.ndarray, rgb_mask: np.ndarray, max_pixels: int = 100_000) -> float | None:
    if id_mask.shape[:2] != rgb_mask.shape[:2]:
        return None
    h, w = id_mask.shape[:2]
    flat_id = id_mask.reshape(-1)
    flat_rgb = rgb_mask.reshape(-1, 3)
    valid = np.isin(flat_id, np.asarray(list(RAW_CMAP.keys()), dtype=flat_id.dtype))
    if valid.sum() == 0:
        return None
    idx = np.where(valid)[0]
    if len(idx) > max_pixels:
        step = max(1, len(idx) // max_pixels)
        idx = idx[::step]
    expected = np.asarray([RAW_CMAP[int(x)]['RGB'] for x in flat_id[idx]], dtype=np.uint8)
    ok = np.all(flat_rgb[idx].astype(np.uint8) == expected, axis=1)
    return float(ok.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description='Validate paired UAVScenes CAM_label id/color PNG masks against RGB images.')
    ap.add_argument('--manifest', type=str, required=True)
    ap.add_argument('--max-samples', type=int, default=0, help='0 means all samples')
    ap.add_argument('--output', type=str, default='artifacts/cam_label_report.csv')
    args = ap.parse_args()

    df = pd.read_parquet(args.manifest) if args.manifest.endswith('.parquet') else pd.read_csv(args.manifest)
    need_cols = {'img_path', 'cam_label_id_path'}
    missing = need_cols.difference(df.columns)
    if missing:
        raise ValueError(f'Manifest is missing required CAM-label columns: {sorted(missing)}')

    df = df[df['img_path'].notna() & (df['cam_label_id_path'].notna() | df.get('cam_label_rgb_path', pd.Series(index=df.index)).notna())].copy()
    if args.max_samples > 0:
        df = df.iloc[:args.max_samples].copy()

    rows = []
    n_ok = 0
    n_bad = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc='validate CAM labels'):
        img_path = _path_or_none(row.get('img_path'))
        id_path = _path_or_none(row.get('cam_label_id_path'))
        rgb_path = _path_or_none(row.get('cam_label_rgb_path'))
        status = 'ok'
        errors: list[str] = []
        item = {
            'scene': row.get('scene'),
            'timestamp': row.get('timestamp'),
            'img_path': str(img_path) if img_path else '',
            'cam_label_id_path': str(id_path) if id_path else '',
            'cam_label_rgb_path': str(rgb_path) if rgb_path else '',
        }
        try:
            if img_path is None or not img_path.exists():
                errors.append('missing image')
                img_shape = None
            else:
                img = read_image(img_path)
                img_shape = img.shape[:2]
                item['img_h'] = int(img_shape[0])
                item['img_w'] = int(img_shape[1])

            id_mask = None
            if id_path is None or not id_path.exists():
                errors.append('missing CAM label-id')
            else:
                id_mask = read_cam_label_id_image(id_path)
                item['id_h'] = int(id_mask.shape[0])
                item['id_w'] = int(id_mask.shape[1])
                vals = np.unique(id_mask)
                item['id_unique_count'] = int(len(vals))
                item['id_unique_preview'] = ' '.join(map(str, vals[:40].tolist()))
                if img_shape is not None and tuple(id_mask.shape[:2]) != tuple(img_shape):
                    errors.append(f'id shape mismatch image {id_mask.shape[:2]} vs {img_shape}')

            rgb_mask = None
            if rgb_path is not None and rgb_path.exists():
                rgb_mask = read_cam_label_rgb_image(rgb_path)
                item['rgb_h'] = int(rgb_mask.shape[0])
                item['rgb_w'] = int(rgb_mask.shape[1])
                if img_shape is not None and tuple(rgb_mask.shape[:2]) != tuple(img_shape):
                    errors.append(f'rgb shape mismatch image {rgb_mask.shape[:2]} vs {img_shape}')
                if id_mask is not None:
                    agreement = _exact_rgb_agreement(id_mask, rgb_mask)
                    item['id_rgb_exact_agreement'] = np.nan if agreement is None else agreement
            elif rgb_path is not None:
                errors.append('missing CAM label-color')

            if img_path is not None and id_path is not None:
                img_ts = timestamp_from_stem(img_path)
                id_ts = timestamp_from_stem(id_path)
                item['image_label_dt_s'] = np.nan if img_ts is None or id_ts is None else abs(float(img_ts) - float(id_ts))

        except Exception as exc:
            errors.append(repr(exc))

        if errors:
            status = 'bad'
            n_bad += 1
        else:
            n_ok += 1
        item['status'] = status
        item['errors'] = '; '.join(errors)
        rows.append(item)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'[green]OK CAM-label samples[/green]: {n_ok}')
    print(f'[red]Bad CAM-label samples[/red]: {n_bad}')
    print(f'[bold cyan]Wrote CAM-label report[/bold cyan]: {out}')
    if n_bad:
        print(pd.DataFrame(rows).query("status == 'bad'").head(20))


if __name__ == '__main__':
    main()
