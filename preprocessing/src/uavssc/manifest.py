from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .io import (
    SampleRecord,
    discover_cam_label_files,
    discover_camera_files,
    discover_label_files,
    discover_lidar_files,
    find_first_existing,
    infer_intrinsics_from_record,
    infer_pose_matrix_from_record,
    infer_timestamp_from_record,
    label_file_dual_timestamp_info,
    parse_dual_timestamp_label_stem,
    read_sampleinfos_json,
)
from .utils import pair_by_nearest_timestamp, timestamp_from_stem


def serialize_matrix(T: np.ndarray | None) -> str | None:
    if T is None:
        return None
    return np.asarray(T, dtype=np.float64).round(9).tolist().__repr__()


def manifest_to_dataframe(records: list[SampleRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append({
            'scene': r.scene,
            'run': r.run,
            'timestamp': r.timestamp,
            'img_path': r.img_path,
            'lidar_path': r.lidar_path,
            'lidar_label_id_path': r.lidar_label_id_path,
            'lidar_label_rgb_path': r.lidar_label_rgb_path,
            'cam_label_id_path': r.cam_label_id_path,
            'cam_label_rgb_path': r.cam_label_rgb_path,
            'T_world_cam': serialize_matrix(r.T_world_cam),
            'T_world_lidar': serialize_matrix(r.T_world_lidar),
            'K': serialize_matrix(r.K),
            'dist': serialize_matrix(r.dist),
        })
    return pd.DataFrame(rows)


def _scene_tokens(scene_name: str) -> list[str]:
    """Scene aliases used to filter label folders.

    Examples:
    interval1_AMtown01 -> [interval1_amtown01, amtown01]
    AMtown01 -> [amtown01]
    """
    s = scene_name.lower()
    out = [s]
    for prefix in ('interval1_', 'interval5_'):
        if s.startswith(prefix):
            out.append(s[len(prefix):])
    # Also allow the physical scene/run token if nested paths include only AMtown01.
    if '_' in s:
        out.append(s.split('_')[-1])
    return sorted(set(x for x in out if x))


def _filter_paths_for_scene(paths: list[Path], scene_name: str) -> list[Path]:
    tokens = _scene_tokens(scene_name)
    filtered = [p for p in paths if any(tok in p.as_posix().lower() for tok in tokens)]
    return filtered if filtered else paths


def _candidate_label_search_roots(scene_root: Path, cfg: dict | None = None, key: str | None = None) -> list[Path]:
    cfg = cfg or {}
    dcfg = cfg.get('dataset', {}) if isinstance(cfg, dict) else {}
    roots: list[Path] = []

    if key and dcfg.get(key):
        raw = Path(str(dcfg[key]))
        if raw.is_absolute():
            roots.append(raw)
        else:
            # resolve relative hints against likely project/data locations
            roots.extend([
                scene_root / raw,
                scene_root.parent / raw,
                scene_root.parent.parent / raw,
            ])

    roots.extend([scene_root, scene_root.parent, scene_root.parent.parent])
    uniq: list[Path] = []
    seen: set[Path] = set()
    for r in roots:
        try:
            rp = r.resolve()
        except Exception:
            continue
        if rp in seen or not r.exists() or not r.is_dir():
            continue
        seen.add(rp)
        uniq.append(r)
    return uniq


def _discover_across_roots(
    roots: list[Path],
    discover: Callable[..., list[Path]],
    *args: Any,
    **kwargs: Any,
) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for r in roots:
        try:
            for p in discover(r, *args, **kwargs):
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    out.append(p)
        except Exception:
            continue
    return sorted(out)


def _match_single_timestamp_paths(anchor_ts: list[float], paths: list[Path], max_delta: float) -> list[Path | None]:
    valid = [(p, timestamp_from_stem(p)) for p in paths]
    valid = [(p, ts) for p, ts in valid if ts is not None]
    if not valid:
        return [None] * len(anchor_ts)
    ts = [float(x[1]) for x in valid]
    nearest = pair_by_nearest_timestamp(anchor_ts, ts, max_abs_delta=max_delta)
    return [None if m is None else valid[m][0] for m in nearest]


def _match_dual_timestamp_labels(
    rec_ts: list[float],
    lidar_ts_for_rec: list[float | None],
    label_info: list[dict[str, Any]],
    max_img_delta: float = 0.25,
    max_lidar_delta: float = 0.25,
) -> list[Path | None]:
    """Match labels to image records.

    UAVScenes LiDAR labels may be named either as a single timestamp or as
    image{camts}_lidar{lidarts}.  The label_info helper normalizes both forms.
    """
    out: list[Path | None] = []
    usable = [x for x in label_info if x.get('img_ts') is not None]
    if not usable:
        return [None] * len(rec_ts)

    for ts_img, ts_lidar in zip(rec_ts, lidar_ts_for_rec):
        best = None
        best_score = float('inf')
        for x in usable:
            d_img = abs(float(x['img_ts']) - ts_img)
            if d_img > max_img_delta:
                continue
            if ts_lidar is not None and x.get('lidar_ts') is not None:
                d_lidar = abs(float(x['lidar_ts']) - float(ts_lidar))
                if d_lidar > max_lidar_delta:
                    continue
            else:
                d_lidar = 0.0
            score = d_img + d_lidar
            if score < best_score:
                best = x['path']
                best_score = score
        out.append(best)
    return out


def build_manifest_for_scene(scene_root: str | Path, cfg: dict | None = None) -> list[SampleRecord]:
    """Build a per-image manifest for one UAVScenes run.

    This version supports both the older per-scene layout and the official/new
    layout where interval1_CAM_label / interval5_CAM_label and LIDAR_label live
    as sibling folders outside the per-run CAM/LiDAR folder.
    """
    scene_root = Path(scene_root)
    scene_name = scene_root.name
    cfg = cfg or {}
    dcfg = cfg.get('dataset', {}) if isinstance(cfg, dict) else {}

    interval = dcfg.get('use_interval', None)
    cam_hint = str(dcfg.get('camera_folder_hint', 'CAM'))
    lidar_hint = str(dcfg.get('lidar_folder_hint', 'LiDAR'))
    lidar_label_id_hint = str(dcfg.get('lidar_label_id_hint', 'label_id'))
    lidar_label_rgb_hint = str(dcfg.get('lidar_label_rgb_hint', 'label_color'))
    cam_label_folder_hint = str(dcfg.get('cam_label_folder_hint', 'CAM_label'))
    cam_label_id_hint = str(dcfg.get('cam_label_id_hint', 'label_id'))
    cam_label_rgb_hint = str(dcfg.get('cam_label_rgb_hint', 'label_color'))
    sampleinfos_name = str(dcfg.get('sampleinfos_json_name', 'sampleinfos_interpolated.json'))
    max_pair_delta_s = float(dcfg.get('max_pair_delta_s', 0.25))
    max_cam_label_delta_s = float(dcfg.get('max_cam_label_delta_s', 0.05))

    sampleinfos = find_first_existing(scene_root, [sampleinfos_name])

    cam_files = discover_camera_files(scene_root, interval=interval, camera_folder_hint=cam_hint)
    lidar_files = discover_lidar_files(scene_root, interval=interval, lidar_folder_hint=lidar_hint)

    label_roots = _candidate_label_search_roots(scene_root, cfg, key='lidar_label_root')
    lidar_label_id_files = _discover_across_roots(
        label_roots,
        discover_label_files,
        mode='id',
        interval=interval,
        label_id_hint=lidar_label_id_hint,
        label_rgb_hint=lidar_label_rgb_hint,
        lidar_folder_hint=lidar_hint,
    )
    lidar_label_rgb_files = _discover_across_roots(
        label_roots,
        discover_label_files,
        mode='rgb',
        interval=interval,
        label_id_hint=lidar_label_id_hint,
        label_rgb_hint=lidar_label_rgb_hint,
        lidar_folder_hint=lidar_hint,
    )
    lidar_label_id_files = _filter_paths_for_scene(lidar_label_id_files, scene_name)
    lidar_label_rgb_files = _filter_paths_for_scene(lidar_label_rgb_files, scene_name)

    cam_label_roots = _candidate_label_search_roots(scene_root, cfg, key='cam_label_root')
    cam_label_id_files = _discover_across_roots(
        cam_label_roots,
        discover_cam_label_files,
        mode='id',
        interval=interval,
        cam_label_folder_hint=cam_label_folder_hint,
        cam_label_id_hint=cam_label_id_hint,
        cam_label_rgb_hint=cam_label_rgb_hint,
    )
    cam_label_rgb_files = _discover_across_roots(
        cam_label_roots,
        discover_cam_label_files,
        mode='rgb',
        interval=interval,
        cam_label_folder_hint=cam_label_folder_hint,
        cam_label_id_hint=cam_label_id_hint,
        cam_label_rgb_hint=cam_label_rgb_hint,
    )
    cam_label_id_files = _filter_paths_for_scene(cam_label_id_files, scene_name)
    cam_label_rgb_files = _filter_paths_for_scene(cam_label_rgb_files, scene_name)

    cam_valid = [(p, timestamp_from_stem(p)) for p in cam_files]
    cam_valid = [(p, ts) for p, ts in cam_valid if ts is not None]
    lidar_valid = [(p, timestamp_from_stem(p)) for p in lidar_files]
    lidar_valid = [(p, ts) for p, ts in lidar_valid if ts is not None]

    cam_ts = [float(ts) for _, ts in cam_valid]
    lidar_ts = [float(ts) for _, ts in lidar_valid]

    if cam_valid:
        anchor_img_paths: list[str | None] = [str(p) for p, _ in cam_valid]
        anchor_ts = cam_ts
    elif sampleinfos is not None:
        sample_records_tmp = read_sampleinfos_json(sampleinfos)
        rec_ts_tmp = [infer_timestamp_from_record(r) for r in sample_records_tmp]
        valid_idx = [i for i, ts in enumerate(rec_ts_tmp) if ts is not None]
        anchor_ts = [float(rec_ts_tmp[i]) for i in valid_idx]
        anchor_img_paths = [None] * len(anchor_ts)
    else:
        return []

    sample_records: list[dict[str, Any]] = []
    rec_match: list[int | None] = [None] * len(anchor_ts)
    if sampleinfos is not None:
        sample_records_raw = read_sampleinfos_json(sampleinfos)
        raw_rec_ts = [infer_timestamp_from_record(r) for r in sample_records_raw]
        valid_idx = [i for i, ts in enumerate(raw_rec_ts) if ts is not None]
        sample_records = [sample_records_raw[i] for i in valid_idx]
        rec_ts = [float(raw_rec_ts[i]) for i in valid_idx]
        rec_match = pair_by_nearest_timestamp(anchor_ts, rec_ts, max_abs_delta=max_pair_delta_s)

    lidar_match = pair_by_nearest_timestamp(anchor_ts, lidar_ts, max_abs_delta=max_pair_delta_s)
    lidar_ts_for_anchor = [None if m is None else lidar_ts[m] for m in lidar_match]

    lidar_label_id_info = label_file_dual_timestamp_info(lidar_label_id_files)
    lidar_label_rgb_info = label_file_dual_timestamp_info(lidar_label_rgb_files)
    lidar_id_paths = _match_dual_timestamp_labels(anchor_ts, lidar_ts_for_anchor, lidar_label_id_info, max_img_delta=max_pair_delta_s, max_lidar_delta=max_pair_delta_s)
    lidar_rgb_paths = _match_dual_timestamp_labels(anchor_ts, lidar_ts_for_anchor, lidar_label_rgb_info, max_img_delta=max_pair_delta_s, max_lidar_delta=max_pair_delta_s)

    cam_id_paths = _match_single_timestamp_paths(anchor_ts, cam_label_id_files, max_delta=max_cam_label_delta_s)
    cam_rgb_paths = _match_single_timestamp_paths(anchor_ts, cam_label_rgb_files, max_delta=max_cam_label_delta_s)

    records: list[SampleRecord] = []
    for i, ts in enumerate(anchor_ts):
        img_path = anchor_img_paths[i]
        lidar_idx = lidar_match[i]

        id_path = None if lidar_id_paths[i] is None else str(lidar_id_paths[i])
        rgb_path = None if lidar_rgb_paths[i] is None else str(lidar_rgb_paths[i])
        cam_id_path = None if cam_id_paths[i] is None else str(cam_id_paths[i])
        cam_rgb_path = None if cam_rgb_paths[i] is None else str(cam_rgb_paths[i])

        # Robust fallback: if a label filename encodes a LiDAR timestamp, prefer matching by that.
        if lidar_idx is None and id_path is not None and lidar_ts:
            _, lbl_lidar_ts = parse_dual_timestamp_label_stem(id_path)
            if lbl_lidar_ts is not None:
                cand = pair_by_nearest_timestamp([float(lbl_lidar_ts)], lidar_ts, max_abs_delta=max_pair_delta_s)[0]
                if cand is not None:
                    lidar_idx = cand
        if lidar_idx is None and rgb_path is not None and lidar_ts:
            _, lbl_lidar_ts = parse_dual_timestamp_label_stem(rgb_path)
            if lbl_lidar_ts is not None:
                cand = pair_by_nearest_timestamp([float(lbl_lidar_ts)], lidar_ts, max_abs_delta=max_pair_delta_s)[0]
                if cand is not None:
                    lidar_idx = cand

        lidar_path = None if lidar_idx is None else str(lidar_valid[lidar_idx][0])

        K = None
        dist = None
        T_world_cam = None
        if rec_match[i] is not None:
            rec = sample_records[rec_match[i]]
            K, dist = infer_intrinsics_from_record(rec)
            T_world_cam = infer_pose_matrix_from_record(rec)

        records.append(SampleRecord(
            scene=scene_name,
            run=scene_name,
            timestamp=float(ts),
            img_path=img_path,
            lidar_path=lidar_path,
            lidar_label_id_path=id_path,
            lidar_label_rgb_path=rgb_path,
            cam_label_id_path=cam_id_path,
            cam_label_rgb_path=cam_rgb_path,
            T_world_cam=T_world_cam,
            T_world_lidar=T_world_cam.copy() if T_world_cam is not None else None,
            K=K,
            dist=dist,
        ))

    return records
