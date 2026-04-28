from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import (
    SampleRecord,
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
            'T_world_cam': serialize_matrix(r.T_world_cam),
            'T_world_lidar': serialize_matrix(r.T_world_lidar),
            'K': serialize_matrix(r.K),
            'dist': serialize_matrix(r.dist),
        })
    return pd.DataFrame(rows)



def _match_dual_timestamp_labels(
    rec_ts: list[float],
    lidar_ts_for_rec: list[float | None],
    label_info: list[dict[str, Any]],
    max_img_delta: float = 0.2,
    max_lidar_delta: float = 0.2,
) -> list[Path | None]:
    """Prefer exact/near label filenames of form image{camts}_lidar{lidarts}."""
    out: list[Path | None] = []
    dual = [x for x in label_info if x['img_ts'] is not None]
    if not dual:
        return [None] * len(rec_ts)

    for ts_img, ts_lidar in zip(rec_ts, lidar_ts_for_rec):
        best = None
        best_score = float('inf')
        for x in dual:
            d_img = abs(float(x['img_ts']) - ts_img)
            if d_img > max_img_delta:
                continue
            if ts_lidar is not None and x['lidar_ts'] is not None:
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



def build_manifest_for_scene(scene_root: str | Path) -> list[SampleRecord]:
    """Build a per-image manifest for one scene.

    Why this version is needed for UAVScenes:
    - sampleinfos_interpolated.json may contain all interval-1 camera poses.
    - a local scene folder may only contain a subset of actual images (e.g. interval5).
    - therefore we anchor on *existing camera files* and attach nearest JSON pose records.
    """
    scene_root = Path(scene_root)
    scene_name = scene_root.name

    sampleinfos = find_first_existing(scene_root, ['sampleinfos_interpolated.json'])
    records: list[SampleRecord] = []

    cam_files = discover_camera_files(scene_root)
    lidar_files = discover_lidar_files(scene_root)
    label_id_files = discover_label_files(scene_root, mode='id')
    label_rgb_files = discover_label_files(scene_root, mode='rgb')

    # Keep only files whose stem is a numeric timestamp.
    cam_valid = [(p, timestamp_from_stem(p)) for p in cam_files]
    cam_valid = [(p, ts) for p, ts in cam_valid if ts is not None]
    lidar_valid = [(p, timestamp_from_stem(p)) for p in lidar_files]
    lidar_valid = [(p, ts) for p, ts in lidar_valid if ts is not None]
    label_id_info = label_file_dual_timestamp_info(label_id_files)
    label_rgb_info = label_file_dual_timestamp_info(label_rgb_files)

    cam_ts = [float(ts) for _, ts in cam_valid]
    lidar_ts = [float(ts) for _, ts in lidar_valid]

    # Primary anchors = actual existing images.
    if cam_valid:
        anchor_img_paths = [str(p) for p, _ in cam_valid]
        anchor_ts = cam_ts
    elif sampleinfos is not None:
        # Fallback if a scene has JSON but no images in the local folder.
        sample_records_tmp = read_sampleinfos_json(sampleinfos)
        rec_ts_tmp = [infer_timestamp_from_record(r) for r in sample_records_tmp]
        valid_idx = [i for i, ts in enumerate(rec_ts_tmp) if ts is not None]
        anchor_ts = [float(rec_ts_tmp[i]) for i in valid_idx]
        anchor_img_paths = [None] * len(anchor_ts)
    else:
        return []

    # Read JSON metadata and match each actual image to the nearest JSON pose row.
    sample_records: list[dict[str, Any]] = []
    rec_ts: list[float] = []
    rec_match: list[int | None] = [None] * len(anchor_ts)
    if sampleinfos is not None:
        sample_records = read_sampleinfos_json(sampleinfos)
        raw_rec_ts = [infer_timestamp_from_record(r) for r in sample_records]
        valid_idx = [i for i, ts in enumerate(raw_rec_ts) if ts is not None]
        sample_records = [sample_records[i] for i in valid_idx]
        rec_ts = [float(raw_rec_ts[i]) for i in valid_idx]
        rec_match = pair_by_nearest_timestamp(anchor_ts, rec_ts, max_abs_delta=0.25)

    lidar_match = pair_by_nearest_timestamp(anchor_ts, lidar_ts, max_abs_delta=0.25)
    lidar_ts_for_anchor = [None if m is None else lidar_ts[m] for m in lidar_match]

    # Prefer labels whose filename encodes both image timestamp and lidar timestamp.
    id_paths = _match_dual_timestamp_labels(anchor_ts, lidar_ts_for_anchor, label_id_info, max_img_delta=0.25, max_lidar_delta=0.25)
    rgb_paths = _match_dual_timestamp_labels(anchor_ts, lidar_ts_for_anchor, label_rgb_info, max_img_delta=0.25, max_lidar_delta=0.25)

    # Fallback to nearest-to-image timestamp if dual-timestamp matching fails.
    if any(p is None for p in id_paths):
        single_label_id = [(x['path'], x['img_ts']) for x in label_id_info if x['img_ts'] is not None]
        single_id_ts = [float(ts) for _, ts in single_label_id]
        nearest = pair_by_nearest_timestamp(anchor_ts, single_id_ts, max_abs_delta=0.25)
        for i, m in enumerate(nearest):
            if id_paths[i] is None and m is not None:
                id_paths[i] = single_label_id[m][0]

    if any(p is None for p in rgb_paths):
        single_label_rgb = [(x['path'], x['img_ts']) for x in label_rgb_info if x['img_ts'] is not None]
        single_rgb_ts = [float(ts) for _, ts in single_label_rgb]
        nearest = pair_by_nearest_timestamp(anchor_ts, single_rgb_ts, max_abs_delta=0.25)
        for i, m in enumerate(nearest):
            if rgb_paths[i] is None and m is not None:
                rgb_paths[i] = single_label_rgb[m][0]

    for i, ts in enumerate(anchor_ts):
        img_path = anchor_img_paths[i]
        id_path = None if id_paths[i] is None else str(id_paths[i])
        rgb_path = None if rgb_paths[i] is None else str(rgb_paths[i])

        # Primary LiDAR match: nearest actual LiDAR timestamp to the image timestamp.
        lidar_idx = lidar_match[i]

        # Robust fallback: if the label filename encodes a LiDAR timestamp, prefer matching by that.
        if lidar_idx is None and id_path is not None and lidar_ts:
            _, lbl_lidar_ts = parse_dual_timestamp_label_stem(id_path)
            if lbl_lidar_ts is not None:
                cand = pair_by_nearest_timestamp([float(lbl_lidar_ts)], lidar_ts, max_abs_delta=0.25)[0]
                if cand is not None:
                    lidar_idx = cand
        if lidar_idx is None and rgb_path is not None and lidar_ts:
            _, lbl_lidar_ts = parse_dual_timestamp_label_stem(rgb_path)
            if lbl_lidar_ts is not None:
                cand = pair_by_nearest_timestamp([float(lbl_lidar_ts)], lidar_ts, max_abs_delta=0.25)[0]
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
            T_world_cam=T_world_cam,
            T_world_lidar=T_world_cam.copy() if T_world_cam is not None else None,
            K=K,
            dist=dist,
        ))

    return records


# =========================
# Patched manifest builder
# =========================

def build_manifest_for_scene(scene_root: str | Path, cfg: dict | None = None) -> list[SampleRecord]:
    scene_root = Path(scene_root)
    scene_name = scene_root.name
    cfg = cfg or {}
    dcfg = cfg.get('dataset', {}) if isinstance(cfg, dict) else {}

    interval = dcfg.get('use_interval', None)
    cam_hint = str(dcfg.get('camera_folder_hint', 'CAM'))
    lidar_hint = str(dcfg.get('lidar_folder_hint', 'LiDAR'))
    label_id_hint = str(dcfg.get('lidar_label_id_hint', 'label_id'))
    label_rgb_hint = str(dcfg.get('lidar_label_rgb_hint', 'label_color'))
    sampleinfos_name = str(dcfg.get('sampleinfos_json_name', 'sampleinfos_interpolated.json'))

    sampleinfos = find_first_existing(scene_root, [sampleinfos_name])
    records: list[SampleRecord] = []

    cam_files = discover_camera_files(scene_root, interval=interval, camera_folder_hint=cam_hint)
    lidar_files = discover_lidar_files(scene_root, interval=interval, lidar_folder_hint=lidar_hint)
    label_id_files = discover_label_files(scene_root, mode='id', interval=interval, label_id_hint=label_id_hint, label_rgb_hint=label_rgb_hint, lidar_folder_hint=lidar_hint)
    label_rgb_files = discover_label_files(scene_root, mode='rgb', interval=interval, label_id_hint=label_id_hint, label_rgb_hint=label_rgb_hint, lidar_folder_hint=lidar_hint)

    cam_valid = [(p, timestamp_from_stem(p)) for p in cam_files]
    cam_valid = [(p, ts) for p, ts in cam_valid if ts is not None]
    lidar_valid = [(p, timestamp_from_stem(p)) for p in lidar_files]
    lidar_valid = [(p, ts) for p, ts in lidar_valid if ts is not None]
    label_id_info = label_file_dual_timestamp_info(label_id_files)
    label_rgb_info = label_file_dual_timestamp_info(label_rgb_files)

    cam_ts = [float(ts) for _, ts in cam_valid]
    lidar_ts = [float(ts) for _, ts in lidar_valid]

    if cam_valid:
        anchor_img_paths = [str(p) for p, _ in cam_valid]
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
    rec_ts: list[float] = []
    rec_match: list[int | None] = [None] * len(anchor_ts)
    if sampleinfos is not None:
        sample_records = read_sampleinfos_json(sampleinfos)
        raw_rec_ts = [infer_timestamp_from_record(r) for r in sample_records]
        valid_idx = [i for i, ts in enumerate(raw_rec_ts) if ts is not None]
        sample_records = [sample_records[i] for i in valid_idx]
        rec_ts = [float(raw_rec_ts[i]) for i in valid_idx]
        rec_match = pair_by_nearest_timestamp(anchor_ts, rec_ts, max_abs_delta=0.25)

    lidar_match = pair_by_nearest_timestamp(anchor_ts, lidar_ts, max_abs_delta=0.25)
    lidar_ts_for_anchor = [None if m is None else lidar_ts[m] for m in lidar_match]

    id_paths = _match_dual_timestamp_labels(anchor_ts, lidar_ts_for_anchor, label_id_info, max_img_delta=0.25, max_lidar_delta=0.25)
    rgb_paths = _match_dual_timestamp_labels(anchor_ts, lidar_ts_for_anchor, label_rgb_info, max_img_delta=0.25, max_lidar_delta=0.25)

    if any(p is None for p in id_paths):
        single_label_id = [(x['path'], x['img_ts']) for x in label_id_info if x['img_ts'] is not None]
        single_id_ts = [float(ts) for _, ts in single_label_id]
        nearest = pair_by_nearest_timestamp(anchor_ts, single_id_ts, max_abs_delta=0.25)
        for i, m in enumerate(nearest):
            if id_paths[i] is None and m is not None:
                id_paths[i] = single_label_id[m][0]

    if any(p is None for p in rgb_paths):
        single_label_rgb = [(x['path'], x['img_ts']) for x in label_rgb_info if x['img_ts'] is not None]
        single_rgb_ts = [float(ts) for _, ts in single_label_rgb]
        nearest = pair_by_nearest_timestamp(anchor_ts, single_rgb_ts, max_abs_delta=0.25)
        for i, m in enumerate(nearest):
            if rgb_paths[i] is None and m is not None:
                rgb_paths[i] = single_label_rgb[m][0]

    for i, ts in enumerate(anchor_ts):
        img_path = anchor_img_paths[i]
        id_path = None if id_paths[i] is None else str(id_paths[i])
        rgb_path = None if rgb_paths[i] is None else str(rgb_paths[i])

        lidar_idx = lidar_match[i]
        if lidar_idx is None and id_path is not None and lidar_ts:
            _, lbl_lidar_ts = parse_dual_timestamp_label_stem(id_path)
            if lbl_lidar_ts is not None:
                cand = pair_by_nearest_timestamp([float(lbl_lidar_ts)], lidar_ts, max_abs_delta=0.25)[0]
                if cand is not None:
                    lidar_idx = cand
        if lidar_idx is None and rgb_path is not None and lidar_ts:
            _, lbl_lidar_ts = parse_dual_timestamp_label_stem(rgb_path)
            if lbl_lidar_ts is not None:
                cand = pair_by_nearest_timestamp([float(lbl_lidar_ts)], lidar_ts, max_abs_delta=0.25)[0]
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
            T_world_cam=T_world_cam,
            T_world_lidar=T_world_cam.copy() if T_world_cam is not None else None,
            K=K,
            dist=dist,
        ))
    return records
