from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd

from .transforms import make_transform, quaternion_xyzw_to_rotmat


@dataclass
class SensorCalibration:
    K: np.ndarray | None
    dist: np.ndarray | None
    T_cam_lidar: np.ndarray | None
    T_world_cam: np.ndarray | None
    T_world_lidar: np.ndarray | None


@dataclass
class SampleRecord:
    scene: str
    run: str
    timestamp: float
    img_path: str | None
    lidar_path: str | None
    lidar_label_id_path: str | None
    lidar_label_rgb_path: str | None
    T_world_cam: np.ndarray | None
    T_world_lidar: np.ndarray | None
    K: np.ndarray | None
    dist: np.ndarray | None


COMMON_K_KEYS = [
    'K', 'camera_matrix', 'intrinsic', 'intrinsics', 'cam_intrinsic', 'camera_intrinsic',
    # UAVScenes
    'P3x3',
]
COMMON_DIST_KEYS = [
    'dist', 'dist_coeffs', 'distortion', 'distortion_coefficients', 'D'
]
# UAVScenes stores distortion as scalar fields K1,K2,K3,P1,P2
UAVSCENES_DIST_SCALAR_KEYS = ['K1', 'K2', 'P1', 'P2', 'K3']
COMMON_TRANSLATION_KEYS = ['t', 'translation', 'trans', 'position', 'xyz']
COMMON_QUAT_KEYS = ['quat_xyzw', 'quaternion_xyzw', 'q_xyzw', 'quat']
COMMON_ROT_KEYS = ['R', 'rotation_matrix', 'rotmat', 'rotation']
COMMON_POSE_KEYS = ['pose', 'extrinsic', 'T_world_cam', 'T_wc', 'cam2world', 'camera_pose', 'T4x4']
COMMON_TIMESTAMP_KEYS = ['timestamp', 'headerstamp', 'time', 'ts']
COMMON_IMAGE_KEYS = ['img_path', 'image_path', 'rgb_path', 'img_file', 'image_file', 'img', 'OriginalImageName']
DUAL_TS_RE = re.compile(r'image(?P<img>[0-9]+(?:\.[0-9]+)?)_lidar(?P<lidar>[0-9]+(?:\.[0-9]+)?)')


def read_json(path: str | Path) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)



def read_image(path: str | Path, rgb: bool = True) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Failed to read image: {path}')
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img



def read_lidar_txt(path: str | Path) -> np.ndarray:
    pts = np.loadtxt(str(path), dtype=np.float32)
    pts = np.atleast_2d(pts)
    if pts.shape[1] < 3:
        raise ValueError(f'Expected >=3 columns in LiDAR txt, got {pts.shape}')
    return pts[:, :3]



def read_label_id_txt(path: str | Path) -> np.ndarray:
    arr = np.loadtxt(str(path), dtype=np.int32)
    return np.atleast_1d(arr).astype(np.int32)



def read_label_rgb_txt(path: str | Path) -> np.ndarray:
    arr = np.loadtxt(str(path), dtype=np.int32)
    arr = np.atleast_2d(arr)
    if arr.shape[1] != 3:
        raise ValueError(f'Expected (N,3) RGB label txt, got {arr.shape}')
    return arr



def try_read_rtk_excel(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    if path.suffix.lower() in {'.csv', '.txt'}:
        return pd.read_csv(path)
    raise ValueError(f'Unsupported RTK table format: {path}')



def maybe_import_python_file(path: str | Path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import python file: {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod



def parse_calibration_results_py(path: str | Path) -> dict[str, Any]:
    mod = maybe_import_python_file(path)
    out = {}
    for name in dir(mod):
        if name.startswith('_'):
            continue
        out[name] = getattr(mod, name)
    return out



def find_first_existing(root: str | Path, names: Iterable[str]) -> Path | None:
    root = Path(root)
    names = set(names)
    for p in root.rglob('*'):
        if p.name in names:
            return p
    return None



def discover_scene_dirs(data_root: str | Path) -> list[Path]:
    data_root = Path(data_root)
    out = []
    for p in sorted([x for x in data_root.iterdir() if x.is_dir()]):
        name = p.name.lower()
        if name.startswith('__pycache__'):
            continue
        if 'terra_3dmap_pointcloud_mesh' in name:
            continue
        out.append(p)
    return out



def infer_timestamp_from_record(record: dict[str, Any]) -> float | None:
    # 1) explicit timestamp fields
    for k in COMMON_TIMESTAMP_KEYS:
        if k in record:
            try:
                return float(record[k])
            except Exception:
                pass

    # 2) image-like fields, including UAVScenes OriginalImageName
    for k in COMMON_IMAGE_KEYS:
        if k in record:
            try:
                return float(Path(str(record[k])).stem)
            except Exception:
                pass

    # 3) defensive fallback for any string field that looks like a jpg/png filename
    for v in record.values():
        if isinstance(v, str) and (v.lower().endswith('.jpg') or v.lower().endswith('.png') or v.lower().endswith('.jpeg')):
            try:
                return float(Path(v).stem)
            except Exception:
                pass
    return None



def _extract_nested(d: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None



def infer_intrinsics_from_record(record: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    K = _extract_nested(record, COMMON_K_KEYS)
    D = _extract_nested(record, COMMON_DIST_KEYS)

    K_arr = None
    if K is not None:
        K_arr = np.asarray(K, dtype=np.float64).reshape(3, 3)

    D_arr = None
    if D is not None:
        D_arr = np.asarray(D, dtype=np.float64).reshape(-1)
    else:
        # UAVScenes: K1,K2,P1,P2,K3 stored as separate scalars
        vals = []
        have_any = False
        for k in UAVSCENES_DIST_SCALAR_KEYS:
            if k in record:
                have_any = True
                vals.append(float(record[k]))
            else:
                vals.append(0.0)
        if have_any:
            D_arr = np.asarray(vals, dtype=np.float64)

    return K_arr, D_arr



def infer_pose_matrix_from_record(record: dict[str, Any]) -> np.ndarray | None:
    for k in COMMON_POSE_KEYS:
        if k in record:
            v = np.asarray(record[k], dtype=np.float64)
            if v.shape == (4, 4):
                return v
            if v.shape == (3, 4):
                T = np.eye(4, dtype=np.float64)
                T[:3, :] = v
                return T

    R = _extract_nested(record, COMMON_ROT_KEYS)
    t = _extract_nested(record, COMMON_TRANSLATION_KEYS)
    if R is not None and t is not None:
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        t = np.asarray(t, dtype=np.float64).reshape(3)
        return make_transform(R, t)

    q = _extract_nested(record, COMMON_QUAT_KEYS)
    t = _extract_nested(record, COMMON_TRANSLATION_KEYS)
    if q is not None and t is not None:
        R = quaternion_xyzw_to_rotmat(np.asarray(q, dtype=np.float64).reshape(4))
        t = np.asarray(t, dtype=np.float64).reshape(3)
        return make_transform(R, t)

    return None



def read_sampleinfos_json(path: str | Path) -> list[dict[str, Any]]:
    data = read_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ['samples', 'frames', 'data', 'records', 'infos']:
            if key in data and isinstance(data[key], list):
                return data[key]
        if all(isinstance(v, dict) for v in data.values()):
            return list(data.values())
    raise TypeError(f'Unsupported JSON top-level structure in {path}')



def infer_image_path_from_record(record: dict[str, Any], scene_root: str | Path) -> str | None:
    scene_root = Path(scene_root)
    for k in COMMON_IMAGE_KEYS:
        if k in record:
            p = Path(str(record[k]))
            if p.is_absolute() and p.exists():
                return str(p)
            cand = scene_root / p
            if cand.exists():
                return str(cand)
            matches = list(scene_root.rglob(p.name))
            if matches:
                return str(matches[0])

    # UAVScenes-specific defensive fallback
    if 'OriginalImageName' in record:
        name = Path(str(record['OriginalImageName'])).name
        matches = list(scene_root.rglob(name))
        if matches:
            return str(matches[0])

    return None



def discover_camera_files(scene_root: str | Path) -> list[Path]:
    scene_root = Path(scene_root)
    files = [
        p for p in scene_root.rglob('*')
        if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'}
    ]
    return sorted(files)



def discover_lidar_files(scene_root: str | Path) -> list[Path]:
    scene_root = Path(scene_root)
    files = [
        p for p in scene_root.rglob('*.txt')
        if 'label' not in p.as_posix().lower() and 'rtk' not in p.as_posix().lower()
    ]
    out = []
    for p in files:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                line = f.readline().strip().split()
            if len(line) >= 3:
                float(line[0]); float(line[1]); float(line[2])
                out.append(p)
        except Exception:
            continue
    return sorted(out)



def discover_label_files(scene_root: str | Path, mode: str) -> list[Path]:
    assert mode in {'id', 'rgb'}
    files = [p for p in Path(scene_root).rglob('*.txt') if 'label' in p.as_posix().lower()]
    out = []
    for p in files:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                line = f.readline().strip().split()
            if mode == 'id' and len(line) == 1:
                int(float(line[0]))
                out.append(p)
            if mode == 'rgb' and len(line) == 3:
                int(float(line[0])); int(float(line[1])); int(float(line[2]))
                out.append(p)
        except Exception:
            continue
    return sorted(out)



def scene_prefix(scene_name: str) -> str:
    m = re.match(r'([A-Za-z_]+)', scene_name)
    return m.group(1) if m else scene_name



def infer_scene_calibration_dict(calib_vars: dict[str, Any], scene_name: str) -> dict[str, Any] | None:
    prefix = scene_prefix(scene_name)
    candidates = []
    for k, v in calib_vars.items():
        if not isinstance(v, dict):
            continue
        if prefix.lower() in k.lower():
            candidates.append(v)
    if candidates:
        return candidates[0]
    return None



def calibration_dict_to_matrices(calib_dict: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if calib_dict is None:
        return None, None, None
    K = None
    D = None
    T = None
    if 'camera_intrinsic' in calib_dict:
        K = np.asarray(calib_dict['camera_intrinsic'], dtype=np.float64).reshape(3, 3)
    elif 'K' in calib_dict:
        K = np.asarray(calib_dict['K'], dtype=np.float64).reshape(3, 3)

    if 'camera_dist_coeffs' in calib_dict:
        D = np.asarray(calib_dict['camera_dist_coeffs'], dtype=np.float64).reshape(-1)
    elif 'dist' in calib_dict:
        D = np.asarray(calib_dict['dist'], dtype=np.float64).reshape(-1)

    if 'camera_ext_R' in calib_dict and 'camera_ext_t' in calib_dict:
        R = np.asarray(calib_dict['camera_ext_R'], dtype=np.float64).reshape(3, 3)
        t = np.asarray(calib_dict['camera_ext_t'], dtype=np.float64).reshape(3)
        T = make_transform(R, t)
    return K, D, T



def parse_dual_timestamp_label_stem(path_or_stem: str | Path) -> tuple[float | None, float | None]:
    stem = Path(path_or_stem).stem
    m = DUAL_TS_RE.search(stem)
    if not m:
        return None, None
    try:
        img_ts = float(m.group('img'))
        lidar_ts = float(m.group('lidar'))
        return img_ts, lidar_ts
    except Exception:
        return None, None



def label_file_dual_timestamp_info(paths: list[Path]) -> list[dict[str, Any]]:
    info = []
    for p in paths:
        img_ts, lidar_ts = parse_dual_timestamp_label_stem(p)
        info.append({
            'path': p,
            'img_ts': img_ts,
            'lidar_ts': lidar_ts,
            'stem': p.stem,
        })
    return info



def read_ply_header_counts(path: str | Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with open(path, 'rb') as f:
        while True:
            line = f.readline()
            if not line:
                break
            s = line.decode('utf-8', errors='ignore').strip()
            if s.startswith('element '):
                _, name, n = s.split()
                counts[name] = int(n)
            if s == 'end_header':
                break
    return counts


# =========================
# Patched discovery helpers
# =========================

def _preferred_interval_dirs(scene_root: str | Path, interval: int | None, required_tokens: list[str], forbidden_tokens: list[str] | None = None) -> list[Path]:
    scene_root = Path(scene_root)
    forbidden_tokens = forbidden_tokens or []
    interval_token = None if interval is None else f'interval{int(interval)}'
    dirs: list[Path] = []
    for p in scene_root.rglob('*'):
        if not p.is_dir():
            continue
        name = p.name.lower()
        full = p.as_posix().lower()
        if interval_token is not None and interval_token not in name and interval_token not in full:
            continue
        if any(tok.lower() not in name and tok.lower() not in full for tok in required_tokens):
            continue
        if any(tok.lower() in name or tok.lower() in full for tok in forbidden_tokens):
            continue
        dirs.append(p)
    return sorted(dirs)


def discover_camera_files(scene_root: str | Path, interval: int | None = None, camera_folder_hint: str = 'CAM') -> list[Path]:
    scene_root = Path(scene_root)
    preferred_dirs = _preferred_interval_dirs(
        scene_root,
        interval=interval,
        required_tokens=[camera_folder_hint],
        forbidden_tokens=['label', 'terra_3dmap_pointcloud_mesh', '.git'],
    )
    search_roots = preferred_dirs if preferred_dirs else [scene_root]
    files: list[Path] = []
    for root in search_roots:
        for p in root.rglob('*'):
            s = p.as_posix().lower()
            if not p.is_file():
                continue
            if p.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
                continue
            if 'label' in s or 'terra_3dmap_pointcloud_mesh' in s:
                continue
            files.append(p)
    return sorted(set(files))


def discover_lidar_files(scene_root: str | Path, interval: int | None = None, lidar_folder_hint: str = 'LiDAR') -> list[Path]:
    scene_root = Path(scene_root)
    preferred_dirs = _preferred_interval_dirs(
        scene_root,
        interval=interval,
        required_tokens=[lidar_folder_hint],
        forbidden_tokens=['label', 'rtk', 'terra_3dmap_pointcloud_mesh', '.git'],
    )
    search_roots = preferred_dirs if preferred_dirs else [scene_root]
    files: list[Path] = []
    for root in search_roots:
        for p in root.rglob('*.txt'):
            s = p.as_posix().lower()
            if 'label' in s or 'rtk' in s or 'terra_3dmap_pointcloud_mesh' in s:
                continue
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    line = f.readline().strip().split()
                if len(line) >= 3:
                    float(line[0]); float(line[1]); float(line[2])
                    files.append(p)
            except Exception:
                continue
    return sorted(set(files))


def discover_label_files(
    scene_root: str | Path,
    mode: str,
    interval: int | None = None,
    label_id_hint: str = 'label_id',
    label_rgb_hint: str = 'label_color',
    lidar_folder_hint: str = 'LiDAR',
) -> list[Path]:
    assert mode in {'id', 'rgb'}
    scene_root = Path(scene_root)

    preferred_dirs: list[Path] = []
    if mode == 'id':
        preferred_dirs.extend(_preferred_interval_dirs(scene_root, interval, [lidar_folder_hint, 'label', label_id_hint], forbidden_tokens=['terra_3dmap_pointcloud_mesh', '.git']))
        preferred_dirs.extend(_preferred_interval_dirs(scene_root, interval, ['label', label_id_hint], forbidden_tokens=['terra_3dmap_pointcloud_mesh', '.git']))
    else:
        preferred_dirs.extend(_preferred_interval_dirs(scene_root, interval, [lidar_folder_hint, 'label', label_rgb_hint], forbidden_tokens=['terra_3dmap_pointcloud_mesh', '.git']))
        preferred_dirs.extend(_preferred_interval_dirs(scene_root, interval, ['label', label_rgb_hint], forbidden_tokens=['terra_3dmap_pointcloud_mesh', '.git']))

    if not preferred_dirs:
        preferred_dirs = _preferred_interval_dirs(scene_root, interval, ['label'], forbidden_tokens=['terra_3dmap_pointcloud_mesh', '.git'])
    search_roots = preferred_dirs if preferred_dirs else [scene_root]

    out: list[Path] = []
    for root in search_roots:
        for p in root.rglob('*.txt'):
            s = p.as_posix().lower()
            if 'label' not in s:
                continue
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    line = f.readline().strip().split()
                if mode == 'id' and len(line) == 1:
                    int(float(line[0]))
                    out.append(p)
                if mode == 'rgb' and len(line) == 3:
                    int(float(line[0])); int(float(line[1])); int(float(line[2]))
                    out.append(p)
            except Exception:
                continue
    return sorted(set(out))
