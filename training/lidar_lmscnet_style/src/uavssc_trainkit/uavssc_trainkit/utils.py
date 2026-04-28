import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

IGNORE_LABEL = 255

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_json(path: str | Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def npz_string(v):
    if isinstance(v, np.ndarray):
        if v.shape == ():
            v = v.item()
        elif v.size == 1:
            v = v.reshape(-1)[0]
    if isinstance(v, bytes):
        v = v.decode("utf-8")
    return str(v)

def normalize_path(p):
    return str(p).replace("\\", "/")

def resolve_uav_path(img_path: str, data_root: str | None = None) -> str:
    img_path = normalize_path(img_path)
    if os.path.isabs(img_path) and os.path.exists(img_path):
        return img_path
    data_root = data_root or os.environ.get("UAVSSC_DATA_ROOT", None)
    if data_root is None:
        return os.path.normpath(img_path)
    data_root = normalize_path(str(data_root))
    if os.path.exists(img_path):
        return os.path.normpath(img_path)
    marker = "/UAVScenes/"
    if marker in img_path:
        suffix = img_path.split(marker, 1)[1]
        cand = os.path.join(data_root, suffix)
        if os.path.exists(cand):
            return os.path.normpath(cand)
    if "UAVScenes/" in img_path:
        suffix = img_path.split("UAVScenes/", 1)[1]
        cand = os.path.join(data_root, suffix)
        if os.path.exists(cand):
            return os.path.normpath(cand)
    if "UAVScenes\\" in img_path:
        suffix = img_path.split("UAVScenes\\", 1)[1].replace("\\", "/")
        cand = os.path.join(data_root, suffix)
        if os.path.exists(cand):
            return os.path.normpath(cand)
    tail_parts = Path(img_path).parts
    for keep in range(min(len(tail_parts), 6), 0, -1):
        tail = os.path.join(*tail_parts[-keep:])
        cand = os.path.join(data_root, tail)
        if os.path.exists(cand):
            return os.path.normpath(cand)
    return os.path.normpath(img_path)

def stem_timestamp(path: str | Path):
    stem = Path(path).stem
    try:
        return float(stem)
    except Exception:
        return stem

def discover_scene_npz(preprocess_root: str | Path, scene_filter: Optional[Sequence[str]] = None) -> Dict[str, List[Path]]:
    preprocess_root = Path(preprocess_root)
    scenes = {}
    allow = set(scene_filter) if scene_filter else None
    for p in sorted(preprocess_root.iterdir()):
        if not p.is_dir():
            continue
        if allow is not None and p.name not in allow:
            continue
        files = sorted(p.glob("*.npz"), key=lambda x: stem_timestamp(x))
        if files:
            scenes[p.name] = files
    if not scenes:
        raise RuntimeError(f"No scene folders with npz files found under {preprocess_root}")
    return scenes

def split_scene_files(scene_files: Dict[str, List[Path]], split: str, split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> List[Path]:
    train_r, val_r, test_r = split_ratios
    total = train_r + val_r + test_r
    if abs(total - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")
    selected = []
    for _, files in scene_files.items():
        n = len(files)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        n_test = n - n_train - n_val
        if split == "train":
            part = files[:n_train]
        elif split == "val":
            part = files[n_train:n_train + n_val]
        elif split == "test":
            part = files[n_train + n_val:n_train + n_val + n_test]
        else:
            raise ValueError(split)
        selected.extend(part)
    return selected

def infer_num_classes(npz_paths: Sequence[str | Path], fallback: int = 26) -> int:
    max_id = -1
    for p in npz_paths:
        try:
            arr = np.load(str(p), allow_pickle=False)
            target = arr["target"]
            valid = target[target != IGNORE_LABEL]
            if valid.size > 0:
                max_id = max(max_id, int(valid.max()))
        except Exception:
            continue
    return max(max_id + 1, fallback if max_id < 0 else max_id + 1)

def compute_log_class_weights(npz_paths: Sequence[str | Path], num_classes: int) -> torch.Tensor:
    counts = np.ones(num_classes, dtype=np.float64)
    for p in npz_paths:
        arr = np.load(str(p), allow_pickle=False)
        target = arr["target"].astype(np.int64, copy=False)
        valid = target[target != IGNORE_LABEL]
        if valid.size == 0:
            continue
        binc = np.bincount(valid.reshape(-1), minlength=num_classes)
        counts[: len(binc)] += binc[:num_classes]
    w = 1.0 / np.log(counts + 1.02)
    w = w / w.mean()
    return torch.from_numpy(w.astype(np.float32))

def move_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(x, device) for x in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(x, device) for x in batch)
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    return batch


def load_sample_list(sample_list_path: str | Path | None, preprocess_root: str | Path | None = None) -> List[Path]:
    """Load an explicit list of NPZ samples.

    Each non-empty, non-comment line may be:
    - an absolute path
    - a path relative to the split file
    - a path relative to preprocess_root
    - a path of the form scene_name/file.npz
    """
    if sample_list_path is None:
        return []
    sample_list_path = Path(sample_list_path)
    if not sample_list_path.exists():
        raise FileNotFoundError(f"Split file not found: {sample_list_path}")
    preprocess_root = Path(preprocess_root) if preprocess_root is not None else None
    out: List[Path] = []
    with open(sample_list_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.replace("\\", "/")
            candidates = []
            p = Path(line)
            if p.is_absolute():
                candidates.append(p)
            candidates.append(sample_list_path.parent / p)
            if preprocess_root is not None:
                candidates.append(preprocess_root / p)
            resolved = None
            for cand in candidates:
                if cand.exists():
                    resolved = cand.resolve()
                    break
            if resolved is None:
                # keep a best-effort path for clearer downstream error messages
                resolved = (preprocess_root / p).resolve() if preprocess_root is not None else (sample_list_path.parent / p).resolve()
            out.append(resolved)
    if len(out) == 0:
        raise RuntimeError(f"No sample paths found in split file: {sample_list_path}")
    return out
