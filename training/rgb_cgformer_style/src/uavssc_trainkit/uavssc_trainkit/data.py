from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from .utils import discover_scene_npz, load_sample_list, npz_string, resolve_uav_path, split_scene_files


def _transpose_target_xyz_to_zyx(vol: np.ndarray) -> np.ndarray:
    vol = np.asarray(vol)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {vol.shape}")
    return np.transpose(vol, (2, 1, 0)).copy()


def _dense_lidar_channels(arr: np.lib.npyio.NpzFile) -> np.ndarray:
    occ = arr["input_occ_lidar"].astype(np.float32)
    den = np.log1p(arr["input_density_lidar"].astype(np.float32))
    maxh = arr["input_max_rel_height"].astype(np.float32)
    meanh = arr["input_mean_rel_height"].astype(np.float32)
    feats = [
        _transpose_target_xyz_to_zyx(occ),
        _transpose_target_xyz_to_zyx(den),
        _transpose_target_xyz_to_zyx(maxh),
        _transpose_target_xyz_to_zyx(meanh),
    ]
    return np.stack(feats, axis=0)


def _pil_to_tensor_norm(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[:, :, None]
    arr = np.transpose(arr, (2, 0, 1))
    t = torch.from_numpy(arr)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(-1, 1, 1)
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    return (t - mean) / std


def _light_color_jitter(img: Image.Image) -> Image.Image:
    if random.random() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
    if random.random() < 0.8:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))
    if random.random() < 0.5:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.9, 1.1))
    return img


def _optional_cam_label_tensor(arr: np.lib.npyio.NpzFile, data_root: str | None, image_size: Tuple[int, int]):
    """Load paired UAVScenes CAM_label id mask when an exported NPZ stores its path."""
    if "cam_label_id_path" not in arr.files:
        return None, ""
    raw_path = npz_string(arr["cam_label_id_path"])
    if not raw_path or raw_path.lower() in {"nan", "none", "<na>"}:
        return None, ""
    label_path = resolve_uav_path(raw_path, data_root)
    if not os.path.exists(label_path):
        return None, label_path
    mask = Image.open(label_path).convert("L")
    new_h, new_w = image_size
    mask = mask.resize((new_w, new_h), Image.NEAREST)
    return torch.from_numpy(np.asarray(mask, dtype=np.int64)), label_path


class BaseNPZDataset(Dataset):
    def __init__(
        self,
        preprocess_root: str | Path,
        split: str,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        scene_filter: Optional[Sequence[str]] = None,
        sample_list_path: str | Path | None = None,
    ):
        self.preprocess_root = Path(preprocess_root)
        self.split = split
        self.scene_filter = list(scene_filter) if scene_filter else None
        self.sample_list_path = sample_list_path

        if sample_list_path:
            self.samples = load_sample_list(sample_list_path, self.preprocess_root)
        else:
            scene_files = discover_scene_npz(self.preprocess_root, self.scene_filter)
            self.samples = split_scene_files(scene_files, split, split_ratios)

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split={split} under {preprocess_root}")

    def __len__(self):
        return len(self.samples)

    def get_sample_paths(self):
        return [str(p) for p in self.samples]


class RGBSSCNPZDataset(BaseNPZDataset):
    def __init__(
        self,
        preprocess_root: str | Path,
        data_root: str | None,
        split: str,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        scene_filter: Optional[Sequence[str]] = None,
        image_size: Tuple[int, int] = (640, 640),
        color_jitter: bool = False,
        sample_list_path: str | Path | None = None,
        return_cam_label: bool = False,
    ):
        super().__init__(preprocess_root, split, split_ratios, scene_filter, sample_list_path)
        self.data_root = data_root
        self.image_size = tuple(image_size)
        self.use_jitter = color_jitter
        self.return_cam_label = return_cam_label

    def __getitem__(self, index):
        npz_path = self.samples[index]
        arr = np.load(str(npz_path), allow_pickle=False)
        img_path = resolve_uav_path(npz_string(arr["img_path"]), self.data_root)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path} from {npz_path}")
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        if self.use_jitter:
            img = _light_color_jitter(img)
        new_h, new_w = self.image_size
        img = img.resize((new_w, new_h), Image.BILINEAR)
        img_t = _pil_to_tensor_norm(img)

        target = _transpose_target_xyz_to_zyx(arr["target"]).astype(np.int64)
        out = {
            "image": img_t,
            "target": torch.from_numpy(target),
            "grid_size_xyz": torch.from_numpy(arr["grid_size_xyz"].astype(np.int64)),
            "voxel_size": torch.tensor(float(arr["voxel_size"]), dtype=torch.float32),
            "vox_origin": torch.from_numpy(arr["vox_origin"].astype(np.float32)),
            "scene": npz_string(arr["scene"]),
            "timestamp": npz_string(arr["timestamp"]),
            "img_path": img_path,
            "npz_path": str(npz_path),
        }
        sx = float(new_w) / float(orig_w)
        sy = float(new_h) / float(orig_h)
        for name in arr.files:
            if name.startswith("projected_pix_"):
                scale = name.split("_")[-1]
                uv = arr[name].astype(np.float32)
                uv[:, 0] *= sx
                uv[:, 1] *= sy
                out[f"projected_pix_{scale}"] = torch.from_numpy(uv)
                out[f"fov_mask_{scale}"] = torch.from_numpy(arr[f"fov_mask_{scale}"].astype(bool))
                out[f"pix_z_{scale}"] = torch.from_numpy(arr[f"pix_z_{scale}"].astype(np.float32))

        if "cam_label_id_path" in arr.files:
            raw_cam_label_path = npz_string(arr["cam_label_id_path"])
            out["cam_label_id_path"] = resolve_uav_path(raw_cam_label_path, self.data_root) if raw_cam_label_path else ""
        if self.return_cam_label:
            cam_label, cam_label_path = _optional_cam_label_tensor(arr, self.data_root, self.image_size)
            if cam_label is not None:
                out["cam_label_id"] = cam_label
                out["cam_label_id_path"] = cam_label_path
        return out


class LidarSSCNPZDataset(BaseNPZDataset):
    def __init__(
        self,
        preprocess_root: str | Path,
        split: str,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        scene_filter: Optional[Sequence[str]] = None,
        sample_list_path: str | Path | None = None,
    ):
        super().__init__(preprocess_root, split, split_ratios, scene_filter, sample_list_path)

    def __getitem__(self, index):
        npz_path = self.samples[index]
        arr = np.load(str(npz_path), allow_pickle=False)
        x = _dense_lidar_channels(arr).astype(np.float32)
        target = _transpose_target_xyz_to_zyx(arr["target"]).astype(np.int64)
        return {
            "lidar_dense": torch.from_numpy(x),
            "target": torch.from_numpy(target),
            "scene": npz_string(arr["scene"]),
            "timestamp": npz_string(arr["timestamp"]),
            "grid_size_xyz": torch.from_numpy(arr["grid_size_xyz"].astype(np.int64)),
            "voxel_size": torch.tensor(float(arr["voxel_size"]), dtype=torch.float32),
            "npz_path": str(npz_path),
        }


class FusionSSCNPZDataset(BaseNPZDataset):
    def __init__(
        self,
        preprocess_root: str | Path,
        data_root: str | None,
        split: str,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        scene_filter: Optional[Sequence[str]] = None,
        image_size: Tuple[int, int] = (640, 640),
        color_jitter: bool = False,
        sample_list_path: str | Path | None = None,
        return_cam_label: bool = False,
    ):
        super().__init__(preprocess_root, split, split_ratios, scene_filter, sample_list_path)
        self.data_root = data_root
        self.image_size = tuple(image_size)
        self.use_jitter = color_jitter
        self.return_cam_label = return_cam_label

    def __getitem__(self, index):
        npz_path = self.samples[index]
        arr = np.load(str(npz_path), allow_pickle=False)
        img_path = resolve_uav_path(npz_string(arr["img_path"]), self.data_root)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path} from {npz_path}")
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        if self.use_jitter:
            img = _light_color_jitter(img)
        new_h, new_w = self.image_size
        img = img.resize((new_w, new_h), Image.BILINEAR)
        img_t = _pil_to_tensor_norm(img)
        target = _transpose_target_xyz_to_zyx(arr["target"]).astype(np.int64)
        out = {
            "image": img_t,
            "lidar_dense": torch.from_numpy(_dense_lidar_channels(arr).astype(np.float32)),
            "target": torch.from_numpy(target),
            "grid_size_xyz": torch.from_numpy(arr["grid_size_xyz"].astype(np.int64)),
            "voxel_size": torch.tensor(float(arr["voxel_size"]), dtype=torch.float32),
            "scene": npz_string(arr["scene"]),
            "timestamp": npz_string(arr["timestamp"]),
            "img_path": img_path,
            "npz_path": str(npz_path),
        }
        sx = float(new_w) / float(orig_w)
        sy = float(new_h) / float(orig_h)
        for name in arr.files:
            if name.startswith("projected_pix_"):
                scale = name.split("_")[-1]
                uv = arr[name].astype(np.float32)
                uv[:, 0] *= sx
                uv[:, 1] *= sy
                out[f"projected_pix_{scale}"] = torch.from_numpy(uv)
                out[f"fov_mask_{scale}"] = torch.from_numpy(arr[f"fov_mask_{scale}"].astype(bool))
                out[f"pix_z_{scale}"] = torch.from_numpy(arr[f"pix_z_{scale}"].astype(np.float32))

        if "cam_label_id_path" in arr.files:
            raw_cam_label_path = npz_string(arr["cam_label_id_path"])
            out["cam_label_id_path"] = resolve_uav_path(raw_cam_label_path, self.data_root) if raw_cam_label_path else ""
        if self.return_cam_label:
            cam_label, cam_label_path = _optional_cam_label_tensor(arr, self.data_root, self.image_size)
            if cam_label is not None:
                out["cam_label_id"] = cam_label
                out["cam_label_id_path"] = cam_label_path
        return out
