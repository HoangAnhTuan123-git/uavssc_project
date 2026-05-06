import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class UAVScenesDataset(Dataset):
    def __init__(
        self,
        split,
        preprocess_root,
        project_scale=2,
        frustum_size=4,
        color_jitter=None,
        fliplr=0.0,
        split_ratios=(0.7, 0.15, 0.15),
        scene_filter=None,
        data_root=None,
        sample_list_path=None,
        input_image_hw=None,
    ):
        super(UAVScenesDataset, self).__init__()
        assert split in ["train", "val", "test"]
        self.split = split
        self.preprocess_root = preprocess_root
        self.project_scale = int(project_scale)
        self.output_scale = int(self.project_scale / 2)
        self.frustum_size = frustum_size
        self.fliplr = fliplr
        self.scene_filter = scene_filter
        self.data_root = data_root or os.environ.get("UAVSSC_DATA_ROOT", None)
        self.sample_list_path = sample_list_path
        self.input_image_hw = self._normalize_hw(input_image_hw)

        self.color_jitter = transforms.ColorJitter(*color_jitter) if color_jitter else None
        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.samples = self._discover_samples(split_ratios)
        if len(self.samples) == 0:
            raise RuntimeError(
                "No UAVScenes samples found for split='{}' under {}".format(split, preprocess_root)
            )

    @staticmethod
    def _stem_timestamp(path_str):
        stem = Path(path_str).stem
        try:
            return float(stem)
        except Exception:
            return stem

    @staticmethod
    def _normalize_path_string(path_str):
        if isinstance(path_str, bytes):
            path_str = path_str.decode("utf-8")
        path_str = str(path_str)
        path_str = path_str.replace("\\", os.sep)
        return path_str

    @staticmethod
    def _load_npz_string(value):
        if isinstance(value, np.ndarray):
            if value.shape == ():
                value = value.item()
            elif value.size == 1:
                value = value.reshape(-1)[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return str(value)

    @staticmethod
    def _normalize_hw(value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            if value.size < 2:
                return None
            flat = value.reshape(-1)
            return int(flat[0]), int(flat[1])
        if isinstance(value, str):
            txt = value.strip()
            if not txt:
                return None
            txt = txt.replace("[", "").replace("]", "").replace("x", ",")
            parts = [p.strip() for p in txt.split(",") if p.strip()]
            if len(parts) < 2:
                return None
            return int(float(parts[0])), int(float(parts[1]))
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return int(value[0]), int(value[1])
        return None

    @classmethod
    def _npz_hw(cls, arr, key):
        if key not in arr.files:
            return None
        return cls._normalize_hw(arr[key])

    @classmethod
    def _projection_hw(cls, arr, fallback_hw):
        return (
            cls._npz_hw(arr, "projection_image_shape_hw")
            or cls._npz_hw(arr, "image_shape_hw")
            or fallback_hw
        )

    @staticmethod
    def _scale_cam_k(K, sx, sy):
        K = K.astype(np.float32).copy()
        K[0, 0] *= float(sx)
        K[0, 2] *= float(sx)
        K[1, 1] *= float(sy)
        K[1, 2] *= float(sy)
        return K

    def _resolve_img_path(self, img_path):
        img_path = img_path.replace("\\", "/")
        if os.path.isabs(img_path) and os.path.exists(img_path):
            return img_path
        if os.path.exists(img_path):
            return os.path.normpath(img_path)

        if self.data_root:
            marker = "/UAVScenes/"
            if marker in img_path:
                suffix = img_path.split(marker, 1)[1]
                cand = os.path.join(self.data_root, suffix)
                if os.path.exists(cand):
                    return os.path.normpath(cand)
            if "UAVScenes/" in img_path:
                suffix = img_path.split("UAVScenes/", 1)[1]
                cand = os.path.join(self.data_root, suffix)
                if os.path.exists(cand):
                    return os.path.normpath(cand)
            if "UAVScenes\\" in img_path:
                suffix = img_path.split("UAVScenes\\", 1)[1].replace("\\", "/")
                cand = os.path.join(self.data_root, suffix)
                if os.path.exists(cand):
                    return os.path.normpath(cand)
            parts = Path(img_path).parts
            for keep in range(min(len(parts), 6), 0, -1):
                tail = os.path.join(*parts[-keep:])
                cand = os.path.join(self.data_root, tail)
                if os.path.exists(cand):
                    return os.path.normpath(cand)
        return os.path.normpath(img_path)


    @staticmethod
    def _load_sample_list(sample_list_path, preprocess_root):
        sample_list_path = Path(sample_list_path)
        if not sample_list_path.exists():
            raise FileNotFoundError("Split file not found: {}".format(sample_list_path))
        out = []
        with open(sample_list_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                line = line.replace("\\", "/")
                p = Path(line)
                candidates = []
                if p.is_absolute():
                    candidates.append(p)
                candidates.append(sample_list_path.parent / p)
                candidates.append(Path(preprocess_root) / p)
                resolved = None
                for cand in candidates:
                    if cand.exists():
                        resolved = cand.resolve()
                        break
                if resolved is None:
                    resolved = (Path(preprocess_root) / p).resolve()
                out.append(resolved)
        if len(out) == 0:
            raise RuntimeError("No sample paths found in split file: {}".format(sample_list_path))
        return out

    def _discover_samples(self, split_ratios):
        train_ratio, val_ratio, test_ratio = split_ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError("split ratios must sum to 1.0")

        if self.sample_list_path:
            return self._load_sample_list(self.sample_list_path, self.preprocess_root)

        scene_dirs = []
        for p in sorted(Path(self.preprocess_root).iterdir()):
            if p.is_dir():
                if self.scene_filter and p.name not in self.scene_filter:
                    continue
                scene_dirs.append(p)

        selected = []
        for scene_dir in scene_dirs:
            files = sorted(scene_dir.glob("*.npz"), key=lambda x: self._stem_timestamp(str(x)))
            n = len(files)
            if n == 0:
                continue

            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            n_test = n - n_train - n_val

            if self.split == "train":
                part = files[:n_train]
            elif self.split == "val":
                part = files[n_train:n_train + n_val]
            else:
                part = files[n_train + n_val:n_train + n_val + n_test]

            selected.extend(part)

        return selected

    def get_sample_paths(self):
        return [str(p) for p in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        npz_path = self.samples[index]
        arr = np.load(str(npz_path), allow_pickle=False)

        scene = self._load_npz_string(arr["scene"])
        frame_id = Path(str(npz_path)).stem
        img_path = self._normalize_path_string(self._load_npz_string(arr["img_path"]))
        img_path = self._resolve_img_path(img_path)

        orig_projection_hw = self._projection_hw(arr, fallback_hw=None)

        data = {
            "frame_id": frame_id,
            "sequence": scene,
            "img_path": img_path,
            "cam_E": arr["cam_E"].astype(np.float32),
            "target": arr["target"].astype(np.uint8),
            "CP_mega_matrix": arr["CP_mega_matrix"].astype(np.uint8),
        }

        scale_3ds = []
        for k in arr.files:
            if k.startswith("projected_pix_"):
                scale_3ds.append(int(k.split("_")[-1]))
        scale_3ds = sorted(scale_3ds)
        data["scale_3ds"] = scale_3ds

        if "frustums_masks" in arr.files:
            data["frustums_masks"] = arr["frustums_masks"].astype(bool)
        else:
            data["frustums_masks"] = None

        if "frustums_class_dists" in arr.files:
            data["frustums_class_dists"] = arr["frustums_class_dists"].astype(np.int32)
        else:
            data["frustums_class_dists"] = None

        if not os.path.exists(img_path):
            raise FileNotFoundError(
                "Image path inside npz does not exist: {} (from {})".format(img_path, npz_path)
            )

        img_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_pil.size
        if orig_projection_hw is None:
            orig_projection_hw = (orig_h, orig_w)

        target_hw = self.input_image_hw or self._npz_hw(arr, "image_shape_hw") or orig_projection_hw
        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        proj_h, proj_w = int(orig_projection_hw[0]), int(orig_projection_hw[1])
        if target_h <= 0 or target_w <= 0:
            raise ValueError("Invalid input image size: {}".format(target_hw))

        sx = float(target_w) / float(proj_w)
        sy = float(target_h) / float(proj_h)

        for scale in scale_3ds:
            key = "projected_pix_{}".format(scale)
            uv = arr[key].astype(np.float32).copy()
            uv[:, 0] *= sx
            uv[:, 1] *= sy
            uv = np.rint(uv).astype(np.int32)
            data[key] = uv
            data["fov_mask_{}".format(scale)] = arr["fov_mask_{}".format(scale)].astype(bool)
            data["pix_z_{}".format(scale)] = arr["pix_z_{}".format(scale)].astype(np.float32)

        data["cam_k"] = self._scale_cam_k(arr["cam_k"].astype(np.float32), sx, sy)
        data["image_shape_hw"] = np.asarray([target_h, target_w], dtype=np.int32)
        data["projection_shape_hw"] = np.asarray([proj_h, proj_w], dtype=np.int32)

        if img_pil.size != (target_w, target_h):
            img_pil = img_pil.resize((target_w, target_h), Image.BILINEAR)
        if self.color_jitter is not None:
            img_pil = self.color_jitter(img_pil)
        img = np.array(img_pil, dtype=np.float32, copy=False) / 255.0

        if np.random.rand() < self.fliplr:
            img = np.ascontiguousarray(np.fliplr(img))
            width = img.shape[1]
            for scale in scale_3ds:
                key = "projected_pix_{}".format(scale)
                data[key][:, 0] = width - 1 - data[key][:, 0]

        data["img"] = self.normalize_rgb(img)
        return data
