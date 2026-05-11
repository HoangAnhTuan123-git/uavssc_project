from pathlib import Path
import argparse
import sys
import random

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from uavssc_trainkit.uavssc_trainkit.data import RGBSSCNPZDataset
from uavssc_trainkit.uavssc_trainkit.models_rgb import VoxFormerStyleSSC
from uavssc_trainkit.uavssc_trainkit.utils import load_yaml

COLORS = np.array([
    [0,0,0], [70,70,70], [128,64,128], [128,128,128], [0,0,255], [0,120,255],
    [150,100,100], [100,100,40], [180,180,180], [255,255,0], [107,142,35],
    [80,120,40], [255,0,255], [255,120,0], [180,220,220], [100,100,100], [160,160,160],
], dtype=np.float32) / 255.0
CMAP = ListedColormap(COLORS)


def apply_orientation(a: np.ndarray, mode: str) -> np.ndarray:
    if mode == "identity": return a
    if mode == "flipud": return np.flipud(a)
    if mode == "fliplr": return np.fliplr(a)
    if mode == "rot180": return np.rot90(a, 2)
    if mode == "transpose": return a.T
    if mode == "transpose_flipud": return np.flipud(a.T)
    if mode == "transpose_fliplr": return np.fliplr(a.T)
    if mode == "transpose_rot180": return np.rot90(a.T, 2)
    raise ValueError(f"Unknown orientation: {mode}")


def denorm_image(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu().float()
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    x = (x * std + mean).clamp(0, 1)
    return (x.permute(1,2,0).numpy() * 255).astype(np.uint8)


def topdown_semantic(vol_zyx: np.ndarray, orientation: str) -> np.ndarray:
    vol = np.asarray(vol_zyx)
    occ = (vol > 0) & (vol != 255)
    zdim, ydim, xdim = vol.shape
    td = np.zeros((ydim, xdim), dtype=np.int64)
    for z in range(zdim - 1, -1, -1):
        m = occ[z]
        td[m] = vol[z][m]
    return apply_orientation(td, orientation)


def topdown_height(vol_zyx: np.ndarray, orientation: str) -> np.ndarray:
    vol = np.asarray(vol_zyx)
    occ = (vol > 0) & (vol != 255)
    zdim, ydim, xdim = vol.shape
    h = np.zeros((ydim, xdim), dtype=np.float32)
    for z in range(zdim):
        m = occ[z]
        h[m] = z + 1
    if h.max() > 0:
        h /= h.max()
    return apply_orientation(h, orientation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--orientation", default="identity", choices=["identity","flipud","fliplr","rot180","transpose","transpose_flipud","transpose_fliplr","transpose_rot180"])
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    split_ratios = tuple(cfg["data"].get("split_ratios", [0.7,0.15,0.15]))
    image_size = tuple(cfg["data"].get("image_size", [224,320]))
    split_files = cfg["data"].get("split_files", {}) or {}
    num_classes = int(cfg["model"].get("num_classes", 17))

    ds = RGBSSCNPZDataset(
        preprocess_root=cfg["data"]["preprocess_root"],
        data_root=cfg["data"].get("data_root", None),
        split=args.split,
        split_ratios=split_ratios,
        scene_filter=cfg["data"].get("scene_filter", None) or None,
        image_size=image_size,
        color_jitter=False,
        sample_list_path=split_files.get(args.split) or None,
    )

    model = VoxFormerStyleSSC(
        num_classes=num_classes,
        image_size=image_size,
        feat_dim=int(cfg["model"].get("feat_dim", 64)),
        hidden_dim=int(cfg["model"].get("hidden_dim", 64)),
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    indices = list(range(len(ds)))
    if args.random:
        random.seed(42); random.shuffle(indices)
    indices = indices[:args.max_samples]
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for n, idx in enumerate(indices):
            sample = ds[idx]
            batch = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else [v]) for k,v in sample.items()}
            out = model(batch)
            logits = torch.nan_to_num(out["sem_logits"], nan=0.0, posinf=20.0, neginf=-20.0)
            pred = logits.argmax(dim=1)[0].cpu().numpy()
            target = batch["target"][0].cpu().numpy()
            rgb = denorm_image(batch["image"][0])
            pred_td = topdown_semantic(pred, args.orientation)
            target_td = topdown_semantic(target, args.orientation)
            pred_h = topdown_height(pred, args.orientation)

            fig, axes = plt.subplots(2,2,figsize=(12,9))
            axes[0,0].imshow(rgb); axes[0,0].set_title("RGB input")
            axes[0,1].imshow(pred_td, cmap=CMAP, vmin=0, vmax=num_classes-1, interpolation="nearest"); axes[0,1].set_title("Prediction top-down")
            axes[1,0].imshow(target_td, cmap=CMAP, vmin=0, vmax=num_classes-1, interpolation="nearest"); axes[1,0].set_title("Target top-down")
            axes[1,1].imshow(pred_h, cmap="gray", vmin=0, vmax=1, interpolation="nearest"); axes[1,1].set_title("Prediction height / occupancy")
            for ax in axes.reshape(-1): ax.axis("off")
            fig.suptitle(f"{sample.get('scene','')} / {sample.get('timestamp','')}", fontsize=10)
            fig.tight_layout()
            out_path = out_dir / f"{n:04d}_{str(sample.get('scene','scene')).replace('/','_')}_{str(sample.get('timestamp',idx)).replace('/','_')}.png"
            fig.savefig(out_path, dpi=150); plt.close(fig)
            print(f"[{n+1}/{len(indices)}] saved {out_path}")

if __name__ == "__main__":
    main()
