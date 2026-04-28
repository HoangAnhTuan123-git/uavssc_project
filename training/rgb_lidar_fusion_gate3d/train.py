from pathlib import Path
import argparse
import sys
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from uavssc_trainkit.uavssc_trainkit.data import FusionSSCNPZDataset
from uavssc_trainkit.uavssc_trainkit.models_fusion import FusionGate3DSSC
from uavssc_trainkit.uavssc_trainkit.trainer import train_loop
from uavssc_trainkit.uavssc_trainkit.utils import load_yaml, seed_everything, infer_num_classes, compute_log_class_weights

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))

    split_ratios = tuple(cfg["data"].get("split_ratios", [0.7, 0.15, 0.15]))
    scene_filter = cfg["data"].get("scene_filter", None) or None
    split_files = cfg["data"].get("split_files", {}) or {}
    image_size = tuple(cfg["data"].get("image_size", [640, 640]))

    train_ds = FusionSSCNPZDataset(cfg["data"]["preprocess_root"], cfg["data"].get("data_root", None), "train", split_ratios=split_ratios, scene_filter=scene_filter, image_size=image_size, color_jitter=True)
    val_ds = FusionSSCNPZDataset(cfg["data"]["preprocess_root"], cfg["data"].get("data_root", None), "val", split_ratios=split_ratios, scene_filter=scene_filter, image_size=image_size, color_jitter=False)

    if bool(cfg["model"].get("infer_num_classes_from_data", True)):
        cfg["model"]["num_classes"] = int(infer_num_classes(train_ds.get_sample_paths(), fallback=int(cfg["model"].get("num_classes", 26))))
    num_classes = int(cfg["model"]["num_classes"])

    model = FusionGate3DSSC(
        num_classes=num_classes,
        image_size=image_size,
        rgb_dim=int(cfg["model"].get("rgb_dim", 96)),
        lidar_hidden=int(cfg["model"].get("lidar_hidden", 48)),
        fuse_hidden=int(cfg["model"].get("fuse_hidden", 64)),
    )
    class_weights = compute_log_class_weights(train_ds.get_sample_paths(), num_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best = train_loop(model, train_ds, val_ds, cfg, cfg["output_dir"], device, class_weights)
    print("Best checkpoint:", best)

if __name__ == "__main__":
    main()
