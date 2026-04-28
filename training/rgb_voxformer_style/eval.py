from pathlib import Path
import argparse
import sys
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from torch.utils.data import DataLoader
from uavssc_trainkit.uavssc_trainkit.data import RGBSSCNPZDataset
from uavssc_trainkit.uavssc_trainkit.models_rgb import VoxFormerStyleSSC
from uavssc_trainkit.uavssc_trainkit.metrics import evaluate_ssc
from uavssc_trainkit.uavssc_trainkit.utils import load_yaml, infer_num_classes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--checkpoint", type=str, required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    split_ratios = tuple(cfg["data"].get("split_ratios", [0.7, 0.15, 0.15]))
    scene_filter = cfg["data"].get("scene_filter", None) or None
    split_files = cfg["data"].get("split_files", {}) or {}
    image_size = tuple(cfg["data"].get("image_size", [640, 640]))

    val_ds = RGBSSCNPZDataset(
        preprocess_root=cfg["data"]["preprocess_root"],
        data_root=cfg["data"].get("data_root", None),
        split="val",
        split_ratios=split_ratios,
        scene_filter=scene_filter,
        image_size=image_size,
        color_jitter=False,
        sample_list_path=split_files.get("test", split_files.get("val")),
    )
    if bool(cfg["model"].get("infer_num_classes_from_data", True)):
        cfg["model"]["num_classes"] = int(infer_num_classes(val_ds.get_sample_paths(), fallback=int(cfg["model"].get("num_classes", 26))))
    num_classes = int(cfg["model"]["num_classes"])

    model = VoxFormerStyleSSC(
        num_classes=num_classes,
        image_size=image_size,
        feat_dim=int(cfg["model"].get("feat_dim", 96)),
        hidden_dim=int(cfg["model"].get("hidden_dim", 64)),
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=int(cfg["train"].get("num_workers", 4)), pin_memory=True)
    metrics = evaluate_ssc(model, loader, device, num_classes)
    print(metrics)

if __name__ == "__main__":
    main()
