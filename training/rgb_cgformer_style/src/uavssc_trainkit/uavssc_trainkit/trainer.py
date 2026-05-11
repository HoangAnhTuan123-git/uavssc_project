from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import semantic_occupancy_loss, finite_or_zero_loss
from .metrics import evaluate_ssc
from .utils import move_to_device, save_json


def make_loader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )


def _ref_tensor_from_batch(batch):
    if isinstance(batch, dict):
        for key in ("target", "image", "lidar_dense"):
            v = batch.get(key)
            if torch.is_tensor(v):
                return v
        for v in batch.values():
            if torch.is_tensor(v):
                return v
    return torch.tensor(0.0)


def train_loop(model, train_ds, val_ds, cfg: dict, out_dir: str | Path, device: str, class_weights=None):
    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(train_ds, cfg["train"]["batch_size"], cfg["train"]["num_workers"], True)
    val_loader = make_loader(val_ds, 1, cfg["train"]["num_workers"], False)

    model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(cfg["train"]["epochs"]), 1))
    scaler = GradScaler(enabled=(device.startswith("cuda") and bool(cfg["train"].get("amp", True))))

    if class_weights is not None:
        class_weights = class_weights.to(device)

    best = -1.0
    history = []
    history_csv = out_dir / "history.csv"
    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_sc_iou", "val_ssc_miou"])
        for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
            model.train()
            running_loss = 0.0
            valid_steps = 0
            pbar = tqdm(train_loader, desc=f"train epoch {epoch}", leave=False)
            for batch in pbar:
                batch = move_to_device(batch, device)
                ref = _ref_tensor_from_batch(batch)
                opt.zero_grad(set_to_none=True)
                with autocast(enabled=(device.startswith("cuda") and bool(cfg["train"].get("amp", True)))):
                    out = model(batch)
                    sem_logits = torch.nan_to_num(out["sem_logits"], nan=0.0, posinf=20.0, neginf=-20.0)
                    occ_logits = out.get("occ_logits", None)
                    if occ_logits is not None:
                        occ_logits = torch.nan_to_num(occ_logits, nan=0.0, posinf=20.0, neginf=-20.0)
                    losses = semantic_occupancy_loss(
                        sem_logits,
                        occ_logits,
                        batch["target"],
                        class_weights=class_weights,
                        occ_weight=float(cfg["train"].get("occ_loss_weight", 0.25)),
                    )
                    loss = finite_or_zero_loss(losses["loss"], ref)

                scaler.scale(loss).backward()
                if float(cfg["train"].get("grad_clip", 0.0)) > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
                scaler.step(opt)
                scaler.update()

                loss_value = float(loss.detach().item())
                if loss_value == loss_value and abs(loss_value) != float("inf"):
                    running_loss += loss_value
                    valid_steps += 1
                pbar.set_postfix(loss=f"{loss_value:.4f}")

            scheduler.step()
            val_metrics = evaluate_ssc(model, val_loader, device, int(cfg["model"]["num_classes"]))
            train_loss = running_loss / max(valid_steps, 1)
            writer.writerow([epoch, train_loss, val_metrics["sc_IoU"], val_metrics["ssc_mIoU"]])
            f.flush()
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": cfg,
                "val_metrics": val_metrics,
            }
            torch.save(ckpt, ckpt_dir / "last.pt")
            if val_metrics["ssc_mIoU"] > best:
                best = val_metrics["ssc_mIoU"]
                torch.save(ckpt, ckpt_dir / "best.pt")
            row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
            history.append(row)
            save_json(out_dir / "history.json", history)
    return str(ckpt_dir / "best.pt")
