from __future__ import annotations

import torch

def fast_hist(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    if pred.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.float64, device=pred.device)
    k = target * num_classes + pred
    binc = torch.bincount(k, minlength=num_classes * num_classes).double()
    return binc.view(num_classes, num_classes)

def iou_from_hist(hist: torch.Tensor) -> torch.Tensor:
    eps = 1e-9
    inter = torch.diag(hist)
    union = hist.sum(dim=1) + hist.sum(dim=0) - inter
    return inter / (union + eps)

def summarize_hist(hist: torch.Tensor) -> dict:
    iou = iou_from_hist(hist)
    ssc_miou = float(iou[1:].mean().item()) if iou.numel() > 1 else 0.0
    acc = float(torch.diag(hist).sum().item() / max(hist.sum().item(), 1.0))
    return {
        "ssc_mIoU": ssc_miou,
        "acc": acc,
        "class_iou": iou.detach().cpu().tolist(),
    }

@torch.no_grad()
def evaluate_ssc(model, dataloader, device, num_classes: int):
    model.eval()
    hist = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)
    occ_inter = 0.0
    occ_union = 0.0
    for batch in dataloader:
        for k, v in list(batch.items()):
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)
        out = model(batch)
        pred = out["sem_logits"].argmax(dim=1)
        target = batch["target"]
        hist += fast_hist(pred, target, num_classes)
        valid = target != 255
        pred_occ = (pred > 0) & valid
        target_occ = (target > 0) & valid
        occ_inter += (pred_occ & target_occ).sum().item()
        occ_union += (pred_occ | target_occ).sum().item()

    metrics = summarize_hist(hist)
    metrics["sc_IoU"] = float(occ_inter / max(occ_union, 1.0))
    return metrics
