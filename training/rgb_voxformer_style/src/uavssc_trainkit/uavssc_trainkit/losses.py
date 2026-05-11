from __future__ import annotations

import torch
import torch.nn.functional as F


def finite_or_zero_loss(loss, ref):
    if not torch.is_tensor(loss):
        return ref.sum() * 0.0
    if not torch.isfinite(loss).all():
        return ref.sum() * 0.0
    return loss


def semantic_occupancy_loss(
    sem_logits: torch.Tensor,
    occ_logits: torch.Tensor | None,
    target: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    occ_weight: float = 0.25,
):
    """Stable semantic + occupancy loss for sparse UAVScenes SSC crops."""
    sem_logits = torch.nan_to_num(sem_logits, nan=0.0, posinf=20.0, neginf=-20.0)
    if occ_logits is not None:
        occ_logits = torch.nan_to_num(occ_logits, nan=0.0, posinf=20.0, neginf=-20.0)

    valid = target != 255
    if not valid.any():
        zero = sem_logits.sum() * 0.0
        return {"loss_sem": zero, "loss_occ": zero, "loss": zero}

    ce = F.cross_entropy(sem_logits, target.long(), weight=class_weights, ignore_index=255)
    ce = finite_or_zero_loss(ce, sem_logits)
    loss = ce
    out = {"loss_sem": ce}

    if occ_logits is not None:
        target_occ = (target > 0).float()
        occ = F.binary_cross_entropy_with_logits(occ_logits.squeeze(1)[valid], target_occ[valid])
        occ = finite_or_zero_loss(occ, occ_logits)
        loss = loss + float(occ_weight) * occ
        out["loss_occ"] = occ
    else:
        out["loss_occ"] = sem_logits.sum() * 0.0

    out["loss"] = finite_or_zero_loss(loss, sem_logits)
    return out
