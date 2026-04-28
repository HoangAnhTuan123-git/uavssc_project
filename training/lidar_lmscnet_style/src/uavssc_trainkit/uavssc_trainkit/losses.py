from __future__ import annotations

import torch
import torch.nn.functional as F

def semantic_occupancy_loss(sem_logits: torch.Tensor, occ_logits: torch.Tensor | None, target: torch.Tensor, class_weights: torch.Tensor | None = None, occ_weight: float = 0.25):
    ce = F.cross_entropy(sem_logits, target.long(), weight=class_weights, ignore_index=255)
    loss = ce
    out = {"loss_sem": ce}
    if occ_logits is not None:
        valid = target != 255
        target_occ = (target > 0).float()
        if valid.any():
            occ = F.binary_cross_entropy_with_logits(occ_logits.squeeze(1)[valid], target_occ[valid])
            loss = loss + occ_weight * occ
            out["loss_occ"] = occ
        else:
            out["loss_occ"] = torch.tensor(0.0, device=sem_logits.device)
    out["loss"] = loss
    return out
