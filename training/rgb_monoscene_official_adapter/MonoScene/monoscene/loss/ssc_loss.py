import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext


def _autocast_off():
    return torch.cuda.amp.autocast(enabled=False) if torch.cuda.is_available() else nullcontext()


def _zero_like_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.float().sum() * 0.0


def _safe_prob(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    with _autocast_off():
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=0.0)
        return x.clamp(min=eps, max=1.0 - eps)


def _bce_to_one_autocast_safe(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """BCE(prob, 1) in fp32 with NaN/inf protection.

    MonoScene's semantic/geometric scaling losses call BCE on scalar
    probabilities computed from softmax. Under AMP this is both unsafe and
    can assert if a rare UAV crop produces NaN/inf or a tiny value outside
    [0, 1].
    """
    with _autocast_off():
        prob32 = _safe_prob(prob, eps=eps)
        target32 = torch.ones_like(prob32)
        loss = F.binary_cross_entropy(prob32, target32)
        if not torch.isfinite(loss).all():
            return prob32.sum() * 0.0
        return loss


def _safe_div(num: torch.Tensor, den: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    den_val = torch.nan_to_num(den.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if den_val <= 0:
        return fallback.sum() * 0.0
    out = num.float() / den_val
    return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)


def KL_sep(p, target):
    nonzeros = target != 0
    if nonzeros.sum() == 0:
        return p.sum() * 0.0
    nonzero_p = _safe_prob(p[nonzeros])
    target_nonzero = torch.nan_to_num(target[nonzeros].float(), nan=0.0, posinf=1.0, neginf=0.0)
    target_nonzero = target_nonzero.clamp(min=0.0)
    kl_term = F.kl_div(torch.log(nonzero_p), target_nonzero, reduction="sum")
    if not torch.isfinite(kl_term).all():
        return p.sum() * 0.0
    return kl_term


def geo_scal_loss(pred, ssc_target):
    pred = torch.nan_to_num(pred, nan=0.0, posinf=20.0, neginf=-20.0)
    prob = F.softmax(pred, dim=1)
    empty_probs = prob[:, 0]
    nonempty_probs = 1.0 - empty_probs

    mask = ssc_target != 255
    if mask.sum() == 0:
        return _zero_like_logits(pred)

    nonempty_target = (ssc_target != 0)[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = _safe_div(intersection, nonempty_probs.sum(), pred)
    recall = _safe_div(intersection, nonempty_target.sum(), pred)
    spec = _safe_div(((1 - nonempty_target) * empty_probs).sum(), (1 - nonempty_target).sum(), pred)
    loss = _bce_to_one_autocast_safe(precision) + _bce_to_one_autocast_safe(recall) + _bce_to_one_autocast_safe(spec)
    return loss if torch.isfinite(loss).all() else _zero_like_logits(pred)


def sem_scal_loss(pred, ssc_target):
    pred = torch.nan_to_num(pred, nan=0.0, posinf=20.0, neginf=-20.0)
    prob = F.softmax(pred, dim=1)
    loss = pred.sum() * 0.0
    count = 0
    mask = ssc_target != 255
    if mask.sum() == 0:
        return _zero_like_logits(pred)

    n_classes = prob.shape[1]
    target = ssc_target[mask]
    for i in range(n_classes):
        p = prob[:, i][mask]
        completion_target = torch.zeros_like(target, dtype=torch.float32)
        completion_target[target == i] = 1.0
        if completion_target.sum() <= 0:
            continue
        count += 1
        nominator = torch.sum(p * completion_target)
        loss_class = pred.sum() * 0.0
        if torch.sum(p) > 0:
            precision = _safe_div(nominator, torch.sum(p), pred)
            loss_class = loss_class + _bce_to_one_autocast_safe(precision)
        if torch.sum(completion_target) > 0:
            recall = _safe_div(nominator, torch.sum(completion_target), pred)
            loss_class = loss_class + _bce_to_one_autocast_safe(recall)
        neg = 1.0 - completion_target
        if torch.sum(neg) > 0:
            specificity = _safe_div(torch.sum((1 - p) * neg), torch.sum(neg), pred)
            loss_class = loss_class + _bce_to_one_autocast_safe(specificity)
        loss = loss + loss_class
    if count == 0:
        return _zero_like_logits(pred)
    loss = loss / float(count)
    return loss if torch.isfinite(loss).all() else _zero_like_logits(pred)


def CE_ssc_loss(pred, target, class_weights):
    pred = torch.nan_to_num(pred, nan=0.0, posinf=20.0, neginf=-20.0)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, reduction="mean")
    loss = criterion(pred, target.long())
    if not torch.isfinite(loss).all():
        return pred.sum() * 0.0
    return loss
