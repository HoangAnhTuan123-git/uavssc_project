import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext


def _bce_to_one_autocast_safe(prob, eps=1e-6):
    """
    MonoScene semantic/geometric scaling losses call BCE on probabilities
    produced by softmax. torch.cuda.amp autocast forbids binary_cross_entropy
    because BCE on probabilities is numerically fragile in fp16.

    Keep the original MonoScene loss behavior, but run this tiny scalar BCE
    computation in float32 outside autocast so mixed-precision training works.
    """
    ctx = torch.cuda.amp.autocast(enabled=False) if torch.cuda.is_available() else nullcontext()
    with ctx:
        prob32 = prob.float().clamp(min=eps, max=1.0 - eps)
        target32 = torch.ones_like(prob32)
        return F.binary_cross_entropy(prob32, target32)


def KL_sep(p, target):
    """
    KL divergence on nonzeros classes
    """
    nonzeros = target != 0
    nonzero_p = p[nonzeros]
    kl_term = F.kl_div(torch.log(nonzero_p), target[nonzeros], reduction="sum")
    return kl_term


def geo_scal_loss(pred, ssc_target):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, 0, :, :, :]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (
        _bce_to_one_autocast_safe(precision)
        + _bce_to_one_autocast_safe(recall)
        + _bce_to_one_autocast_safe(spec)
    )


def sem_scal_loss(pred, ssc_target):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != 255
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i, :, :, :]

        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = _bce_to_one_autocast_safe(precision)
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = _bce_to_one_autocast_safe(recall)
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = _bce_to_one_autocast_safe(specificity)
                loss_class += loss_specificity
            loss += loss_class
    return loss / count


def CE_ssc_loss(pred, target, class_weights):
    """
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    """
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="mean"
    )
    loss = criterion(pred, target.long())

    return loss
