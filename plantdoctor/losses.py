import torch
import torch.nn.functional as F

from .data import IGNORE_INDEX


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft multi-class Dice over the classes present in the batch."""
    num_classes = logits.shape[1]
    valid = (target != IGNORE_INDEX).unsqueeze(1)
    t = target.masked_fill(target == IGNORE_INDEX, 0)
    prob = logits.float().softmax(1) * valid
    onehot = F.one_hot(t, num_classes).permute(0, 3, 1, 2).float() * valid
    dims = (0, 2, 3)
    inter = (prob * onehot).sum(dims)
    card = prob.sum(dims) + onehot.sum(dims)
    present = onehot.sum(dims) > 0
    dice = (2 * inter + eps) / (card + eps)
    return 1 - dice[present].mean()


def seg_loss(logits, target, dice_weight: float = 1.0):
    ce = F.cross_entropy(logits.float(), target, ignore_index=IGNORE_INDEX)
    if dice_weight <= 0:
        return ce
    return ce + dice_weight * dice_loss(logits, target)
