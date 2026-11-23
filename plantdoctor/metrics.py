"""Confusion-matrix based segmentation metrics."""
from __future__ import annotations

import torch

from .data import IGNORE_INDEX


class ConfusionMatrix:
    def __init__(self, num_classes: int, device="cuda"):
        self.n = num_classes
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        k = target != IGNORE_INDEX
        idx = target[k] * self.n + pred[k]
        self.mat += torch.bincount(idx, minlength=self.n ** 2).reshape(self.n, self.n)

    def compute(self) -> dict:
        m = self.mat.double()
        tp = m.diag()
        fp = m.sum(0) - tp
        fn = m.sum(1) - tp
        gt = m.sum(1)
        union = tp + fp + fn
        present = gt > 0  # classes that appear in ground truth
        iou = tp / union.clamp(min=1)
        dice = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
        prec = tp / (tp + fp).clamp(min=1)
        rec = tp / gt.clamp(min=1)
        fg = present.clone()
        fg[0] = False  # exclude background

        # Binary lesion-vs-background (all disease classes merged)
        btp = m[1:, 1:].sum()
        bfp = m[0, 1:].sum()
        bfn = m[1:, 0].sum()

        def mean(v, mask):
            return v[mask].mean().item() if mask.any() else 0.0

        return {
            "mIoU": mean(iou, present),
            "mIoU_disease": mean(iou, fg),
            "mDice": mean(dice, present),
            "mDice_disease": mean(dice, fg),
            "mPrecision": mean(prec, present),
            "mRecall": mean(rec, present),
            "pixel_acc": (tp.sum() / m.sum().clamp(min=1)).item(),
            "binary_IoU": (btp / (btp + bfp + bfn).clamp(min=1)).item(),
            "binary_Dice": (2 * btp / (2 * btp + bfp + bfn).clamp(min=1)).item(),
            "binary_precision": (btp / (btp + bfp).clamp(min=1)).item(),
            "binary_recall": (btp / (btp + bfn).clamp(min=1)).item(),
            "per_class_IoU": {int(i): iou[i].item() for i in range(self.n) if present[i]},
        }
