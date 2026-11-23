"""Evaluation loop, area/severity metrics and latency benchmark."""
from __future__ import annotations

import statistics
import time

import numpy as np
import torch

from .analysis import area_stats, leaf_mask, severity_from_pct
from .data import IGNORE_INDEX, denormalize
from .metrics import ConfusionMatrix


def _mode_nonzero(x: torch.Tensor) -> int:
    x = x[(x > 0) & (x != IGNORE_INDEX)]
    if x.numel() == 0:
        return 0
    return int(torch.bincount(x).argmax())


@torch.no_grad()
def evaluate(model, loader, num_classes, device="cuda", amp_dtype=torch.bfloat16,
             area_metrics=False, max_batches=None) -> dict:
    model.eval()
    cm = ConfusionMatrix(num_classes, device)
    img_correct = img_total = 0
    abs_err, sev_correct, sev_total = [], 0, 0
    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        pred = logits.argmax(1)
        cm.update(pred, y)
        for i in range(x.shape[0]):
            gt_cls = _mode_nonzero(y[i])
            if gt_cls > 0:  # image-level disease classification from segmentation
                img_total += 1
                img_correct += int(_mode_nonzero(pred[i]) == gt_cls)
            if area_metrics:
                gt_les = ((y[i] > 0) & (y[i] != IGNORE_INDEX)).cpu().numpy()
                pr_les = (pred[i] > 0).cpu().numpy()
                leaf = leaf_mask(denormalize(x[i]), gt_les | pr_les)
                a_gt, _ = area_stats(gt_les, leaf)
                a_pr, _ = area_stats(pr_les, leaf)
                abs_err.append(abs(a_gt - a_pr))
                sev_total += 1
                sev_correct += int(severity_from_pct(a_gt) == severity_from_pct(a_pr))
    res = cm.compute()
    res["image_disease_acc"] = img_correct / max(img_total, 1)
    if area_metrics:
        res["affected_area_MAE_pct"] = float(np.mean(abs_err)) if abs_err else 0.0
        res["severity_acc"] = sev_correct / max(sev_total, 1)
    return res


@torch.no_grad()
def benchmark_latency(model, size=512, device="cuda", amp_dtype=None, warmup=20, iters=100,
                      batch_size=1) -> dict:
    model.eval()
    x = torch.randn(batch_size, 3, size, size, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for i in range(warmup + iters):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            model(x)
        torch.cuda.synchronize(device)
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "median_ms": statistics.median(times),
        "p95_ms": times[int(0.95 * (len(times) - 1))],
        "fps": 1000.0 * batch_size / statistics.mean(times),
        "peak_mem_MB": torch.cuda.max_memory_allocated(device) / 2 ** 20,
    }
