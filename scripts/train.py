#!/usr/bin/env python
"""Train one benchmark model, then evaluate on test and benchmark latency.

    python scripts/train.py --config configs/segformer_b2.yaml --data-root data/plantseg

Writes to <out-dir>/<name>/: best.pt, last.pt, log.csv, results.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plantdoctor.data import PlantSegDataset, load_classes  # noqa: E402
from plantdoctor.evaluate import benchmark_latency, evaluate  # noqa: E402
from plantdoctor.losses import seg_loss  # noqa: E402
from plantdoctor.models import build_model, count_params  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data-root", default="data/plantseg")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    p.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline")
    p.add_argument("--resume", action="store_true", help="resume from last.pt if present")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    a = parse()
    cfg = yaml.safe_load(open(a.config))
    if a.epochs:
        cfg["epochs"] = a.epochs
    if a.batch_size:
        cfg["batch_size"] = a.batch_size
    name = cfg["name"] + ("_smoke" if a.smoke else "")
    out = os.path.join(a.out_dir, name)
    os.makedirs(out, exist_ok=True)
    seed_all(cfg.get("seed", 42))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")

    id2label = load_classes(a.data_root)
    nc = len(id2label)
    size = cfg.get("img_size", 512)
    train_ds = PlantSegDataset(a.data_root, "train", size, train=True)
    val_ds = PlantSegDataset(a.data_root, "val", size)
    test_ds = PlantSegDataset(a.data_root, "test", size)
    if a.smoke:
        train_ds.items = train_ds.items[:64]
        val_ds.items = val_ds.items[:32]
        test_ds.items = test_ds.items[:32]
        cfg["epochs"] = 2
    bs = cfg["batch_size"]
    dl_kw = dict(num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    train_dl = DataLoader(train_ds, bs, shuffle=True, drop_last=True, **dl_kw)
    val_dl = DataLoader(val_ds, bs, shuffle=False, **dl_kw)
    test_dl = DataLoader(test_ds, bs, shuffle=False, **dl_kw)
    print(f"[{name}] classes={nc} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

    model = build_model(cfg["model"], nc, pretrained=True, id2label=id2label).to(dev)
    n_params = count_params(model)
    lr = cfg["lr"]
    opt = torch.optim.AdamW(
        [
            {"params": model.encoder_params(), "lr": lr},
            {"params": model.head_params(), "lr": lr * cfg.get("head_lr_mult", 1.0)},
        ],
        lr=lr, weight_decay=cfg.get("weight_decay", 0.01),
    )
    total_it = cfg["epochs"] * len(train_dl)
    warm = min(cfg.get("warmup_iters", 1500), total_it // 10)

    def lr_lambda(it):
        if it < warm:
            return 1e-6 + (1 - 1e-6) * it / max(warm, 1)
        return (1 - (it - warm) / max(total_it - warm, 1)) ** 1.0

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    amp = torch.bfloat16 if cfg.get("amp", True) else None

    start_ep, best, train_time = 0, -1.0, 0.0
    last_path, best_path = os.path.join(out, "last.pt"), os.path.join(out, "best.pt")
    if a.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_ep, best, train_time = ck["epoch"] + 1, ck["best"], ck["train_time"]
        print(f"resumed at epoch {start_ep}", flush=True)

    log_path = os.path.join(out, "log.csv")
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if start_ep == 0:
        log.writerow(["epoch", "train_loss", "val_mIoU", "val_mDice", "lr", "epoch_time_s"])

    torch.cuda.reset_peak_memory_stats(dev)
    eval_every = cfg.get("eval_every", 2)
    for ep in range(start_ep, cfg["epochs"]):
        model.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for x, y in train_dl:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                logits = model(x)
            loss = seg_loss(logits, y, cfg.get("dice_weight", 1.0))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            opt.step()
            sched.step()
            if not math.isfinite(loss.item()):
                raise RuntimeError("loss is NaN/inf")
            tot += loss.item()
            n += 1
        torch.cuda.synchronize()
        ep_time = time.time() - t0
        train_time += ep_time
        val_miou = val_dice = float("nan")
        if (ep + 1) % eval_every == 0 or ep + 1 == cfg["epochs"]:
            r = evaluate(model, val_dl, nc, dev, amp)
            val_miou, val_dice = r["mIoU"], r["mDice"]
            if val_miou > best:
                best = val_miou
                torch.save({"model": model.state_dict(), "cfg": cfg, "id2label": id2label,
                            "epoch": ep, "val_mIoU": best}, best_path)
        log.writerow([ep, tot / max(n, 1), val_miou, val_dice, sched.get_last_lr()[0], round(ep_time, 1)])
        log_f.flush()
        print(f"[{name}] ep {ep+1}/{cfg['epochs']} loss {tot/max(n,1):.4f} val mIoU {val_miou:.4f} "
              f"best {best:.4f} ({ep_time:.0f}s)", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best": best, "train_time": train_time}, last_path)
    peak_train_mem = torch.cuda.max_memory_allocated(dev) / 2 ** 20
    log_f.close()

    # ---- final test evaluation with best checkpoint ----
    ck = torch.load(best_path, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    test = evaluate(model, test_dl, nc, dev, amp, area_metrics=True)
    lat_fp32 = benchmark_latency(model, size, dev, None)
    lat_bf16 = benchmark_latency(model, size, dev, torch.bfloat16)

    results = {
        "name": cfg["name"],
        "model": cfg["model"],
        "params_M": n_params / 1e6,
        "epochs": cfg["epochs"],
        "batch_size": bs,
        "img_size": size,
        "best_epoch": ck["epoch"],
        "val_mIoU": ck["val_mIoU"],
        "train_time_h": train_time / 3600,
        "peak_train_mem_GB": peak_train_mem / 1024,
        "latency_fp32": lat_fp32,
        "latency_bf16": lat_bf16,
        "gpu": torch.cuda.get_device_name(dev),
        "test": {k: v for k, v in test.items() if k != "per_class_IoU"},
        "test_per_class_IoU": {id2label[k]: v for k, v in test["per_class_IoU"].items()},
    }
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != "test_per_class_IoU"}, indent=2))


if __name__ == "__main__":
    main()
