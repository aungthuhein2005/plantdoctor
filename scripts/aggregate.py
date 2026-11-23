#!/usr/bin/env python
"""Collect outputs/*/results.json into a benchmark table, plots and a winner.

    python scripts/aggregate.py --out-dir outputs --report results
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

ORDER = ["unet_resnet34", "unet_resnet50", "segformer_b0", "segformer_b2", "segformer_b4"]

COLUMNS = [
    ("Model", lambda r: r["name"]),
    ("Params (M)", lambda r: f"{r['params_M']:.1f}"),
    ("mIoU", lambda r: f"{100*r['test']['mIoU']:.2f}"),
    ("mIoU (disease)", lambda r: f"{100*r['test']['mIoU_disease']:.2f}"),
    ("mDice", lambda r: f"{100*r['test']['mDice']:.2f}"),
    ("Precision", lambda r: f"{100*r['test']['mPrecision']:.2f}"),
    ("Recall", lambda r: f"{100*r['test']['mRecall']:.2f}"),
    ("Lesion IoU", lambda r: f"{100*r['test']['binary_IoU']:.2f}"),
    ("Lesion Dice", lambda r: f"{100*r['test']['binary_Dice']:.2f}"),
    ("Disease acc", lambda r: f"{100*r['test']['image_disease_acc']:.2f}"),
    ("Area MAE (pp)", lambda r: f"{r['test'].get('affected_area_MAE_pct', float('nan')):.2f}"),
    ("Severity acc", lambda r: f"{100*r['test'].get('severity_acc', float('nan')):.2f}"),
    ("Train mem (GB)", lambda r: f"{r['peak_train_mem_GB']:.1f}"),
    ("Train time (h)", lambda r: f"{r['train_time_h']:.2f}"),
    ("Latency fp32 (ms)", lambda r: f"{r['latency_fp32']['median_ms']:.1f}"),
    ("Latency bf16 (ms)", lambda r: f"{r['latency_bf16']['median_ms']:.1f}"),
    ("Infer mem (MB)", lambda r: f"{r['latency_fp32']['peak_mem_MB']:.0f}"),
]


def pick_winner(rows, tol=0.005):
    """Highest test mIoU; among models within `tol` of the best, prefer the fastest (bf16 latency)."""
    best = max(r["test"]["mIoU"] for r in rows)
    close = [r for r in rows if best - r["test"]["mIoU"] <= tol]
    return min(close, key=lambda r: r["latency_bf16"]["median_ms"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--report", default="results")
    a = ap.parse_args()
    rows = []
    for p in glob.glob(os.path.join(a.out_dir, "*", "results.json")):
        if "_smoke" in p:
            continue
        rows.append(json.load(open(p)))
    if not rows:
        raise SystemExit("no results.json found")
    rows.sort(key=lambda r: ORDER.index(r["name"]) if r["name"] in ORDER else 99)
    os.makedirs(a.report, exist_ok=True)

    with open(os.path.join(a.report, "benchmark.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([c for c, _ in COLUMNS])
        for r in rows:
            w.writerow([fn(r) for _, fn in COLUMNS])

    win = pick_winner(rows)
    md = ["| " + " | ".join(c for c, _ in COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for r in rows:
        cells = [fn(r) for _, fn in COLUMNS]
        if r is win:
            cells = [f"**{c}**" for c in cells]
        md.append("| " + " | ".join(cells) + " |")
    md.append("")
    md.append(f"GPU: {rows[0]['gpu']}. Latency: batch 1, {rows[0]['img_size']}x{rows[0]['img_size']}, median of 100 runs.")
    md.append(f"\n**Winner: {win['name']}** (test mIoU {100*win['test']['mIoU']:.2f}, "
              f"{win['latency_bf16']['median_ms']:.1f} ms bf16, {win['params_M']:.1f}M params)")
    with open(os.path.join(a.report, "benchmark.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(a.report, "winner.json"), "w") as f:
        json.dump({"name": win["name"], "checkpoint": os.path.join(a.out_dir, win["name"], "best.pt")}, f, indent=2)
    print("\n".join(md))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for r in rows:
            x, y = r["latency_bf16"]["median_ms"], 100 * r["test"]["mIoU"]
            ax[0].scatter(x, y, s=20 + 6 * r["params_M"], alpha=0.6)
            ax[0].annotate(r["name"], (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax[0].set_xlabel("Inference latency, bf16 (ms)")
        ax[0].set_ylabel("Test mIoU (%)")
        ax[0].set_title("Accuracy vs speed (bubble = params)")
        ax[0].grid(alpha=0.3)
        names = [r["name"] for r in rows]
        ax[1].bar(names, [r["train_time_h"] for r in rows], color="#4C72B0")
        ax[1].set_ylabel("Training time (h)")
        ax2 = ax[1].twinx()
        ax2.plot(names, [r["peak_train_mem_GB"] for r in rows], "o-", color="#DD8452")
        ax2.set_ylabel("Peak training memory (GB)")
        ax[1].set_title("Training cost")
        ax[1].tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(os.path.join(a.report, "benchmark.png"), dpi=150)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for r in rows:
            log = list(csv.DictReader(open(os.path.join(a.out_dir, r["name"], "log.csv"))))
            pts = [(int(l["epoch"]) + 1, 100 * float(l["val_mIoU"])) for l in log if l["val_mIoU"] != "nan"]
            if pts:
                ax.plot(*zip(*pts), label=r["name"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val mIoU (%)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(a.report, "val_curves.png"), dpi=150)
    except ImportError:
        print("matplotlib not installed; skipped plots")


if __name__ == "__main__":
    main()
