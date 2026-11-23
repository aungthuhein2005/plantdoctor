#!/usr/bin/env python
"""Run the PlantDoctor report on an image.

    python scripts/predict.py --ckpt outputs/segformer_b2/best.pt --image leaf.jpg --out leaf_overlay.png
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plantdoctor.analysis import build_report, format_report, overlay  # noqa: E402
from plantdoctor.data import load_image, normalize  # noqa: E402
from plantdoctor.models import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=None, help="save overlay PNG here")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg, id2label = ck["cfg"], {int(k): v for k, v in ck["id2label"].items()}
    model = build_model(cfg["model"], len(id2label), pretrained=False, id2label=id2label)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()

    img = load_image(a.image)
    long_side = max(img.shape[:2])
    if long_side > 1024:  # keep the report resolution (and memory) bounded
        s = 1024 / long_side
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    size = cfg.get("img_size", 512)
    x = normalize(cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)).unsqueeze(0).to(dev)
    with torch.no_grad():
        logits = model(x)
        logits = F.interpolate(logits, size=img.shape[:2], mode="bilinear", align_corners=False)
        probs = logits.softmax(1)[0]
    pred = probs.argmax(0).cpu().numpy()
    r = build_report(img, pred, probs.cpu().numpy(), id2label)
    print(format_report(r))
    if a.out:
        cv2.imwrite(a.out, cv2.cvtColor(overlay(img, r["lesion_mask"], r["leaf_mask"]), cv2.COLOR_RGB2BGR))
        print(f"overlay saved to {a.out}")


if __name__ == "__main__":
    main()
