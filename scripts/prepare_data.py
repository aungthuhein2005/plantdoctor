#!/usr/bin/env python
"""Index the Kaggle PlantSeg download into train/val/test CSVs + classes.json.

    python scripts/prepare_data.py --raw data/raw --out data/plantseg

The script auto-discovers the layout (images/annotations folders, split folders, metadata CSV),
so it works whether the archive extracts to plantseg/... or plantsegv2/... . Mask pixel values
are class ids (0 = background). Disease names are recovered from the metadata CSV by majority vote
of mask ids per disease. Output paths in the CSVs are absolute.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import random
import re
from multiprocessing import Pool

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train": "train", "training": "train", "val": "val", "valid": "val",
          "validation": "val", "test": "test", "testing": "test"}


def split_of(path):
    for part in reversed(os.path.normpath(path).lower().split(os.sep)):
        if part in SPLITS:
            return SPLITS[part]
    return None


def kind_of(path):
    parts = os.path.normpath(path).lower().split(os.sep)
    for part in reversed(parts[:-1]):
        if re.search(r"annotation|mask|label|gt", part):
            return "mask"
        if re.search(r"image|img|jpeg", part):
            return "image"
    return None


def mask_values(path):
    m = np.array(Image.open(path))
    if m.ndim == 3:
        if not (np.array_equal(m[..., 0], m[..., 1]) and np.array_equal(m[..., 0], m[..., 2])):
            return path, None  # true RGB colour mask
        m = m[..., 0]
    return path, np.unique(m).tolist()


def find_metadata(raw):
    best = None
    for p in glob.glob(os.path.join(raw, "**", "*.csv"), recursive=True):
        with open(p, newline="", encoding="utf-8", errors="ignore") as f:
            r = csv.DictReader(f)
            cols = r.fieldnames or []
            if any("disease" in c.lower() for c in cols):
                best = p
                break
    return best


def load_metadata(p):
    """Return {image_stem: disease_name} from the metadata CSV."""
    with open(p, newline="", encoding="utf-8", errors="ignore") as f:
        rows = list(csv.DictReader(f))
    cols = rows[0].keys()
    dcol = next(c for c in cols if "disease" in c.lower())
    pcol = next((c for c in cols if c.lower().strip() in ("plant", "host", "plant name", "crop")), None)
    # column whose values look like file names
    ncol = None
    for c in cols:
        v = str(rows[0][c])
        if re.search(r"\.(jpe?g|png)$", v, re.I):
            ncol = c
            break
    if ncol is None:
        ncol = next((c for c in cols if c.lower() in ("name", "image", "filename", "file", "image name")), None)
    if ncol is None:
        return {}
    out = {}
    for r in rows:
        dis = r[dcol].strip()
        if pcol and r[pcol].strip() and r[pcol].strip().lower() not in dis.lower():
            dis = f"{r[pcol].strip()} {dis}"
        out[os.path.splitext(os.path.basename(r[ncol].strip()))[0]] = dis
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/plantseg")
    ap.add_argument("--val-frac", type=float, default=0.1, help="used only if no val split exists")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    raw = os.path.abspath(a.raw)
    os.makedirs(a.out, exist_ok=True)

    images, masks = {}, {}
    for p in glob.glob(os.path.join(raw, "**", "*"), recursive=True):
        if os.path.splitext(p)[1].lower() not in IMG_EXT:
            continue
        k = kind_of(os.path.relpath(p, raw))
        stem = os.path.splitext(os.path.basename(p))[0]
        if k == "mask":
            masks[stem] = p
        elif k == "image":
            images[stem] = p
    pairs = [(s, images[s], masks[s]) for s in sorted(images) if s in masks]
    print(f"found {len(images)} images, {len(masks)} masks, {len(pairs)} pairs")
    if not pairs:
        raise SystemExit("no image/mask pairs found - inspect the raw folder layout")

    with Pool(a.workers) as pool:
        vals = dict(pool.map(mask_values, [m for _, _, m in pairs], chunksize=64))
    bad = [p for p, v in vals.items() if v is None]
    if bad:
        raise SystemExit(f"{len(bad)} masks are RGB colour masks (e.g. {bad[0]}); need a palette->id map")

    splits = collections.defaultdict(list)
    for s, ip, mp in pairs:
        splits[split_of(os.path.relpath(mp, raw)) or split_of(os.path.relpath(ip, raw)) or "train"].append((s, ip, mp))
    if not splits.get("val"):
        random.seed(0)
        tr = splits["train"]
        random.shuffle(tr)
        nv = int(len(tr) * a.val_frac)
        splits["val"], splits["train"] = tr[:nv], tr[nv:]
        print(f"no val split found: carved {nv} images from train")
    if not splits.get("test"):
        raise SystemExit("no test split found")

    max_id = max(max(v) for v in vals.values())
    id_counter = collections.Counter(i for v in vals.values() for i in v)
    print(f"mask ids: 0..{max_id}, {len(id_counter)} distinct values")

    names = {0: "background"}
    meta_p = find_metadata(raw)
    if meta_p:
        meta = load_metadata(meta_p)
        print(f"metadata: {meta_p} ({len(meta)} rows)")
        votes = collections.defaultdict(collections.Counter)
        for s, _, mp in pairs:
            if s in meta:
                for i in vals[mp]:
                    if i != 0 and i != 255:
                        votes[i][meta[s]] += 1
        for i, c in votes.items():
            names[i] = c.most_common(1)[0][0]
    for i in range(1, max_id + 1):
        if i != 255:
            names.setdefault(i, f"class_{i}")
    names.pop(255, None)
    with open(os.path.join(a.out, "classes.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in sorted(names.items())}, f, indent=2, ensure_ascii=False)

    for sp in ("train", "val", "test"):
        with open(os.path.join(a.out, f"index_{sp}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "mask"])
            for _, ip, mp in splits[sp]:
                w.writerow([ip, mp])
        print(f"{sp}: {len(splits[sp])}")
    print(f"classes: {len(names)} (incl. background) -> {a.out}/classes.json")
    unnamed = [i for i, n in names.items() if n.startswith("class_")]
    if unnamed:
        print(f"WARNING: {len(unnamed)} ids have no disease name from metadata: {unnamed[:10]}")


if __name__ == "__main__":
    main()
