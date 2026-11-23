"""PlantSeg dataset loading and augmentation.

`scripts/prepare_data.py` writes `index_{train,val,test}.csv` (columns: image, mask)
and `classes.json` (id -> disease name) into the data root. This module only reads those.
"""
from __future__ import annotations

import csv
import json
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = 255
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0

cv2.setNumThreads(0)


def read_index(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_classes(data_root: str) -> dict[int, str]:
    with open(os.path.join(data_root, "classes.json"), encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


def load_mask(path: str) -> np.ndarray:
    """Load a class-index mask. Palette ('P') and grayscale ('L') PNGs give indices directly."""
    m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m[..., 0]
    return m.astype(np.uint8)


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def normalize(img: np.ndarray) -> torch.Tensor:
    x = (img.astype(np.float32) - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def denormalize(x: torch.Tensor) -> np.ndarray:
    """CHW normalized tensor -> HWC uint8 RGB."""
    img = x.detach().cpu().float().numpy().transpose(1, 2, 0) * STD + MEAN
    return np.clip(img, 0, 255).astype(np.uint8)


def train_augment(img, mask, size, scale_range=(0.75, 1.5)):
    h, w = img.shape[:2]
    s = size / min(h, w) * random.uniform(*scale_range)
    nh, nw = max(int(h * s + 0.5), 1), max(int(w * s + 0.5), 1)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    ph, pw = max(size - nh, 0), max(size - nw, 0)
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=IGNORE_INDEX)
    y = random.randint(0, img.shape[0] - size)
    x = random.randint(0, img.shape[1] - size)
    img, mask = img[y:y + size, x:x + size], mask[y:y + size, x:x + size]
    if random.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if random.random() < 0.5:
        img, mask = img[::-1], mask[::-1]
    if random.random() < 0.8:  # brightness / contrast jitter
        alpha = random.uniform(0.8, 1.2)
        beta = random.uniform(-20, 20)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


class PlantSegDataset(Dataset):
    def __init__(self, data_root: str, split: str, size: int = 512, train: bool = False):
        self.root = data_root
        self.items = read_index(os.path.join(data_root, f"index_{split}.csv"))
        self.size = size
        self.train = train

    def __len__(self):
        return len(self.items)

    def _path(self, p):
        return p if os.path.isabs(p) else os.path.join(self.root, p)

    def __getitem__(self, i):
        it = self.items[i]
        img = load_image(self._path(it["image"]))
        mask = load_mask(self._path(it["mask"]))
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if self.train:
            img, mask = train_augment(img, mask, self.size)
        else:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return normalize(img), torch.from_numpy(mask.astype(np.int64))
