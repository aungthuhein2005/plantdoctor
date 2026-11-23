"""Turn a segmentation mask into the PlantDoctor report:

    Disease prediction: Early blight
    Affected leaf area: 23.6%
    Healthy leaf area: 76.4%
    Severity: Moderate

PlantSeg only annotates *lesions*, not the whole leaf, so the leaf region is estimated with a
colour heuristic (green/yellow vegetation in HSV) united with the predicted lesion area, then
cleaned with morphology. Affected % = lesion pixels / leaf pixels.
"""
from __future__ import annotations

import cv2
import numpy as np

# Upper bounds (in % of leaf area) for each severity level; anything above the last is "Severe".
SEVERITY_LEVELS = [(1.0, "Healthy"), (10.0, "Mild"), (30.0, "Moderate"), (100.0, "Severe")]


def severity_from_pct(pct: float) -> str:
    for upper, name in SEVERITY_LEVELS:
        if pct < upper:
            return name
    return SEVERITY_LEVELS[-1][1]


def leaf_mask(img_rgb: np.ndarray, lesion: np.ndarray | None = None) -> np.ndarray:
    """Approximate leaf region (bool HxW)."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    veg = (h >= 20) & (h <= 95) & (s >= 35) & (v >= 35)  # yellow-green to green (OpenCV hue 0-180)
    if lesion is not None:
        veg |= lesion.astype(bool)
    m = veg.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # fill holes and drop tiny components
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.01 * m.size
    filled = np.zeros_like(m)
    keep = [c for c in contours if cv2.contourArea(c) >= min_area]
    cv2.drawContours(filled, keep, -1, 1, thickness=cv2.FILLED)
    out = filled.astype(bool)
    if lesion is not None:
        out |= lesion.astype(bool)
    return out


def area_stats(lesion: np.ndarray, leaf: np.ndarray) -> tuple[float, float]:
    leaf_px = int(leaf.sum())
    if leaf_px == 0:
        return 0.0, 0.0
    affected = 100.0 * float((lesion & leaf).sum()) / leaf_px
    return affected, 100.0 - affected


def build_report(img_rgb: np.ndarray, pred: np.ndarray, probs: np.ndarray | None,
                 id2label: dict[int, str]) -> dict:
    """pred: HxW class ids (0 = background). probs: CxHxW softmax (optional, for confidence)."""
    lesion = pred > 0
    leaf = leaf_mask(img_rgb, lesion)
    affected, healthy = area_stats(lesion, leaf)
    severity = severity_from_pct(affected)
    if lesion.sum() == 0 or severity == "Healthy":
        disease, conf = "Healthy", None
        if lesion.sum() > 0:
            ids, counts = np.unique(pred[lesion], return_counts=True)
            disease = f"Healthy (trace of {id2label.get(int(ids[counts.argmax()]), '?')})"
    else:
        ids, counts = np.unique(pred[lesion], return_counts=True)
        cid = int(ids[counts.argmax()])
        disease = id2label.get(cid, str(cid))
        conf = float(probs[cid][pred == cid].mean()) if probs is not None else None
    return {
        "disease": disease,
        "confidence": conf,
        "affected_pct": round(affected, 1),
        "healthy_pct": round(healthy, 1),
        "severity": severity,
        "lesion_mask": lesion,
        "leaf_mask": leaf,
    }


def format_report(r: dict) -> str:
    conf = f" (confidence {r['confidence']:.2f})" if r.get("confidence") is not None else ""
    return (
        f"Disease prediction: {r['disease']}{conf}\n"
        f"Affected leaf area: {r['affected_pct']:.1f}%\n"
        f"Healthy leaf area: {r['healthy_pct']:.1f}%\n"
        f"Segmentation mask: Highlighted diseased regions\n"
        f"Severity: {r['severity']}"
    )


def overlay(img_rgb: np.ndarray, lesion: np.ndarray, leaf: np.ndarray | None = None) -> np.ndarray:
    out = img_rgb.copy()
    red = np.zeros_like(out)
    red[..., 0] = 255
    out[lesion] = (0.45 * out[lesion] + 0.55 * red[lesion]).astype(np.uint8)
    if leaf is not None:
        contours, _ = cv2.findContours(leaf.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    return out
