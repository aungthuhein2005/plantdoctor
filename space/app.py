"""PlantDoctor demo: upload a leaf photo -> disease, affected/healthy leaf area, severity, highlighted mask."""
from __future__ import annotations

import os

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from plantdoctor.analysis import SEVERITY_LEVELS, build_report, overlay

try:  # ZeroGPU on Hugging Face Spaces; falls back to plain CPU/GPU elsewhere
    import spaces
    gpu = spaces.GPU(duration=20)
    DEVICE = "cuda"
except ImportError:
    def gpu(fn):
        return fn
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_REPO = os.environ.get("MODEL_REPO", "aungthuhein-dev/plantdoctor-segformer-b4")
MAX_SIDE = 1024  # report resolution cap (keeps time and memory bounded)

torch.set_grad_enabled(False)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_REPO).eval().to(DEVICE)
processor = SegformerImageProcessor(size={"height": 512, "width": 512})
id2label = {int(k): v for k, v in model.config.id2label.items()}


@gpu
def segment(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Return per-pixel class probabilities (C, H, W) at `size` (height, width)."""
    inputs = processor(image, return_tensors="pt").to(DEVICE)
    logits = model(**inputs).logits
    logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
    return logits.softmax(1)[0].float().cpu().numpy()


def _bands_text() -> str:
    lo, parts = 0.0, []
    for hi, name in SEVERITY_LEVELS:
        parts.append(f"{name} {lo:g}-{hi:g}%")
        lo = hi
    return " · ".join(parts)


def diagnose(image: Image.Image | None):
    if image is None:
        return None, "Upload a leaf photo to get a diagnosis.", None
    image = image.convert("RGB")
    if max(image.size) > MAX_SIDE:
        s = MAX_SIDE / max(image.size)
        image = image.resize((round(image.width * s), round(image.height * s)), Image.BILINEAR)
    arr = np.array(image)

    probs = segment(image, arr.shape[:2])
    pred = probs.argmax(0)

    r = build_report(arr, pred, probs, id2label)
    vis = Image.fromarray(overlay(arr, r["lesion_mask"], r["leaf_mask"]))

    conf = f" (confidence {r['confidence']:.2f})" if r.get("confidence") is not None else ""
    report = (
        f"### Disease prediction: {r['disease']}{conf}\n\n"
        f"| | |\n|---|---|\n"
        f"| **Affected leaf area** | {r['affected_pct']:.1f}% |\n"
        f"| **Healthy leaf area** | {r['healthy_pct']:.1f}% |\n"
        f"| **Severity** | {r['severity']} |\n\n"
        f"Segmentation mask: diseased regions in red, estimated leaf outline in green.\n\n"
        f"<sub>Severity bands: {_bands_text()}</sub>"
    )

    lesion = r["lesion_mask"]
    shares = None
    if lesion.any():
        ids, counts = np.unique(pred[lesion], return_counts=True)
        shares = {id2label.get(int(i), str(i)): float(c) / counts.sum() for i, c in zip(ids, counts)}
    return vis, report, shares


DESCRIPTION = f"""
Upload a photo of a diseased leaf. The model segments the lesions pixel by pixel, names the disease
(116 diseases across 30+ crops from [PlantSeg](https://www.kaggle.com/datasets/weitianqi/plantseg)),
and estimates how much of the leaf is affected.

Model: [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO}) — SegFormer-B4, the winner of a 5-model
benchmark (U-Net ResNet34/50, SegFormer B0/B2/B4), test mIoU 46.1.
"""

NOTES = """
**How it works.** Each pixel is classified as background or one of 116 diseases. The disease shown is the
most common class among the lesion pixels. PlantSeg labels only lesions, not whole leaves, so the leaf area
is estimated from leaf colour plus the predicted lesions; affected % = lesion pixels / leaf pixels.

**Limitations.** Trained on PlantSeg field photos: accuracy drops for crops or diseases outside the dataset,
and for photos with many leaves or busy backgrounds. Area percentages are approximate. This is a research
demo, not a substitute for expert diagnosis. Model weights: CC BY-NC 4.0 (non-commercial).
"""

with gr.Blocks(title="PlantDoctor") as demo:
    gr.Markdown("# 🌿 PlantDoctor: plant disease segmentation")
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Leaf photo", sources=["upload", "webcam", "clipboard"])
            btn = gr.Button("Diagnose", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="Segmentation mask")
            out_report = gr.Markdown()
            out_label = gr.Label(label="Disease share of lesion area", num_top_classes=3)
    gr.Markdown(NOTES)
    btn.click(diagnose, inputs=inp, outputs=[out_img, out_report, out_label])
    inp.upload(diagnose, inputs=inp, outputs=[out_img, out_report, out_label])

if __name__ == "__main__":
    demo.queue(max_size=16).launch()
