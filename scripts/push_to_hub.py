#!/usr/bin/env python
"""Export the winning model and upload it to the Hugging Face Hub.

    huggingface-cli login            # once, with a write token
    python scripts/push_to_hub.py --repo <hf-user>/plantdoctor-segformer-b2 [--ckpt ...] [--private]

Without --ckpt the winner from results/winner.json is used. Use --dry-run to only build the folder.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from plantdoctor.models import build_model, export_pretrained  # noqa: E402

CARD = """---
license: cc-by-nc-4.0
library_name: {library}
pipeline_tag: image-segmentation
tags:
- plant-disease
- semantic-segmentation
- agriculture
- plantseg
datasets:
- PlantSeg
metrics:
- mean_iou
- dice
---

# PlantDoctor - {name}

Semantic segmentation of plant disease lesions ({n_cls} disease classes + background), trained on
[PlantSeg](https://www.kaggle.com/datasets/weitianqi/plantseg). It is the winning model of the
PlantDoctor benchmark (U-Net ResNet34/50 vs SegFormer B0/B2/B4).

From the mask, PlantDoctor gives a per-leaf report:

```
Disease prediction: Early blight
Affected leaf area: 23.6%
Healthy leaf area: 76.4%
Segmentation mask: Highlighted diseased regions
Severity: Moderate
```

## Test results (PlantSeg test split, {size}x{size})

| Metric | Value |
|---|---|
| mIoU (all classes) | {miou:.2f} |
| mIoU (disease classes) | {miou_d:.2f} |
| mDice | {mdice:.2f} |
| Precision / Recall (macro) | {prec:.2f} / {rec:.2f} |
| Lesion-vs-background IoU / Dice | {biou:.2f} / {bdice:.2f} |
| Image-level disease accuracy | {acc:.2f} |
| Affected-area MAE (percentage points) | {mae:.2f} |
| Severity accuracy | {sev:.2f} |
| Parameters | {params:.1f} M |
| Latency, batch 1 ({gpu}) | {lat32:.1f} ms fp32 / {lat16:.1f} ms bf16 |

## Full benchmark

{benchmark}

## Usage

{usage}

Leaf area is estimated with a colour heuristic plus the predicted lesions (PlantSeg has no leaf
masks); see `plantdoctor/analysis.py`. Severity bands: <1% Healthy, 1-10% Mild, 10-30% Moderate,
>=30% Severe.

## Training

{arch}, ImageNet-pretrained encoder, fine-tuned for {epochs} epochs at {size}x{size} (batch {bs}, AdamW,
warmup + poly LR, cross-entropy + Dice loss, bf16). Best checkpoint by validation mIoU (epoch {best_ep}).
Data: PlantSeg official split, 7,916 train / 1,247 val / 2,295 test images.

## License and data

The weights are released for **non-commercial use** (CC BY-NC 4.0), because the training data,
[PlantSeg](https://www.kaggle.com/datasets/weitianqi/plantseg), is licensed CC BY-NC-ND 4.0.
Please cite the PlantSeg dataset if you use this model.

## Limitations

- Trained on PlantSeg field images; accuracy drops on unseen crops/diseases.
- Affected/healthy percentages depend on the heuristic leaf mask and are approximate.
- Not a substitute for expert diagnosis.
"""

USAGE_SEGFORMER = """```python
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

repo = "{repo}"
model = SegformerForSemanticSegmentation.from_pretrained(repo).eval()
proc = SegformerImageProcessor(size={{"height": {size}, "width": {size}}})
img = Image.open("leaf.jpg").convert("RGB")
with torch.no_grad():
    logits = model(**proc(img, return_tensors="pt")).logits
mask = torch.nn.functional.interpolate(logits, size=img.size[::-1], mode="bilinear").argmax(1)[0]
```

Or the full report: `python scripts/predict.py --ckpt best.pt --image leaf.jpg --out overlay.png`."""

USAGE_UNET = """```python
import segmentation_models_pytorch as smp
model = smp.from_pretrained("{repo}").eval()
# input: ImageNet-normalised RGB, {size}x{size}; output: (B, C, H, W) logits
```

Or the full report: `python scripts/predict.py --ckpt best.pt --image leaf.jpg --out overlay.png`."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--report", default="results")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ckpt = a.ckpt or json.load(open(os.path.join(a.report, "winner.json")))["checkpoint"]
    run_dir = os.path.dirname(ckpt)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg, id2label = ck["cfg"], {int(k): v for k, v in ck["id2label"].items()}
    res = json.load(open(os.path.join(run_dir, "results.json")))
    model = build_model(cfg["model"], len(id2label), pretrained=False, id2label=id2label)
    model.load_state_dict(ck["model"])

    stage = os.path.join(run_dir, "hf_export")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    export_pretrained(model, stage)
    shutil.copy(ckpt, os.path.join(stage, "best.pt"))
    json.dump({str(k): v for k, v in id2label.items()}, open(os.path.join(stage, "classes.json"), "w"), indent=2)
    shutil.copytree(os.path.join(ROOT, "plantdoctor"), os.path.join(stage, "plantdoctor"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    os.makedirs(os.path.join(stage, "scripts"))
    shutil.copy(os.path.join(ROOT, "scripts", "predict.py"), os.path.join(stage, "scripts", "predict.py"))
    for f in ("benchmark.png", "val_curves.png"):
        if os.path.exists(os.path.join(a.report, f)):
            shutil.copy(os.path.join(a.report, f), os.path.join(stage, f))

    bench_md = os.path.join(a.report, "benchmark.md")
    bench = open(bench_md).read() if os.path.exists(bench_md) else "_run scripts/aggregate.py_"
    if os.path.exists(os.path.join(stage, "benchmark.png")):
        bench += "\n![benchmark](benchmark.png)\n"
    t = res["test"]
    seg = cfg["model"].startswith("segformer")
    size = res["img_size"]
    card = CARD.format(
        library="transformers" if seg else "segmentation-models-pytorch",
        name=cfg["name"], n_cls=len(id2label) - 1, size=size,
        miou=100 * t["mIoU"], miou_d=100 * t["mIoU_disease"], mdice=100 * t["mDice"],
        prec=100 * t["mPrecision"], rec=100 * t["mRecall"], biou=100 * t["binary_IoU"],
        bdice=100 * t["binary_Dice"], acc=100 * t["image_disease_acc"],
        mae=t.get("affected_area_MAE_pct", float("nan")), sev=100 * t.get("severity_acc", float("nan")),
        params=res["params_M"], gpu=res["gpu"], lat32=res["latency_fp32"]["median_ms"],
        lat16=res["latency_bf16"]["median_ms"], benchmark=bench,
        arch=("SegFormer (" + cfg["model"].split("_")[1].upper() + ")") if seg
        else "U-Net (" + cfg["model"].split("_", 1)[1] + ")",
        epochs=res["epochs"], bs=res["batch_size"], best_ep=res["best_epoch"] + 1,
        usage=(USAGE_SEGFORMER if seg else USAGE_UNET).format(repo=a.repo, size=size),
    )
    with open(os.path.join(stage, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)
    print(f"export staged at {stage}")
    if a.dry_run:
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(a.repo, private=a.private, exist_ok=True)
    api.upload_folder(folder_path=stage, repo_id=a.repo, commit_message=f"Upload PlantDoctor {cfg['name']}")
    print(f"uploaded: https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
