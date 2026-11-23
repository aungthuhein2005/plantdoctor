---
title: PlantDoctor
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.32.0
python_version: "3.11"
app_file: app.py
pinned: false
license: cc-by-nc-4.0
models:
- aungthuhein-dev/plantdoctor-segformer-b4
short_description: Plant disease segmentation, leaf area and severity
---

# PlantDoctor

Upload a leaf photo and get:

```
Disease prediction: Early blight
Affected leaf area: 23.6%
Healthy leaf area: 76.4%
Segmentation mask: Highlighted diseased regions
Severity: Moderate
```

Model: [aungthuhein-dev/plantdoctor-segformer-b4](https://huggingface.co/aungthuhein-dev/plantdoctor-segformer-b4)
(SegFormer-B4 trained on PlantSeg, 116 diseases). Runs on ZeroGPU; a prediction takes about a second once the GPU is attached.

Weights are CC BY-NC 4.0 (non-commercial), following the PlantSeg dataset license (CC BY-NC-ND 4.0).
