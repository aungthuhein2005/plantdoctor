---
title: PlantDoctor Offline Test
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: static
pinned: false
license: cc-by-nc-4.0
short_description: Run the PlantDoctor model on your phone, fully offline
---

# PlantDoctor offline test

Runs the PlantDoctor plant-disease segmentation model **on your phone**, in the browser, with no internet
after the first load (ONNX Runtime Web, WebAssembly or WebGPU).

1. Open this page and tap **Load model** (downloads and caches the model once).
2. Turn on **airplane mode**.
3. Pick a leaf photo and tap **Diagnose** or **Speed test ×5**.

Models: SegFormer-B4 / B2 from [aungthuhein-dev/plantdoctor-segformer-b4](https://huggingface.co/aungthuhein-dev/plantdoctor-segformer-b4),
exported to ONNX (fp32, fp16, int8). Weights CC BY-NC 4.0 (PlantSeg data is CC BY-NC-ND 4.0).
