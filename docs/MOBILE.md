# PlantDoctor on mobile: offline feasibility test

**Question:** can the trained model run on a phone with no internet, fast and accurate enough for an app?

**Answer: yes, confirmed on an iPhone 14 Pro.** SegFormer-B4 int8 (69 MB, −0.46 mIoU) produced the full
report (disease, affected %, severity, mask) in **airplane mode**, at a median of 4.2 s per photo in Safari
(the slowest way to run it; a native app should be roughly 10× faster). Details in section 4.

## 1. Exported model files

Published in the model repo: [aungthuhein-dev/plantdoctor-segformer-b4/onnx](https://huggingface.co/aungthuhein-dev/plantdoctor-segformer-b4/tree/main/onnx)
(`model.onnx` fp32, `model_fp16.onnx`, `model_int8.onnx`, `labels.json`).

`scripts/export_mobile.py` (run by `slurm/export_mobile.sbatch`, CPU-only) converts a checkpoint to ONNX:

| File | What it is |
|---|---|
| `model_fp32.onnx` | full precision, identical predictions to PyTorch |
| `model_fp16.onnx` | half-precision weights and compute, float32 input/output |
| `model_int8.onnx` | static int8 (QDQ, per-channel), calibrated on 64 training images |
| `labels.json` | class id → disease name (117 entries) |

**Model contract** (all files): input `pixel_values` float32 `[1, 3, 512, 512]`, RGB resized to 512×512,
`(x/255 − mean)/std` with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
Output `logits` float32 `[1, 117, 128, 128]` (1/4 resolution): upsample, then argmax per pixel.

## 2. Accuracy and size (200 test images, ONNX Runtime CPU)

| Model | Version | Size | mIoU | Δ vs PyTorch | Lesion IoU | Pixels identical to PyTorch | CPU time* |
|---|---|--:|--:|--:|--:|--:|--:|
| SegFormer-B4 | fp32 | 257 MB | 48.15 | 0.00 | 67.91 | 100.0% | 501 ms |
| SegFormer-B4 | fp16 | 129 MB | 48.14 | −0.01 | 67.91 | 99.99% | 550 ms |
| SegFormer-B4 | **int8** | **69 MB** | **47.69** | **−0.46** | 66.93 | 97.98% | **272 ms** |
| SegFormer-B2 | fp32 | 110 MB | 46.30 | 0.00 | 65.46 | 100.0% | 378 ms |
| SegFormer-B2 | fp16 | 55 MB | 46.29 | −0.01 | 65.47 | 99.99% | 359 ms |
| SegFormer-B2 | int8 | 29 MB | 43.76 | −2.54 | 63.28 | 96.60% | 158 ms |

<sub>*AMD EPYC 9355 server CPU limited to 4 threads, native ONNX Runtime 1.32. mIoU is on a 200-image
subset of the test split, so it differs from the full-test 46.1 / 44.6.</sub>

- **fp16 is lossless** (−0.01 mIoU) at half the size.
- **B4 int8 keeps almost all accuracy** (−0.46) at a quarter of the size and about 2× the speed.
- **B2 int8 loses 2.5 mIoU**, which makes it *less* accurate than B4 int8 while being only 2.4× smaller,
  so B2 should be used in fp16 if at all.

## 3. Offline browser test (laptop)

Test page: `mobile_test/` (ONNX Runtime Web 1.31.1), live at
**https://aungthuhein-dev-plantdoctor-mobile-test.static.hf.space**. A service worker caches the page,
runtime and model, and enables multithreaded WebAssembly. Checked on the laptop: with the web server
**stopped**, the page reloads, loads the cached model and produces a diagnosis.

Time per photo in Chromium on the development laptop (Intel CPU, 4 WebAssembly threads / NVIDIA RTX 4050
via WebGPU), median after warm-up:

| Model | CPU (WebAssembly) | GPU (WebGPU) |
|---|--:|--:|
| B4 fp32 | 5.7 s | 1.4 s |
| B4 int8 | 3.9 s | 14.8 s |
| B2 int8 | 2.1 s | 7.2 s |
| B2 fp16 | 3.7 s | 0.37 s |

- int8 suits the **CPU** engine; fp32/fp16 suit the **GPU** engine (int8 is very slow on WebGPU).
- The browser is roughly 10× slower than native ONNX Runtime on CPU (3.9 s vs about 0.3–0.5 s), so a
  native app will be much faster than these numbers.
- Same photo, all versions: *apple black rot*, 23.6–24.1% affected, *Moderate*. The Python version gives
  23.3% on a photo of the same disease.

## 4. iPhone results (fill in)

Open the link above in **Safari** on the iPhone, tap **Load model** for each version, turn on **airplane mode**,
then pick a leaf photo and tap **Speed test ×5**. Record the median from the log.

| iPhone model / iOS | Model | Engine | Median time | Worked offline? |
|---|---|---|--:|:--:|
| iPhone 14 Pro (A16), Safari | B4 int8 | CPU | 4,244 ms | ✅ airplane mode |
| | B4 fp16 | GPU | | |
| | B2 fp16 | GPU | | |
| | B2 int8 | CPU | | |

The iPhone 14 Pro matches the laptop's browser CPU speed (4.2 s vs 3.9 s for B4 int8). Native ONNX
Runtime is about 10× faster than the browser on the same CPU (section 3), so the same model in a native
app should take roughly 0.4–0.6 s on this phone (estimate, not measured).

## 5. Recommendation for the app (provisional)

- **Ship SegFormer-B4 int8 (69 MB)** with ONNX Runtime Mobile: −0.46 mIoU, fits easily in an app, and
  native CPU inference should take well under a second on recent phones. Keep fp16 as the option for GPU
  delegates (Core ML / NNAPI) if int8 is slow there.
- Port the report logic (argmax, majority disease, leaf mask with HSV + hole filling, severity) from
  `mobile_test/index.html` into the app; it is already written without OpenCV.
- Final choice after the iPhone numbers in section 4.
