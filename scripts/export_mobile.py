#!/usr/bin/env python
"""Export a trained checkpoint to ONNX for on-device (offline) inference and validate it.

    python scripts/export_mobile.py --ckpt outputs/segformer_b4/best.pt --out mobile/segformer_b4

Produces, in --out:
    model_fp32.onnx   full precision
    model_fp16.onnx   half-precision weights/compute, float32 inputs/outputs
    model_int8.onnx   static int8 (QDQ, per-channel), calibrated on training images
    report.json       size, accuracy vs PyTorch, CPU latency per variant
    labels.json       id -> disease name (for the app)

Model contract (all variants):
    input  "pixel_values": float32 [1, 3, S, S], RGB resized to SxS, (x/255 - mean)/std (ImageNet)
    output "logits":       float32 [1, C, S/4, S/4] for SegFormer; upsample + argmax on device
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plantdoctor.data import PlantSegDataset  # noqa: E402
from plantdoctor.metrics import ConfusionMatrix  # noqa: E402
from plantdoctor.models import build_model  # noqa: E402


class RawLogits(torch.nn.Module):
    """Return the network's native-resolution logits (no upsampling inside the graph)."""

    def __init__(self, wrapper):
        super().__init__()
        self.net = wrapper.net
        self.segformer = wrapper.__class__.__name__ == "SegFormerWrapper"

    def forward(self, pixel_values):
        if self.segformer:
            return self.net(pixel_values=pixel_values).logits
        return self.net(pixel_values)


def export_fp32(model, path, size):
    import onnx

    x = torch.randn(1, 3, size, size)
    kw = dict(input_names=["pixel_values"], output_names=["logits"])
    try:  # legacy TorchScript exporter: most compatible graphs for mobile runtimes
        torch.onnx.export(model, (x,), path, dynamo=False, opset_version=17, do_constant_folding=True, **kw)
        how = "torchscript"
    except Exception as e:  # noqa: BLE001
        print(f"legacy exporter unavailable ({type(e).__name__}: {e}); using dynamo exporter", flush=True)
        torch.onnx.export(model, (x,), path, dynamo=True, opset_version=18, external_data=False, **kw)
        how = "dynamo"
    m = onnx.load(path)
    onnx.checker.check_model(m)
    return how, m.opset_import[0].version


def export_fp16(src, dst):
    import onnx
    from onnxconverter_common import float16

    onnx.save(float16.convert_float_to_float16(onnx.load(src), keep_io_types=True), dst)


def export_int8(src, dst, calib_images):
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = dst.replace(".onnx", "_pre.onnx")
    quant_pre_process(src, pre, skip_symbolic_shape=True)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib_images)

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"pixel_values": x}

    quantize_static(pre, dst, Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    op_types_to_quantize=["Conv", "MatMul", "Gemm"])
    os.remove(pre)


def ort_session(path, threads):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def latency_ms(run, x, warmup=5, iters=20):
    for _ in range(warmup):
        run(x)
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        run(x)
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="data/plantseg")
    ap.add_argument("--n-eval", type=int, default=200, help="test images for the accuracy check")
    ap.add_argument("--n-calib", type=int, default=64, help="train images for int8 calibration")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for latency (phone-like)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(a.threads)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg, id2label = ck["cfg"], {int(k): v for k, v in ck["id2label"].items()}
    nc, size = len(id2label), cfg.get("img_size", 512)
    wrapper = build_model(cfg["model"], nc, pretrained=False, id2label=id2label)
    wrapper.load_state_dict(ck["model"])
    model = RawLogits(wrapper).eval()
    with open(os.path.join(a.out, "labels.json"), "w") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, indent=1)

    paths = {k: os.path.join(a.out, f"model_{k}.onnx") for k in ("fp32", "fp16", "int8")}
    print("exporting fp32 ...", flush=True)
    how, opset = export_fp32(model, paths["fp32"], size)
    print("exporting fp16 ...", flush=True)
    export_fp16(paths["fp32"], paths["fp16"])
    print("calibrating + exporting int8 ...", flush=True)
    calib_ds = PlantSegDataset(a.data_root, "train", size)
    step = max(len(calib_ds) // a.n_calib, 1)
    calib = [calib_ds[i][0].unsqueeze(0).numpy() for i in range(0, len(calib_ds), step)][: a.n_calib]
    export_int8(paths["fp32"], paths["int8"], calib)

    test_ds = PlantSegDataset(a.data_root, "test", size)
    step = max(len(test_ds) // a.n_eval, 1)
    samples = [test_ds[i] for i in range(0, len(test_ds), step)][: a.n_eval]

    def to_pred(logits):
        t = torch.as_tensor(logits, dtype=torch.float32)
        return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False).argmax(1)[0]

    runners = {"pytorch_fp32": lambda x: model(torch.from_numpy(x)).numpy()}
    for k, p in paths.items():
        runners[f"onnx_{k}"] = (lambda s: (lambda x: s.run(["logits"], {"pixel_values": x})[0]))(
            ort_session(p, a.threads))

    report = {
        "model": cfg["model"], "img_size": size, "num_classes": nc, "n_eval": len(samples),
        "threads": a.threads, "cpu": platform.processor() or platform.machine(),
        "exporter": how, "opset": opset,
        "input": {"name": "pixel_values", "shape": [1, 3, size, size], "dtype": "float32",
                  "scale": "x/255", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "output": {"name": "logits", "note": "1/4 resolution for SegFormer; bilinear upsample, then argmax"},
        "variants": {},
    }
    ref = None
    for name, run in runners.items():
        cm = ConfusionMatrix(nc, "cpu")
        preds = []
        for x, y in samples:
            p = to_pred(run(x.unsqueeze(0).numpy()))
            cm.update(p, y)
            preds.append(p)
        m = cm.compute()
        ref = ref or preds
        v = {
            "mIoU": m["mIoU"],
            "lesion_IoU": m["binary_IoU"],
            "pixel_agreement_vs_pytorch": float(np.mean([(p == r).float().mean().item() for p, r in zip(preds, ref)])),
            "cpu_latency_ms": latency_ms(run, samples[0][0].unsqueeze(0).numpy()),
        }
        if name.startswith("onnx_"):
            v["size_MB"] = os.path.getsize(paths[name[5:]]) / 1e6
        report["variants"][name] = v
        print(f"{name:13s} " + "  ".join(f"{k}={val:.4f}" for k, val in v.items()), flush=True)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"report -> {a.out}/report.json")


if __name__ == "__main__":
    main()
