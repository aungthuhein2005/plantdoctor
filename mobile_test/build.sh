#!/bin/bash
# Assemble the offline test page: ONNX Runtime Web files + exported models + manifest.
#   bash mobile_test/build.sh <dir with exported models>   (default: outputs/mobile)
# Expects <dir>/<model>/model_{fp32,fp16,int8}.onnx from scripts/export_mobile.py.
set -euo pipefail
cd "$(dirname "$0")"
SRC=${1:-../outputs/mobile}
ORT_VERSION=1.31.1
mkdir -p ort models
for f in ort.webgpu.min.js ort-wasm-simd-threaded.wasm ort-wasm-simd-threaded.mjs \
         ort-wasm-simd-threaded.jsep.wasm ort-wasm-simd-threaded.jsep.mjs \
         ort-wasm-simd-threaded.asyncify.wasm ort-wasm-simd-threaded.asyncify.mjs; do
  [[ -f ort/$f ]] || curl -sfL -o "ort/$f" "https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/$f"
done
for mv in segformer_b4:fp32 segformer_b4:fp16 segformer_b4:int8 segformer_b2:fp16 segformer_b2:int8; do
  m=${mv%%:*}; v=${mv##*:}
  if [[ -f "$SRC/$m/model_$v.onnx" ]]; then cp "$SRC/$m/model_$v.onnx" "models/${m}_$v.onnx"; fi
  if [[ -f "$SRC/$m/labels.json" ]]; then cp "$SRC/$m/labels.json" models/labels.json; fi
done
python - <<'EOF'
import json, os
man = {f"models/{f}": round(os.path.getsize(f"models/{f}") / 1e6, 1) for f in sorted(os.listdir("models")) if f.endswith(".onnx")}
json.dump(man, open("models/manifest.json", "w"), indent=1)
print(man)
EOF
