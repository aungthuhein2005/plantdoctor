#!/bin/bash
#SBATCH --job-name=pd_mobile
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
# CPU-only: export ONNX fp32/fp16/int8 for mobile and validate accuracy, size and 4-thread CPU latency.
# Usage: sbatch slurm/export_mobile.sbatch segformer_b4 [segformer_b2 ...]

set -eo pipefail
export PYTHONNOUSERSITE=1
source /cm/shared/apps/miniconda3/etc/profile.d/conda.sh
conda activate plantdoctor
set -u
cd "$SLURM_SUBMIT_DIR"
export CUDA_VISIBLE_DEVICES=""

python -c "import torch, onnx, onnxruntime; print('torch', torch.__version__, 'onnx', onnx.__version__, 'ort', onnxruntime.__version__)"
lscpu | grep -E "Model name" || true
for m in "$@"; do
  echo "===== $m ====="
  python scripts/export_mobile.py --ckpt "outputs/$m/best.pt" --out "outputs/mobile/$m" --threads 4
done
echo EXPORT_OK
