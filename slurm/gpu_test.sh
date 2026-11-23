#!/bin/bash
# Quick GPU test for any MIG slice; run it through srun, e.g.
#   srun -p defq -c8 --mem=128G --gres=gpu:1g.35gb:1 --time=00:30:00 bash slurm/gpu_test.sh
#   srun ... bash slurm/gpu_test.sh segformer_b4   # also run a 2-epoch smoke of one model (default: none)
set -eo pipefail
export PYTHONNOUSERSITE=1
source /cm/shared/apps/miniconda3/etc/profile.d/conda.sh
conda activate plantdoctor
set -u
cd "$(dirname "$0")/.."
export TORCH_HOME=$PWD/.cache/torch

echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi -L
python - <<'EOF'
import time, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
d = torch.cuda.get_device_properties(0)
print("gpu", d.name, f"{d.total_memory/2**30:.1f} GB")
x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); t = time.time()
for _ in range(20): x @ x
torch.cuda.synchronize()
print(f"bf16 matmul: {20*2*8192**3/(time.time()-t)/1e12:.1f} TFLOPS")
EOF

if [[ $# -ge 1 ]]; then
  m=$1
  rm -rf "outputs/${m}_smoke"
  python scripts/train.py --config "configs/${m}.yaml" --data-root data/plantseg --out-dir outputs --smoke
  python -c "import json; r=json.load(open('outputs/${m}_smoke/results.json')); print('peak train mem %.1f GB, latency fp32 %.1f ms' % (r['peak_train_mem_GB'], r['latency_fp32']['median_ms']))"
fi
echo GPU_TEST_OK
