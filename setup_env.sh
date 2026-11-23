#!/bin/bash
# One-time environment setup on the HPC login node.
#   bash setup_env.sh
# Creates the conda env "plantdoctor" (set PLANTDOCTOR_CONDA=0 for a .venv instead). Edit the module lines for your cluster.
set -euo pipefail
export PYTHONNOUSERSITE=1   # ignore packages in ~/.local
cd "$(dirname "$0")"

# module load python/3.11 cuda/12.1   # <- uncomment / adjust if your cluster uses modules

if [[ "${PLANTDOCTOR_CONDA:-1}" == "1" ]]; then
  conda env list | grep -q "^plantdoctor " || conda create -y -n plantdoctor python=3.11
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate plantdoctor
else
  python3 -m venv .venv
  source .venv/bin/activate
fi
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch, segmentation_models_pytorch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
