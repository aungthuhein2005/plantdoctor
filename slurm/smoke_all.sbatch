#!/bin/bash
#SBATCH --job-name=pd_smoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=defq
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
# Pipeline check: 2 epochs on 64 images for every model, in one allocation. Fails if any model fails.

set -eo pipefail

# Ignore packages in ~/.local so only the plantdoctor env's own packages are used
export PYTHONNOUSERSITE=1

source /cm/shared/apps/miniconda3/etc/profile.d/conda.sh
conda activate plantdoctor
# Enabled after activation: conda's activate scripts can reference unset variables
set -u

cd "$SLURM_SUBMIT_DIR"
export TORCH_HOME=$PWD/.cache/torch

# Log GPU and driver
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

for m in unet_resnet34 unet_resnet50 segformer_b0 segformer_b2 segformer_b4; do
  echo "===== $m ====="
  rm -rf "outputs/${m}_smoke"
  python scripts/train.py --config "configs/${m}.yaml" --data-root data/plantseg --out-dir outputs --smoke
done
python scripts/predict.py --ckpt outputs/segformer_b0_smoke/best.pt \
  --image "$(sed -n 2p data/plantseg/index_test.csv | cut -d, -f1)" --out outputs/smoke_overlay.png
echo SMOKE_OK
