#!/bin/bash
#SBATCH --job-name=plantdoctor
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=defq
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
# Usage (from the project root): sbatch --job-name=pd_segformer_b2 slurm/train.sbatch configs/segformer_b2.yaml [extra train.py args]
# Every run uses the same MIG slice (3g.71gb) so timings/latency are comparable.
# Requeue-safe: --resume continues from outputs/<name>/last.pt.

set -eo pipefail

# Ignore packages in ~/.local so only the plantdoctor env's own packages are used
export PYTHONNOUSERSITE=1

source /cm/shared/apps/miniconda3/etc/profile.d/conda.sh
conda activate plantdoctor
# Enabled after activation: conda's activate scripts can reference unset variables
set -u

cd "$SLURM_SUBMIT_DIR"
CONFIG=${1:?config path required}
shift || true
export TORCH_HOME=$PWD/.cache/torch

# Log GPU and driver
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
echo "host=$(hostname) config=$CONFIG"

python scripts/train.py --config "$CONFIG" --data-root data/plantseg --out-dir outputs --resume "$@"
