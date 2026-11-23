#!/bin/bash
# Submit the full benchmark: smoke test -> 5 training jobs (start only if smoke passes) -> aggregation.
#   bash slurm/submit_all.sh                        # smoke + full runs on gpu:3g.71gb
#   bash slurm/submit_all.sh --no-smoke             # full runs only
#   GPU=1g.35gb bash slurm/submit_all.sh --no-smoke # run on the smaller MIG slice (needs account access)
# Each GPU job: -c8 --mem=128G --gres=gpu:$GPU:1 (max 5 concurrent per user).
# Keep every model on the same slice type so training time and latency stay comparable.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
GPU=${GPU:-3g.71gb}
case $GPU in
  3g.71gb) scale=1 ;;
  1g.35gb|1g.18gb) scale=3 ;;  # ~1/3 of the compute -> about 3x longer
  7g.141gb) scale=1 ;;
  *) echo "unknown GPU slice $GPU" >&2; exit 1 ;;
esac
gres=(--gres="gpu:${GPU}:1")
dep_opt=()
if [[ "${1:-}" != "--no-smoke" ]]; then
  smoke=$(sbatch --parsable "${gres[@]}" slurm/smoke_all.sbatch)
  echo "smoke -> $smoke"
  dep_opt=(--dependency=afterok:"$smoke" --kill-on-invalid-dep=yes)
fi
ids=()
for m in unet_resnet34 unet_resnet50 segformer_b0 segformer_b2 segformer_b4; do
  # Realistic per-model time limits (hours on 3g.71gb) help the backfill scheduler start jobs sooner
  # (resubmitting resumes from last.pt if a job hits its limit).
  case $m in
    unet_resnet34|segformer_b0) h=4 ;;
    unet_resnet50) h=5 ;;
    segformer_b2) h=8 ;;
    segformer_b4) h=12 ;;
  esac
  t="$((h * scale)):00:00"
  id=$(sbatch --parsable "${dep_opt[@]}" "${gres[@]}" --time="$t" --job-name="pd_${m}" \
       slurm/train.sbatch "configs/${m}.yaml")
  echo "submitted $m on $GPU (limit $t) -> $id"
  ids+=("$id")
done
dep=$(IFS=:; echo "${ids[*]}")
id=$(sbatch --parsable --job-name=pd_aggregate --dependency=afterok:"$dep" --kill-on-invalid-dep=yes \
     --partition=defq --cpus-per-task=2 --mem=8G --time=00:20:00 --output=logs/%x-%j.out \
     --wrap="export PYTHONNOUSERSITE=1; source /cm/shared/apps/miniconda3/etc/profile.d/conda.sh; conda activate plantdoctor; python scripts/aggregate.py --out-dir outputs --report results")
echo "aggregate -> $id (runs after all training jobs succeed)"
