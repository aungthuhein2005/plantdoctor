#!/bin/bash
# Download PlantSeg from Kaggle and build the train/val/test index.
# Needs Kaggle API credentials (https://www.kaggle.com/settings -> API):
#   new-style token (KGAT_...): ~/.kaggle/access_token  (or env KAGGLE_API_TOKEN)
#   legacy:                     ~/.kaggle/kaggle.json
# chmod 600 either file.
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=${RAW:-data/raw}
OUT=${OUT:-data/plantseg}
mkdir -p "$RAW"
if [[ ! -f "$RAW/.done" ]]; then
  kaggle datasets download -d weitianqi/plantseg -p "$RAW" --unzip
  touch "$RAW/.done"
fi
python scripts/prepare_data.py --raw "$RAW" --out "$OUT"
