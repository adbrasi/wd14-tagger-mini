#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace"
if [ ! -d "$ROOT_DIR" ]; then
  ROOT_DIR="/root"
fi

if [ $# -lt 1 ]; then
  echo "Uso: $0 /caminho/para/imagens [batch_size]"
  exit 1
fi

IMAGES_DIR="$1"
BATCH_SIZE="${2:-8}"

cd "$ROOT_DIR"

if [ ! -d "wd14-tagger-mini" ]; then
  git clone https://github.com/adbrasi/wd14-tagger-mini
fi

cd wd14-tagger-mini

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt

python tag_images_by_wd14_tagger.py "$IMAGES_DIR" \
  --taggers wd14,camie,pixai \
  --batch_size "$BATCH_SIZE" \
  --recursive \
  --thresh 0.30 \
  --wd14_thresh 0.28 \
  --camie_thresh 0.28 \
  --pixai_thresh 0.28
