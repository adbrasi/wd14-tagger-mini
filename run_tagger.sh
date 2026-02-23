#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace"
if [ ! -d "$ROOT_DIR" ]; then
  ROOT_DIR="/root"
fi

if [ $# -lt 1 ]; then
  echo "Usage: $0 /path/to/images_or_videos [batch_size] [mode]"
  echo ""
  echo "Modes:"
  echo "  images  (default) - Tag images with wd14/camie/pixai"
  echo "  video             - Extract frame 12 from videos, tag with pixai+grok"
  echo "  video-pro         - PRO mode: 2 frames per video, better quality"
  echo "  grok              - Use OpenRouter API only (requires OPENROUTER_API_KEY)"
  echo ""
  echo "Examples:"
  echo "  $0 /data/images 8"
  echo "  $0 /data/videos 4 video"
  echo "  $0 /data/videos 4 video-pro"
  echo "  $0 /data/videos 1 grok"
  exit 1
fi

IMAGES_DIR="$1"
BATCH_SIZE="${2:-4}"
MODE="${3:-images}"

cd "$ROOT_DIR"

if [ ! -d "wd14-tagger-mini" ]; then
  git clone https://github.com/adbrasi/wd14-tagger-mini
fi

cd wd14-tagger-mini

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q -r requirements.txt

if [ "$MODE" = "video" ]; then
  python tag_images_by_wd14_tagger.py "$IMAGES_DIR" \
    --video \
    --taggers pixai,grok \
    --batch_size "$BATCH_SIZE" \
    --recursive \
    --remove_underscore \
    --thresh 0.30 \
    --grok_concurrency 8
elif [ "$MODE" = "video-pro" ]; then
  python tag_images_by_wd14_tagger.py "$IMAGES_DIR" \
    --video --pro \
    --taggers pixai,grok \
    --batch_size "$BATCH_SIZE" \
    --recursive \
    --remove_underscore \
    --thresh 0.30 \
    --grok_concurrency 8
elif [ "$MODE" = "grok" ]; then
  python tag_images_by_wd14_tagger.py "$IMAGES_DIR" \
    --video \
    --taggers grok \
    --recursive
else
  python tag_images_by_wd14_tagger.py "$IMAGES_DIR" \
    --taggers wd14,camie,pixai \
    --batch_size "$BATCH_SIZE" \
    --recursive \
    --thresh 0.30 \
    --wd14_thresh 0.28 \
    --camie_thresh 0.28 \
    --pixai_thresh 0.28
fi
