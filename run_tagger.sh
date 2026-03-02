#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  echo "Usage: $0 /path/to/images_or_videos [batch_size] [mode]"
  echo ""
  echo "Modes:"
  echo "  images           (default) - Tag images with wd14/camie/pixai"
  echo "  images-caption             - Tag images with pixai + generate caption with grok"
  echo "  images-grok                - Caption images with grok only (uses existing .txt if present)"
  echo "  images-xai-submit          - Submit all image requests to xAI Batch API"
  echo "  images-xai-status          - Check xAI Batch API progress from saved state"
  echo "  images-xai-collect         - Download completed xAI Batch results and write captions"
  echo "  video                      - Extract frame 12 and caption with pixai + grok"
  echo "  video-pro                  - PRO mode: 2 frames per video, better quality"
  echo "  video-grok                 - Caption videos with grok only"
  echo ""
  echo "Examples:"
  echo "  $0 /data/images 8 images"
  echo "  $0 /data/images 4 images-caption"
  echo "  $0 /data/images 1 images-xai-submit"
  echo "  $0 /data/images 1 images-xai-status"
  echo "  $0 /data/images 1 images-xai-collect"
  echo "  $0 /data/videos 4 video"
  echo "  $0 /data/videos 4 video-pro"
}

if [ $# -lt 1 ]; then
  usage
  exit 1
fi

INPUT_DIR="$1"
BATCH_SIZE="${2:-4}"
MODE="${3:-images}"

if [ ! -d "$INPUT_DIR" ]; then
  echo "[!] Input directory not found: $INPUT_DIR"
  exit 1
fi

cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q -r requirements.txt

BASE_CMD=(
  python3 tag_images_by_wd14_tagger.py "$INPUT_DIR"
  --batch_size "$BATCH_SIZE"
  --recursive
  --remove_underscore
  --thresh 0.30
)

if [[ "$MODE" == *"grok"* || "$MODE" == *"caption"* ]]; then
  if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "[!] OPENROUTER_API_KEY is required for mode: $MODE"
    exit 1
  fi
fi

if [[ "$MODE" == images-xai-* ]]; then
  if [ -z "${XAI_API_KEY:-}" ]; then
    echo "[!] XAI_API_KEY is required for mode: $MODE"
    exit 1
  fi
fi

case "$MODE" in
  images)
    "${BASE_CMD[@]}" \
      --taggers wd14,camie,pixai \
      --wd14_thresh 0.28 \
      --camie_thresh 0.28 \
      --pixai_thresh 0.28
    ;;

  images-caption)
    "${BASE_CMD[@]}" \
      --taggers pixai,grok \
      --grok_concurrency 8
    ;;

  images-grok)
    "${BASE_CMD[@]}" \
      --taggers grok \
      --grok_concurrency 8 \
      --append_tags
    ;;

  images-xai-submit)
    "${BASE_CMD[@]}" \
      --taggers grok \
      --append_tags \
      --grok_provider xai-batch \
      --xai_batch_action submit \
      --xai_batch_submit_chunk 1000
    ;;

  images-xai-status)
    "${BASE_CMD[@]}" \
      --taggers grok \
      --append_tags \
      --grok_provider xai-batch \
      --xai_batch_action status
    ;;

  images-xai-collect)
    "${BASE_CMD[@]}" \
      --taggers grok \
      --append_tags \
      --grok_provider xai-batch \
      --xai_batch_action collect
    ;;

  video)
    "${BASE_CMD[@]}" \
      --video \
      --taggers pixai,grok \
      --grok_concurrency 8
    ;;

  video-pro)
    "${BASE_CMD[@]}" \
      --video --pro \
      --taggers pixai,grok \
      --grok_concurrency 8
    ;;

  video-grok)
    "${BASE_CMD[@]}" \
      --video \
      --taggers grok \
      --grok_concurrency 8
    ;;

  *)
    echo "[!] Invalid mode: $MODE"
    usage
    exit 1
    ;;
esac
