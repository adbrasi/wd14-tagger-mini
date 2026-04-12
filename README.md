# data_araknideo

Multi-tagger pipeline for preparing video/image datasets. Downloads from MEGA/HuggingFace, preprocesses videos (frame cut + resize), tags with booru taggers, then sends to a vision LLM for natural language captioning. Uploads results to HuggingFace.

## How It Works

```
Data Source (Local / MEGA / HuggingFace)
  -> Download + flatten subfolders
  -> Validate media/txt pairs
  -> Preprocess: cut frames (F%8==1), resize (W/H multiples of 32)
  -> Booru taggers (pixai/wd14/camie) generate structured tags
  -> Grok (xAI Batch or OpenRouter) receives: tags + image + system prompt
  -> Grok returns JSON with natural language caption
  -> Output: video_name.txt next to video_name.mp4
  -> Upload to HuggingFace (optional)
```

## Required API Keys

| Key | Required For | How to Set |
|-----|-------------|------------|
| `XAI_API_KEY` | xAI Batch API (default for video) | `export XAI_API_KEY=...` |
| `OPENROUTER_API_KEY` | grok via OpenRouter | `export OPENROUTER_API_KEY=sk-or-...` |
| `HF_TOKEN` | pixai (gated model) + HF upload | `export HF_TOKEN=hf_...` |

## Quick Start (Interactive CLI)

```bash
python cli.py
```

The CLI wizard handles everything with Rich UI:
- Data source selection (local / MEGA / HuggingFace)
- Dataset validation (orphan detection)
- Video preprocessing (frame cut + resize)
- Tagger configuration
- xAI Batch or OpenRouter backend selection
- HuggingFace upload

## Manual Usage

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Tag videos (recommended pipeline)

```bash
export XAI_API_KEY=...
export HF_TOKEN=hf_...

python tag_images_by_wd14_tagger.py /path/to/videos \
  --video \
  --taggers pixai,grok \
  --grok_provider xai-batch \
  --xai_batch_action submit \
  --batch_size 4 \
  --recursive
```

### PRO mode (2 frames per video, better quality)

```bash
python tag_images_by_wd14_tagger.py /path/to/videos \
  --video --pro \
  --taggers pixai,grok \
  --batch_size 4 --recursive
```

PRO mode extracts **frame 6** and **frame 30**, runs taggers on both (deduped), and sends **both images** to grok.

### Tag images (no video extraction)

```bash
python tag_images_by_wd14_tagger.py /path/to/images \
  --taggers wd14,pixai \
  --batch_size 8 --recursive
```

## Data Sources

### MEGA Download
The CLI can download from MEGA shared links using MEGAcmd. It will:
1. Install MEGAcmd if not present
2. Download all files from the shared link
3. Flatten subfolders into a single directory
4. Preserve video+txt pairs, resolve name conflicts with numeric suffixes

### HuggingFace Download
Supports HuggingFace dataset URLs or IDs (e.g., `user/dataset`).

## Video Preprocessing

Before tagging, videos can be preprocessed:
- **Frame cutting**: Cuts to first N frames. Frame count is snapped to `F % 8 == 1` (1, 9, 17, 25, 33, 41, 49...)
- **Resize**: Width and height snapped to multiples of 32 (scale-to-fit with minimal crop)
- **Parallel**: Uses ProcessPoolExecutor for fast batch processing

## Dataset Validation

Before processing, the CLI validates media/txt pairs:
- Detects media files without captions
- Detects orphan .txt files without matching media
- Offers to delete or keep problematic files

## Taggers

| Tagger | Type | Description |
|--------|------|-------------|
| **wd14** | Local (ONNX) | `SmilingWolf/wd-eva02-large-tagger-v3` booru tags |
| **camie** | Local (ONNX) | `Camais03/camie-tagger-v2` booru tags |
| **pixai** | Local (PyTorch) | `pixai-labs/pixai-tagger-v0.9` booru tags |
| **grok** | API (xAI/OpenRouter) | Vision LLM - receives tags + image, outputs natural language caption |

**Important:** Grok always runs **last** so it has access to all booru tagger output as context.

## Grok Backends

### xAI Batch API (Recommended for large datasets)
- **50% cheaper** than real-time
- **No rate limits** on processing
- Submit → monitor → collect workflow
- Default model: `grok-4.20-beta-0309-reasoning`

### OpenRouter (Real-time)
- Concurrent requests (default: 32)
- Good for smaller datasets or testing
- Default model: `x-ai/grok-4.20-beta-0309-reasoning`

### Prompt files

Prompts are loaded from `prompts/` directory:
- `prompts/<mode>/<profile>/system_prompt.md`
- `prompts/<mode>/<profile>/user_prompt.md`

The interactive CLI discovers profiles automatically. Optional `profile.json` files can declare variables that the CLI will ask for, then inject via `--prompt_var KEY=VALUE`.

Examples:
- `prompts/image/generic-style/` asks for a style trigger like `anime screencap style`
- `prompts/image/generic-character/` asks for a character trigger like `my_character`

Implementation helpers:
- `prompt_profiles.py` handles preset discovery and variable prompts
- `xai_batch_state.py` centralizes xAI batch state naming/persistence
- `wizard_steps.py` contains reusable interactive wizard steps

Override with: `--grok_system_prompt_file` and `--grok_prompt_file`

## Thresholds

### Global
- `--thresh 0.35` (default for all taggers)

### Per tagger
- `--wd14_thresh`, `--camie_thresh`, `--pixai_thresh`
- `--wd14_general_threshold`, `--wd14_character_threshold`
- `--camie_general_threshold`, `--camie_character_threshold`
- `--pixai_general_threshold`, `--pixai_character_threshold`

## Processing Log

The pipeline saves a `.tagger_log.json` file to track processed files. On subsequent runs, already-processed files are **automatically skipped**.

- To reprocess everything: add `--force`
- If you add new taggers, unprocessed combinations are detected and run

## HuggingFace Upload

After tagging, the CLI offers to upload the dataset to HuggingFace:
- Private by default
- Uses `huggingface_hub` for upload

## Notes

- First run downloads models to `models/` directory
- For max speed, use `auto` batch size (VRAM-based, up to 64 for 80GB+ GPUs)
- Frame 12 is used by default (frame 1 can be black/buggy)
- Grok uses `json_object` response format for reliable JSON
- Retries with exponential backoff on 429 and 5xx errors
- Supported video formats: mp4, avi, mov, mkv, webm, flv, wmv
- Supported image formats: png, jpg, jpeg, webp, bmp, avif, jxl
