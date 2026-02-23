# Video & Image Tagger for LoRA Training

Multi-tagger pipeline for preparing datasets. Supports **images** and **videos** with automatic frame extraction.

## Taggers

| Tagger | Type | Description |
|--------|------|-------------|
| **wd14** | Local (ONNX) | `SmilingWolf/wd-eva02-large-tagger-v3` booru tags |
| **camie** | Local (ONNX) | `Camais03/camie-tagger-v2` booru tags |
| **pixai** | Local (PyTorch) | `pixai-labs/pixai-tagger-v0.9` booru tags |
| **grok** | API (OpenRouter) | Vision LLM captioning with tag context |

## Required API Keys

| Key | Required For | How to Set |
|-----|-------------|------------|
| `OPENROUTER_API_KEY` | grok tagger | `export OPENROUTER_API_KEY=sk-or-...` or `--grok_api_key` |
| `HF_TOKEN` | pixai (gated model) | `export HF_TOKEN=hf_...` or `--hf_token` |

## Quick Start (Interactive CLI)

```bash
python cli.py
```

The CLI will:
1. Create a virtual environment
2. Install all dependencies
3. Ask for your input directory
4. Let you choose taggers and options
5. Run the pipeline with progress bars

## Manual Usage

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Tag images

```bash
python tag_images_by_wd14_tagger.py /path/to/images \
  --taggers wd14,pixai \
  --batch_size 8 --recursive
```

### Tag videos (extract frame 12, tag it)

```bash
python tag_images_by_wd14_tagger.py /path/to/videos \
  --video \
  --taggers pixai,grok \
  --batch_size 4 --recursive
```

Output: `video_name.txt` next to `video_name.mp4`

### Video + Grok pipeline (recommended for video LoRA)

```bash
export OPENROUTER_API_KEY=sk-or-...
export HF_TOKEN=hf_...

python tag_images_by_wd14_tagger.py /path/to/videos \
  --video \
  --taggers pixai,grok \
  --batch_size 4 \
  --grok_concurrency 8 \
  --recursive
```

Pipeline: extract frame 12 -> pixai tags -> grok receives tags + image -> JSON caption output

### PRO mode (2 frames per video)

```bash
python tag_images_by_wd14_tagger.py /path/to/videos \
  --video --pro \
  --taggers pixai,grok \
  --batch_size 4 --recursive
```

PRO mode:
- Extracts **frame 6** and **frame 30** from each video
- Runs taggers on **both frames**, merges tags (deduped)
- Sends **both images** + merged tags to grok
- Better quality captions for videos with motion

Custom frame numbers: `--pro_frame_a 10 --pro_frame_b 45`

## Grok Configuration

Grok uses prompt files from `prompts/`:
- `prompts/system_prompt.md` - System instructions for the LLM
- `prompts/user_prompt.md` - User prompt template (use `{tags}` placeholder for booru tags)

Override with: `--grok_system_prompt_file /path/to/custom.md`

### Grok model

Default: `x-ai/grok-2-vision-1212`

Change with: `--grok_model google/gemini-2.0-flash-001`

Any vision model on OpenRouter works.

## Shell Script

```bash
# Tag videos with pixai+grok
./run_tagger.sh /path/to/videos 4 video

# Tag videos with grok only
./run_tagger.sh /path/to/videos 1 grok

# Tag images (default)
./run_tagger.sh /path/to/images 8
```

## Thresholds

### Global
- `--thresh 0.35` (default for all taggers)

### Per tagger
- `--wd14_thresh`, `--camie_thresh`, `--pixai_thresh`
- `--wd14_general_threshold`, `--wd14_character_threshold`
- `--camie_general_threshold`, `--camie_character_threshold`
- `--pixai_general_threshold`, `--pixai_character_threshold`

### PixAI modes
```bash
--pixai_mode threshold   # default: general=0.30, character=0.85
--pixai_mode topk        # top-K tags: --pixai_topk_general 25 --pixai_topk_character 10
```

## All CLI Options

```
positional:
  train_data_dir              Directory with images/videos (default: .)

mode:
  --video                     Video mode: extract frames and tag
  --pro                       PRO mode: 2 frames per video
  --frame_number N            Frame to extract in normal mode (default: 12)
  --pro_frame_a N             First frame for PRO mode (default: 6)
  --pro_frame_b N             Second frame for PRO mode (default: 30)

taggers:
  --taggers LIST              Comma-separated: wd14,camie,pixai,grok
  --one_tagger NAME           Use only one tagger

grok:
  --grok_api_key KEY          OpenRouter API key
  --grok_model MODEL          Model ID (default: x-ai/grok-2-vision-1212)
  --grok_system_prompt_file   Path to system prompt .md
  --grok_prompt_file          Path to user prompt .md
  --grok_concurrency N        Parallel API calls (default: 8)

processing:
  --batch_size N              Batch size for local taggers (default: 1)
  --max_workers N             Image loading threads (default: 4)
  --recursive                 Search subdirectories
  --remove_underscore         Replace _ with space in tags

output:
  --output_path FILE          Write to JSON/JSONL instead of .txt files
  --caption_extension EXT     File extension (default: .txt)
  --caption_separator SEP     Tag separator (default: ", ")
  --append_tags               Append to existing .txt files

tokens:
  --hf_token TOKEN            HuggingFace token for gated models
```

## Notes

- First run downloads models to `models/` directory
- For max speed, increase `--batch_size` until VRAM is full
- Frame 12 is used by default (frame 1 can be black/buggy in some videos)
- Grok always runs LAST so it has access to booru tagger output
- Supported video formats: mp4, avi, mov, mkv, webm, flv, wmv
