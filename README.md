# Video & Image Tagger for LoRA Training

Multi-tagger pipeline for preparing video/image datasets. Extracts frames from videos, tags them with booru taggers, then sends tags + images to a vision LLM for natural language captioning.

## How It Works

```
Videos (.mp4/.avi/...)
  -> Extract frame 12 (or frames 6+30 in PRO mode)
  -> Booru taggers (pixai/wd14/camie) generate structured tags
  -> Existing .txt tags are merged in (no duplicates)
  -> Grok (OpenRouter) receives: tags + image(s) + system prompt
  -> Grok returns JSON with natural language caption
  -> Output: video_name.txt next to video_name.mp4
```

## Required API Keys

| Key | Required For | How to Set |
|-----|-------------|------------|
| `OPENROUTER_API_KEY` | grok tagger | `export OPENROUTER_API_KEY=sk-or-...` or `--grok_api_key` |
| `HF_TOKEN` | pixai (gated model) | `export HF_TOKEN=hf_...` or `--hf_token` |

Get your OpenRouter key at https://openrouter.ai/keys

Get your HuggingFace token at https://huggingface.co/settings/tokens

## Quick Start (Interactive CLI)

```bash
python cli.py
```

The CLI handles everything: venv creation, dependency install, menu-driven options, progress bars.

## Manual Usage

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Tag videos (recommended pipeline)

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

### PRO mode (2 frames per video, better quality)

```bash
python tag_images_by_wd14_tagger.py /path/to/videos \
  --video --pro \
  --taggers pixai,grok \
  --batch_size 4 --recursive
```

PRO mode extracts **frame 6** and **frame 30**, runs taggers on both (deduped), and sends **both images** to grok.

Custom frame numbers: `--pro_frame_a 10 --pro_frame_b 45`

### Tag images (no video extraction)

```bash
python tag_images_by_wd14_tagger.py /path/to/images \
  --taggers wd14,pixai \
  --batch_size 8 --recursive
```

## Existing .txt Files

In video mode, if a `.txt` file already exists next to a video, its tags are **automatically loaded and merged** with the new tagger output (duplicates removed). This means you can run the pipeline multiple times and it accumulates tags.

For image mode, use `--append_tags` to enable the same behavior.

## Taggers

| Tagger | Type | Description |
|--------|------|-------------|
| **wd14** | Local (ONNX) | `SmilingWolf/wd-eva02-large-tagger-v3` booru tags |
| **camie** | Local (ONNX) | `Camais03/camie-tagger-v2` booru tags |
| **pixai** | Local (PyTorch) | `pixai-labs/pixai-tagger-v0.9` booru tags |
| **grok** | API (OpenRouter) | Vision LLM - receives tags + image, outputs natural language caption |

**Important:** Grok always runs **last** so it has access to all booru tagger output as context.

## Grok / OpenRouter Details

### How it works

1. Booru taggers run first and generate structured tags
2. Grok receives: system prompt + user prompt (with `{tags}` replaced by booru tags) + frame image(s)
3. API call uses `response_format: json_object` for reliable JSON output
4. The `response-healing` plugin auto-fixes malformed JSON
5. The `caption` field is extracted from the JSON and written to `.txt`

### Prompt files

Prompts are loaded from `prompts/` directory:
- `prompts/system_prompt.md` - System instructions (what grok should do)
- `prompts/user_prompt.md` - User prompt template (`{tags}` is replaced with booru tags)

Override with: `--grok_system_prompt_file /path/to/custom.md` and `--grok_prompt_file /path/to/custom.md`

### JSON output structure

The system prompt instructs grok to return:
```json
{
  "caption": "A detailed natural language caption...",
  "tags_used": ["tag1", "tag2"],
  "tags_ignored": ["bad_tag"],
  "motion_description": "Brief motion/loop description",
  "style": "anime / realistic / 3d / etc"
}
```

Only the `caption` field is written to the `.txt` file.

### Model selection

Default: `x-ai/grok-2-vision-1212`

Any vision model on OpenRouter works:
```bash
--grok_model google/gemini-2.0-flash-001
--grok_model anthropic/claude-sonnet-4
--grok_model openai/gpt-4o
```

### Concurrency

For thousands of videos, increase concurrent API calls:
```bash
--grok_concurrency 16
```

Default is 8. The API uses exponential backoff on rate limits (429).

## Shell Script

```bash
# Tag videos with pixai+grok
./run_tagger.sh /path/to/videos 4 video

# PRO mode (2 frames)
./run_tagger.sh /path/to/videos 4 video-pro

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
  --append_tags               Append to existing .txt files (image mode)

tokens:
  --hf_token TOKEN            HuggingFace token for gated models
```

## Notes

- First run downloads models to `models/` directory
- For max speed, increase `--batch_size` until VRAM is full
- Frame 12 is used by default (frame 1 can be black/buggy)
- Grok uses `json_object` response format + `response-healing` plugin for reliable JSON
- Retries with exponential backoff on 429 (rate limit) and 5xx errors
- Supported video formats: mp4, avi, mov, mkv, webm, flv, wmv
- In video mode, existing .txt files are always merged (no `--append_tags` needed)
