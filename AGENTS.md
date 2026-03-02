# Repository Guidelines

## Project Structure & Module Organization
- Core scripts live at the repository root:
  - `tag_images_by_wd14_tagger.py`: main image/video tagging and caption pipeline.
  - `cli.py`: interactive launcher for setup and workflow selection.
  - `xai_batch_manager.py`: lightweight xAI batch status/snapshot utility.
  - `wd14_utils.py`: shared helpers (logging, file discovery, frame extraction, resizing).
  - `run_tagger.sh`: shell wrapper for common modes.
- Prompt templates are separated by modality:
  - `prompts/image/{system_prompt.md,user_prompt.md}`
  - `prompts/video/{system_prompt.md,user_prompt.md}`
- `camie-tagger-v2-metadata.json` is a large model metadata asset; only update it when syncing model data.

## Build, Test, and Development Commands
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python cli.py
```
```bash
./run_tagger.sh /data/videos 4 video-pro
python tag_images_by_wd14_tagger.py /data/images --taggers wd14,pixai --recursive
```
There is no separate build step; scripts are executed directly.

## Coding Style & Naming Conventions
- Target Python 3 with PEP 8 style and 4-space indentation.
- Use `snake_case` for functions/variables/files and `UPPER_SNAKE_CASE` for constants.
- Keep functions small and composable; prefer explicit arguments over hidden globals.
- Use `logging` in pipeline/runtime modules and reserve `print` for interactive CLI UX.

## Testing Guidelines
- No formal test suite is currently configured; add focused tests under `tests/` for new logic.
- Minimum pre-PR validation:
```bash
python -m py_compile cli.py tag_images_by_wd14_tagger.py wd14_utils.py xai_batch_manager.py
python xai_batch_manager.py --help
python tag_images_by_wd14_tagger.py --help
```
- For behavior changes, run a smoke test on a small sample folder and verify generated `.txt` captions.

## Commit & Pull Request Guidelines
- Follow the repository’s existing commit style: concise imperative summaries, often with Conventional Commit prefixes (`feat:`, `fix:`, `chore:`).
- Keep commits scoped to one concern (CLI, tagging engine, prompts, or xAI batch flow).
- PRs should include: objective, commands used for validation, relevant sample output, and any required env vars (`OPENROUTER_API_KEY`, `HF_TOKEN`, `XAI_API_KEY`).
