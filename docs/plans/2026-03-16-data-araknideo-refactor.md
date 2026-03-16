# data_araknideo Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rename project to data_araknideo, add video preprocessing (frame cutting), standardize codebase, and optimize for large batches (96GB GPU).

**Architecture:** Incremental refactor — rename first, then standardize code, then add preprocessing menu, then optimize batches. No new files except the preprocessing module. ffmpeg for video manipulation, ProcessPoolExecutor for parallelism.

**Tech Stack:** Python 3, ffmpeg (subprocess), concurrent.futures, existing ONNX/PyTorch/OpenRouter stack.

---

## Decision Log

| Decision | Alternatives | Why |
|----------|-------------|-----|
| ffmpeg for frame cutting | OpenCV frame-by-frame write | ffmpeg is faster, preserves codecs/audio, single subprocess call per video |
| In-place video modification | Output to separate folder | User explicitly chose option A (modify originals) |
| ProcessPoolExecutor for ffmpeg | asyncio subprocess | Simpler, ffmpeg is CPU-bound, no async complexity needed |
| Single preprocessing module | Separate script | User wants it as a menu option in cli.py, not a separate entrypoint |
| Prompts from files (already works) | Database/config | YAGNI — files in prompts/ already auto-load, just verify and document |

---

### Task 1: Rename GitHub repo and local directory

**Files:**
- Modify: `cli.py:35-39` (banner)
- Modify: `README.md:1` (title)
- Modify: `AGENTS.md:1-10` (references)
- Modify: `run_tagger.sh` (if any internal refs)

**Step 1: Rename GitHub repo**

```bash
gh repo rename data_araknideo
```

**Step 2: Update git remote URL**

```bash
cd /home/adolfocesar/projects/wd14-tagger-mini
git remote set-url origin https://github.com/adbrasi/data_araknideo.git
```

**Step 3: Update banner in cli.py**

Change lines 35-39:
```python
def print_banner():
    print("\n" + "=" * 60)
    print("  DATA ARAKNIDEO")
    print("  dataset preprocessing & tagging pipeline")
    print("=" * 60 + "\n")
```

**Step 4: Update README.md title**

```markdown
# data_araknideo
```
Keep the rest of the description.

**Step 5: Update AGENTS.md references**

Replace "wd14-tagger-mini" references with "data_araknideo" where appropriate.

**Step 6: Rename local directory**

```bash
mv /home/adolfocesar/projects/wd14-tagger-mini /home/adolfocesar/projects/data_araknideo
```

**Step 7: Commit**

```bash
git add -A
git commit -m "chore: rename project to data_araknideo"
```

---

### Task 2: Standardize codebase — naming and style consistency

**Files:**
- Modify: `cli.py` (function naming, organization)
- Modify: `tag_images_by_wd14_tagger.py` (naming consistency)
- Modify: `wd14_utils.py` (cleanup backward compat alias)

**Step 1: Audit and fix naming inconsistencies**

Known issues to fix:
- `recommend_batch_by_vram()` at line 2350 of tagger — currently maxes out at batch 16 for 30GB+. Needs to scale to 96GB.
- `extract_first_frame()` backward compat alias in wd14_utils.py:175 — remove dead code.
- Mixed `camelCase` and `snake_case` if found.
- README.md says default model is `x-ai/grok-2-vision-1212` but code says `x-ai/grok-4.1-fast` — fix README.

**Step 2: Fix recommend_batch_by_vram() for large GPUs**

```python
def recommend_batch_by_vram() -> Optional[int]:
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            return None
        free_mb = int(result.stdout.strip().splitlines()[0])
    except Exception:
        return None

    if free_mb >= 80000:
        return 64
    if free_mb >= 60000:
        return 48
    if free_mb >= 40000:
        return 32
    if free_mb >= 30000:
        return 16
    if free_mb >= 20000:
        return 8
    if free_mb >= 12000:
        return 4
    return 2
```

**Step 3: Remove dead backward compat alias**

Remove `extract_first_frame()` from wd14_utils.py — it is not imported anywhere.

**Step 4: Fix README model reference**

Change `x-ai/grok-2-vision-1212` to `x-ai/grok-4.1-fast` in README.md.

**Step 5: Validate**

```bash
python -m py_compile cli.py tag_images_by_wd14_tagger.py wd14_utils.py
```

**Step 6: Commit**

```bash
git add cli.py tag_images_by_wd14_tagger.py wd14_utils.py README.md
git commit -m "refactor: standardize naming, fix VRAM scaling, remove dead code"
```

---

### Task 3: Add video preprocessing — frame cutting with ffmpeg

**Files:**
- Modify: `cli.py` (add preprocessing menu option)
- Modify: `wd14_utils.py` (add ffmpeg frame cutting function)

**Step 1: Add ffmpeg frame cutting function to wd14_utils.py**

```python
import subprocess
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def cut_video_frames(video_path: str, max_frames: int) -> bool:
    """Cut a video to its first max_frames frames, modifying in-place.

    Uses ffmpeg to re-encode with frame limit. Returns True on success.
    """
    tmp_path = video_path + ".tmp" + os.path.splitext(video_path)[1]
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vframes", str(max_frames),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy",
        tmp_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.warning(f"ffmpeg failed for {video_path}: {result.stderr[:200]}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False
        os.replace(tmp_path, video_path)
        return True
    except Exception as e:
        logger.warning(f"Error cutting {video_path}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def cut_videos_batch(video_paths: list, max_frames: int, max_workers: int = None) -> dict:
    """Cut multiple videos in parallel. Returns {success: int, failed: int}."""
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, len(video_paths), 16)

    success = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(cut_video_frames, vp, max_frames): vp
            for vp in video_paths
        }
        for future in as_completed(futures):
            if future.result():
                success += 1
            else:
                failed += 1

    return {"success": success, "failed": failed}
```

**Step 2: Add preprocessing menu to cli.py**

Insert before the current main() flow, after input directory selection:

```python
# After input_dir is resolved and before mode selection:
preprocess = ask_choice("What do you want to do?", [
    "Pre-process dataset (cut frames, normalize)",
    "Tag dataset (wd14 / pixai / grok pipeline)",
], default=2)

if preprocess == 1:
    run_preprocessing_menu(input_dir, python)
    return

# ... existing tagging flow continues
```

The `run_preprocessing_menu()` function:
```python
def run_preprocessing_menu(input_dir: str, python: str):
    """Interactive preprocessing menu."""
    from pathlib import Path

    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

    action = ask_choice("Preprocessing action:", [
        "Cut videos to first N frames",
    ], default=1)

    if action == 1:
        # Find videos
        videos = []
        for root, _, files in os.walk(input_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in VIDEO_EXTS:
                    videos.append(os.path.join(root, f))

        if not videos:
            print(f"[!] No videos found in {input_dir}")
            return

        print(f"[+] Found {len(videos):,} videos")
        max_frames = ask_input("Cut to how many frames?", "5")
        max_frames = int(max_frames)

        workers = ask_input("Parallel workers", str(min(os.cpu_count() or 4, 16)))
        workers = int(workers)

        if not ask_yes_no(f"Cut {len(videos):,} videos to {max_frames} frames (MODIFIES ORIGINALS)?", default=False):
            print("Aborted.")
            return

        # Import and run via venv subprocess to avoid dependency issues
        # Actually, ffmpeg is external - we can run it directly
        print(f"\n[*] Cutting {len(videos):,} videos to {max_frames} frames...")
        # ... parallel ffmpeg execution
```

**Step 3: Validate**

```bash
python -m py_compile cli.py wd14_utils.py
ffmpeg -version  # verify ffmpeg is available
```

**Step 4: Commit**

```bash
git add cli.py wd14_utils.py
git commit -m "feat: add video frame cutting preprocessing"
```

---

### Task 4: Verify prompts are auto-loaded from files

**Files:**
- Verify: `tag_images_by_wd14_tagger.py:834-865` (get_system_prompt, get_user_prompt_template)
- Verify: `prompts/image/` and `prompts/video/` exist with content

**Step 1: Verify existing behavior**

The code at lines 834-865 already:
1. Checks `--grok_system_prompt_file` override
2. Falls back to `prompts/{video|image}/system_prompt.md`
3. Falls back to `prompts/system_prompt.md`
4. Falls back to hardcoded default

This is already the desired behavior — files in `prompts/` are always loaded by default.

**Step 2: Verify prompt files exist and have content**

- `prompts/video/system_prompt.md` — exists, 218 lines
- `prompts/video/user_prompt.md` — exists, 7 lines
- `prompts/image/system_prompt.md` — exists, 113 lines
- `prompts/image/user_prompt.md` — exists, 10 lines

**Status: Already working. No changes needed.**

---

### Task 5: Optimize for large batches (96GB GPU)

**Files:**
- Modify: `tag_images_by_wd14_tagger.py` (batch sizes, VRAM scaling)
- Modify: `cli.py` (default values)

**Step 1: Already covered in Task 2 — recommend_batch_by_vram() scaling**

Updated to support up to 64 batch size for 80GB+ free VRAM.

**Step 2: Increase default grok concurrency for large datasets**

In cli.py, change default grok_concurrency from "8" to "16":
```python
grok_concurrency = ask_input("Grok API concurrency (parallel requests)", "16")
```

**Step 3: Increase default max_workers for image loading**

In tag_images_by_wd14_tagger.py, change `--max_workers` default from 4 to 8:
```python
parser.add_argument("--max_workers", type=int, default=8)
```

**Step 4: Add --suggest_batch auto-trigger for large datasets**

When batch_size is left at default (1) and there are >100 images, auto-suggest:
Already exists at line 2505-2508, just needs to be wired into CLI defaults.

In cli.py, change batch_size default to "auto" and use VRAM recommendation:
```python
batch_size = ask_input("Batch size for local taggers (or 'auto' for VRAM-based)", "auto")
```

**Step 5: Validate**

```bash
python -m py_compile cli.py tag_images_by_wd14_tagger.py
```

**Step 6: Commit**

```bash
git add cli.py tag_images_by_wd14_tagger.py
git commit -m "perf: optimize defaults for large GPU/batch processing"
```

---

## Execution Order

1. Task 1 — Rename (must be first, changes working directory)
2. Task 2 — Standardize (clean foundation before adding features)
3. Task 3 — Frame cutting (new feature)
4. Task 4 — Verify prompts (no-op, just verification)
5. Task 5 — Batch optimization (tuning)
