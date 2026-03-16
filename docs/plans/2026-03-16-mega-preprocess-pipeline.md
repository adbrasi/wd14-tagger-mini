# MEGA Download + Video Preprocessing + xAI Batch Pipeline

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add MEGA download with flatten, video preprocessing (frame cut + resize), pair validation, xAI Batch as default for video with grok-4.20-beta-0309-reasoning, HuggingFace upload, and Rich UI across the entire CLI.

**Architecture:** Modular Python pipeline — each feature is a separate module imported by cli.py. Rich UI layer (ui.py) replaces all print/input calls. MEGAcmd for downloads, ffmpeg for video manipulation. The CLI wizard orchestrates: Download → Flatten → Validate → Preprocess → Tag → Upload.

**Tech Stack:** Python 3, Rich (console UI), MEGAcmd (subprocess), ffmpeg (subprocess), huggingface_hub, requests, concurrent.futures.

---

## Decision Log

| Decision | Alternatives | Why |
|----------|-------------|-----|
| Rich for UI | Ink (JS), Click, Typer | Rich already used in wd14_utils.py logging, pure Python, no new runtime |
| MEGAcmd for downloads | megatools, rclone | Official MEGA tool, C++ SDK, fastest, Pro Lite support |
| Flatten with numeric suffix | Prefix with folder name | User choice — simpler filenames |
| Scale-to-fit + minimal crop for resize | Pure crop, pure scale, pad | User choice — preserves content with minimal distortion |
| F % 8 == 1 snap for frame count | Exact user input | Required for training compatibility |
| W/H multiples of 32 | Multiples of 8/16/64 | Standard for most video training frameworks |
| xAI Batch as default for video | OpenRouter default | No rate limits on processing, 50% cheaper, user preference |
| grok-4.20-beta-0309-reasoning as default model | grok-4-1-fast-non-reasoning | User choice — better quality captions |
| OpenRouter concurrency 32 | 16, 64, 128 | User has $5 = 5 RPS, with ~5-10s requests 32 concurrent is practical max |
| Kill run_tagger.sh | Keep both | Everything in Python, one entry point, Rich UI everywhere |

---

### Task 1: Create ui.py — Rich UI module

**Files:**
- Create: `ui.py`
- Modify: `requirements.txt` (add rich)

**Step 1: Create ui.py with all UI primitives**

```python
"""Rich UI helpers for data_araknideo CLI.

Centralizes all user interaction: prompts, progress bars, panels, tables.
All other modules import from here instead of using print() directly.
"""
import os
import sys
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

console = Console()


def print_banner():
    banner = Text()
    banner.append("DATA ARAKNIDEO\n", style="bold cyan")
    banner.append("dataset preprocessing & tagging pipeline", style="dim")
    console.print(Panel(banner, border_style="cyan", padding=(1, 4)))


def print_section(title: str):
    console.print(f"\n[bold yellow]{'─' * 60}[/]")
    console.print(f"[bold yellow]  {title}[/]")
    console.print(f"[bold yellow]{'─' * 60}[/]\n")


def print_success(msg: str):
    console.print(f"[bold green]✓[/] {msg}")


def print_warning(msg: str):
    console.print(f"[bold yellow]![/] {msg}")


def print_error(msg: str):
    console.print(f"[bold red]✗[/] {msg}")


def print_info(msg: str):
    console.print(f"[dim]→[/] {msg}")


def ask_input(prompt: str, default: str = "") -> str:
    return Prompt.ask(f"[bold]{prompt}[/]", default=default or None) or default


def ask_choice(prompt: str, options: List[str], default: int = 1) -> int:
    console.print(f"\n[bold]{prompt}[/]")
    for i, opt in enumerate(options, 1):
        marker = " [cyan]*[/]" if i == default else ""
        console.print(f"  [bold]{i})[/] {opt}{marker}")
    while True:
        raw = Prompt.ask("Choice", default=str(default))
        try:
            choice = int(raw)
            if 1 <= choice <= len(options):
                return choice
        except ValueError:
            pass
        console.print(f"  [red]Enter a number between 1 and {len(options)}[/]")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    return Confirm.ask(f"[bold]{prompt}[/]", default=default)


def ask_int(prompt: str, default: int = 1, minimum: int = 1) -> int:
    while True:
        val = IntPrompt.ask(f"[bold]{prompt}[/]", default=default)
        if val >= minimum:
            return val
        console.print(f"  [red]Must be >= {minimum}[/]")


def make_progress(**kwargs) -> Progress:
    """General-purpose progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        **kwargs,
    )


def make_download_progress(**kwargs) -> Progress:
    """Download-specific progress bar with transfer speed."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        **kwargs,
    )


def print_summary_table(title: str, rows: List[tuple]):
    """Print a key-value summary table."""
    table = Table(title=title, show_header=False, border_style="dim")
    table.add_column("Key", style="bold")
    table.add_column("Value")
    for key, value in rows:
        table.add_row(key, str(value))
    console.print(table)
```

**Step 2: Add rich to requirements.txt**

Add `rich>=13.0.0` to requirements.txt.

**Step 3: Validate**

```bash
python -m py_compile ui.py
```

**Step 4: Commit**

```bash
git add ui.py requirements.txt
git commit -m "feat: add Rich UI module (ui.py) for styled CLI"
```

---

### Task 2: Create mega_download.py — MEGA download + flatten

**Files:**
- Create: `mega_download.py`

**Step 1: Create mega_download.py**

```python
"""MEGA download with automatic flatten and pair preservation.

Downloads from a MEGA shared link using MEGAcmd, then flattens all files
from subfolders into a single target directory. Handles name conflicts
with numeric suffixes and preserves video+txt / image+txt pairs.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ui import console, print_error, print_info, print_success, print_warning, make_progress

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl"}
MEDIA_EXTS = VIDEO_EXTS | IMAGE_EXTS
PAIR_EXT = ".txt"


def check_megacmd_installed() -> bool:
    return shutil.which("mega-get") is not None


def install_megacmd():
    """Attempt to install MEGAcmd on Debian/Ubuntu."""
    console.print("[bold]Installing MEGAcmd...[/]")
    cmds = [
        "sudo mkdir -p /etc/apt/keyrings",
        'curl -fsSL https://mega.nz/linux/repo/Debian_12/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/mega.nz.gpg',
        'echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/mega.nz.gpg] https://mega.nz/linux/repo/Debian_12/ ./" | sudo tee /etc/apt/sources.list.d/mega.nz.list > /dev/null',
        "sudo apt update -qq",
        "sudo apt install -y megacmd",
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print_error(f"Failed: {cmd}")
            print_error(result.stderr[:500])
            return False
    # Remove speed limits
    subprocess.run(["mega-speedlimit", "-d", "0"], capture_output=True)
    subprocess.run(["mega-speedlimit", "-u", "0"], capture_output=True)
    print_success("MEGAcmd installed successfully")
    return True


def mega_login(email: str, password: str) -> bool:
    """Login to MEGA. Session persists across calls."""
    # Check existing session
    result = subprocess.run(["mega-whoami"], capture_output=True, text=True, timeout=30)
    if result.returncode == 0 and email.lower() in result.stdout.lower():
        print_success(f"Already logged in as {email}")
        return True

    result = subprocess.run(
        ["mega-login", email, password],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode == 0:
        # Max speed
        subprocess.run(["mega-speedlimit", "-d", "0"], capture_output=True)
        print_success(f"Logged in as {email}")
        return True
    print_error(f"Login failed: {result.stderr[:300]}")
    return False


def mega_download(link: str, local_dir: str) -> bool:
    """Download from MEGA shared link to local directory."""
    os.makedirs(local_dir, exist_ok=True)
    print_info(f"Downloading to {local_dir} ...")
    print_info("(MEGAcmd handles progress internally)")

    result = subprocess.run(
        ["mega-get", link, local_dir],
        timeout=None,  # No timeout for large downloads
    )
    if result.returncode != 0:
        print_error("MEGA download failed")
        return False
    print_success("Download complete")
    return True


def _unique_path(target_dir: str, filename: str) -> str:
    """Generate unique filename with numeric suffix if needed."""
    stem = Path(filename).stem
    ext = Path(filename).suffix
    candidate = os.path.join(target_dir, filename)
    counter = 2
    while os.path.exists(candidate):
        candidate = os.path.join(target_dir, f"{stem}_{counter}{ext}")
        counter += 1
    return candidate


def flatten_directory(source_dir: str, target_dir: str) -> dict:
    """Move all media files + their .txt pairs from subfolders into target_dir.

    Returns stats: {moved: int, conflicts: int, pairs: int}
    """
    os.makedirs(target_dir, exist_ok=True)
    stats = {"moved": 0, "conflicts": 0, "pairs": 0}

    # Collect all media files first
    media_files = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if Path(f).suffix.lower() in MEDIA_EXTS:
                media_files.append(os.path.join(root, f))

    if not media_files:
        print_warning(f"No media files found in {source_dir}")
        return stats

    with make_progress() as progress:
        task = progress.add_task("Flattening files", total=len(media_files))

        for src_path in media_files:
            filename = os.path.basename(src_path)
            dst_path = _unique_path(target_dir, filename)
            actual_name = os.path.basename(dst_path)

            if actual_name != filename:
                stats["conflicts"] += 1

            # Move media file
            shutil.move(src_path, dst_path)
            stats["moved"] += 1

            # Move paired .txt if exists
            txt_src = os.path.splitext(src_path)[0] + PAIR_EXT
            if os.path.exists(txt_src):
                txt_dst = os.path.splitext(dst_path)[0] + PAIR_EXT
                shutil.move(txt_src, txt_dst)
                stats["pairs"] += 1

            progress.advance(task)

    return stats
```

**Step 2: Validate**

```bash
python -m py_compile mega_download.py
```

**Step 3: Commit**

```bash
git add mega_download.py
git commit -m "feat: add MEGA download module with flatten support"
```

---

### Task 3: Create video_preprocess.py — frame cut + resize

**Files:**
- Create: `video_preprocess.py`

**Step 1: Create video_preprocess.py**

```python
"""Video preprocessing: frame cutting and resolution normalization.

- Frame count snaps to F % 8 == 1 (1, 9, 17, 25, 33, 41, 49, ...)
- Width/Height snapped to multiples of 32 (scale-to-fit, minimal crop)
- Uses ffmpeg for all operations
- Parallel processing with ProcessPoolExecutor
"""
import math
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

# NOTE: These functions run in subprocess workers — they must NOT import Rich
# (Rich Console is not fork-safe). Logging is returned as results instead.

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


def snap_frames(n: int) -> int:
    """Snap frame count to nearest value satisfying F % 8 == 1.

    Valid values: 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, ...
    """
    if n <= 1:
        return 1
    # Find nearest F where F % 8 == 1
    lower = ((n - 1) // 8) * 8 + 1
    upper = lower + 8
    if abs(n - lower) <= abs(n - upper):
        return lower
    return upper


def snap_dimension(d: int) -> int:
    """Snap dimension to nearest multiple of 32."""
    return max(32, round(d / 32) * 32)


def get_video_info(video_path: str) -> Optional[dict]:
    """Get video width, height, frame count via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
        import json
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        fmt = data.get("format", {})

        w = int(stream.get("width", 0))
        h = int(stream.get("height", 0))

        # nb_frames can be "N/A" for some containers
        nb = stream.get("nb_frames", "N/A")
        if nb == "N/A" or not nb:
            # Estimate from duration * fps
            duration = float(fmt.get("duration", 0))
            fps_str = stream.get("r_frame_rate", "24/1")
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) else 24.0
            nb = int(duration * fps) if duration else 0
        else:
            nb = int(nb)

        return {"width": w, "height": h, "frames": nb}
    except Exception:
        return None


def process_single_video(
    video_path: str,
    max_frames: Optional[int],
    target_w: Optional[int],
    target_h: Optional[int],
) -> dict:
    """Process a single video: cut frames and/or resize.

    Returns dict with status info (no Rich imports — subprocess safe).
    """
    result = {"path": video_path, "ok": False, "detail": ""}
    tmp_path = video_path + ".tmp" + os.path.splitext(video_path)[1]

    filters = []
    output_args = []

    # Build ffmpeg filter chain
    if target_w and target_h:
        # Scale to fit within target_w x target_h, then crop to exact dimensions
        filters.append(
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
            f"crop={target_w}:{target_h}"
        )

    cmd = ["ffmpeg", "-y", "-i", video_path]

    if max_frames:
        cmd.extend(["-vframes", str(max_frames)])

    if filters:
        cmd.extend(["-vf", ",".join(filters)])

    cmd.extend([
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy",
        tmp_path,
    ])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            result["detail"] = proc.stderr[:200]
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return result
        os.replace(tmp_path, video_path)
        result["ok"] = True
        return result
    except Exception as e:
        result["detail"] = str(e)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return result


def preprocess_videos(
    video_paths: list,
    max_frames: Optional[int] = None,
    resize: bool = False,
    max_workers: int = None,
) -> dict:
    """Preprocess multiple videos in parallel.

    Args:
        video_paths: list of video file paths
        max_frames: if set, snap to F%8==1 and cut. None = skip cutting.
        resize: if True, snap W/H to multiples of 32
        max_workers: parallel workers (default: min(cpu_count, 16))

    Returns: {total, success, failed, skipped, details: [{path, ok, detail}]}
    """
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, len(video_paths), 16)

    snapped_frames = snap_frames(max_frames) if max_frames else None

    # Pre-scan videos for dimensions if resize is needed
    targets = {}  # path -> (target_w, target_h, needs_cut)
    for vp in video_paths:
        tw, th = None, None
        if resize:
            info = get_video_info(vp)
            if info and info["width"] and info["height"]:
                tw = snap_dimension(info["width"])
                th = snap_dimension(info["height"])
                # Skip resize if already aligned
                if tw == info["width"] and th == info["height"]:
                    tw, th = None, None

        needs_work = snapped_frames is not None or (tw is not None)
        targets[vp] = (tw, th, needs_work)

    to_process = [(vp, tw, th) for vp, (tw, th, needs) in targets.items() if needs]
    skipped = len(video_paths) - len(to_process)

    stats = {"total": len(video_paths), "success": 0, "failed": 0, "skipped": skipped, "details": []}

    if not to_process:
        return stats

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_video, vp, snapped_frames, tw, th): vp
            for vp, tw, th in to_process
        }
        for future in as_completed(futures):
            res = future.result()
            stats["details"].append(res)
            if res["ok"]:
                stats["success"] += 1
            else:
                stats["failed"] += 1

    return stats
```

**Step 2: Validate**

```bash
python -m py_compile video_preprocess.py
```

**Step 3: Commit**

```bash
git add video_preprocess.py
git commit -m "feat: add video preprocessing (frame cut + resize with snap)"
```

---

### Task 4: Create dataset_validate.py — pair validation

**Files:**
- Create: `dataset_validate.py`

**Step 1: Create dataset_validate.py**

```python
"""Dataset validation: check media/txt pairs and report orphans.

Scans a directory for video+txt and image+txt pairs. Reports:
- Media files without .txt (uncaptioned)
- .txt files without media (orphans)

Offers interactive options to fix issues.
"""
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl"}
MEDIA_EXTS = VIDEO_EXTS | IMAGE_EXTS


def scan_pairs(directory: str, recursive: bool = True) -> dict:
    """Scan directory for media/txt pairs.

    Returns:
        {
            "media_with_txt": [(media_path, txt_path), ...],
            "media_without_txt": [media_path, ...],
            "txt_without_media": [txt_path, ...],
            "total_media": int,
            "total_txt": int,
        }
    """
    media_files: Set[str] = set()  # stem -> full path
    txt_files: Set[str] = set()
    media_map: Dict[str, str] = {}  # stem -> media path
    txt_map: Dict[str, str] = {}    # stem -> txt path

    walker = os.walk(directory) if recursive else [(directory, [], os.listdir(directory))]

    for root, _, files in walker:
        for f in files:
            full = os.path.join(root, f)
            ext = Path(f).suffix.lower()
            stem = os.path.splitext(full)[0]

            if ext in MEDIA_EXTS:
                media_files.add(stem)
                media_map[stem] = full
            elif ext == ".txt":
                txt_files.add(stem)
                txt_map[stem] = full

    paired_stems = media_files & txt_files
    media_only = media_files - txt_files
    txt_only = txt_files - media_files

    return {
        "media_with_txt": [(media_map[s], txt_map[s]) for s in sorted(paired_stems)],
        "media_without_txt": [media_map[s] for s in sorted(media_only)],
        "txt_without_media": [txt_map[s] for s in sorted(txt_only)],
        "total_media": len(media_files),
        "total_txt": len(txt_files),
    }


def delete_files(file_list: List[str]) -> int:
    """Delete files and return count of successfully deleted."""
    deleted = 0
    for f in file_list:
        try:
            os.remove(f)
            deleted += 1
        except OSError:
            pass
    return deleted
```

**Step 2: Validate**

```bash
python -m py_compile dataset_validate.py
```

**Step 3: Commit**

```bash
git add dataset_validate.py
git commit -m "feat: add dataset pair validation module"
```

---

### Task 5: Rewrite cli.py — full Rich wizard with all modules

**Files:**
- Modify: `cli.py` (full rewrite)

This is the largest task. The new cli.py orchestrates all modules in a wizard flow:

1. Banner
2. Ask: data source (local / HuggingFace / MEGA)
3. If MEGA: download + flatten
4. Validate pairs (ask about orphans)
5. Ask: what to do (preprocess / tag / full pipeline)
6. If preprocess: frame cut + resize
7. If tag: tagger selection, grok config, batch settings
8. Run tagging pipeline
9. Ask: upload to HuggingFace?

**Key changes from current cli.py:**
- All print()/input() → Rich UI (from ui.py)
- MEGA download integration
- Pair validation before any processing
- Preprocessing integrated into wizard
- xAI Batch as default for video (model: grok-4.20-beta-0309-reasoning)
- OpenRouter concurrency default 32
- HuggingFace upload at end
- Batch size "auto" using VRAM detection
- run_tagger.sh deleted (everything in Python)

**The full cli.py code is too large for the plan — implement by adapting the current cli.py with these specific changes:**

**Step 1: Replace all imports and constants at top**

Replace current imports with:
```python
#!/usr/bin/env python3
"""Interactive CLI for data_araknideo.

Full pipeline wizard: download → validate → preprocess → tag → upload.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import zipfile

import requests

from ui import (
    ask_choice, ask_input, ask_int, ask_yes_no, console,
    make_progress, print_banner, print_error, print_info,
    print_section, print_success, print_summary_table, print_warning,
)
```

Update constants:
```python
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(SCRIPT_DIR, ".venv")
REQUIREMENTS = os.path.join(SCRIPT_DIR, "requirements.txt")
TAGGER_SCRIPT = os.path.join(SCRIPT_DIR, "tag_images_by_wd14_tagger.py")
XAI_API_BASE_URL = "https://api.x.ai"
XAI_BATCH_DEFAULT_MODEL = "grok-4.20-beta-0309-reasoning"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
```

**Step 2: Add MEGA download section to main() after input resolution**

After the existing HF detection block, add MEGA detection:
```python
# Detect MEGA link
if raw_input.startswith("https://mega.nz/"):
    from mega_download import (
        check_megacmd_installed, install_megacmd,
        mega_download, flatten_directory,
    )
    if not check_megacmd_installed():
        print_warning("MEGAcmd not installed")
        if ask_yes_no("Install MEGAcmd now?"):
            if not install_megacmd():
                print_error("Could not install MEGAcmd")
                sys.exit(1)
        else:
            sys.exit(1)

    default_dl = os.path.join(os.path.expanduser("~"), "datasets", "mega_download")
    local_dir = ask_input("Download to (temp folder)", default_dl)

    if not mega_download(raw_input, local_dir):
        sys.exit(1)

    # Flatten
    flat_dir = ask_input("Flatten all files to", local_dir + "_flat")
    print_section("FLATTENING")
    stats = flatten_directory(local_dir, flat_dir)
    print_success(f"Moved {stats['moved']:,} files ({stats['conflicts']:,} name conflicts resolved, {stats['pairs']:,} txt pairs preserved)")
    input_dir = flat_dir
```

**Step 3: Add pair validation section**

After input_dir is resolved (for any source), add:
```python
# Validate media/txt pairs
from dataset_validate import scan_pairs, delete_files

print_section("DATASET VALIDATION")
pairs = scan_pairs(input_dir, recursive=True)
print_info(f"Media files: {pairs['total_media']:,}")
print_info(f"Text files: {pairs['total_txt']:,}")
print_info(f"Paired: {len(pairs['media_with_txt']):,}")

if pairs["media_without_txt"]:
    print_warning(f"{len(pairs['media_without_txt']):,} media files WITHOUT captions")
    action = ask_choice("What to do with uncaptioned media?", [
        f"Delete {len(pairs['media_without_txt']):,} uncaptioned files",
        "Keep them (will be captioned during tagging)",
        "Show file list first",
    ], default=2)
    if action == 1:
        deleted = delete_files(pairs["media_without_txt"])
        print_success(f"Deleted {deleted:,} uncaptioned media files")
    elif action == 3:
        for p in pairs["media_without_txt"][:20]:
            print_info(p)
        if len(pairs["media_without_txt"]) > 20:
            print_info(f"... and {len(pairs['media_without_txt']) - 20:,} more")
        if ask_yes_no(f"Delete all {len(pairs['media_without_txt']):,}?", default=False):
            deleted = delete_files(pairs["media_without_txt"])
            print_success(f"Deleted {deleted:,} files")

if pairs["txt_without_media"]:
    print_warning(f"{len(pairs['txt_without_media']):,} orphan .txt files (no matching media)")
    if ask_yes_no(f"Delete {len(pairs['txt_without_media']):,} orphan .txt files?", default=False):
        deleted = delete_files(pairs["txt_without_media"])
        print_success(f"Deleted {deleted:,} orphan .txt files")
```

**Step 4: Add preprocessing section**

The existing preprocessing menu gets expanded:
```python
# Preprocessing
workflow = ask_choice("What do you want to do?", [
    "Pre-process dataset (cut frames, resize)",
    "Tag dataset (wd14 / pixai / grok pipeline)",
    "Full pipeline (preprocess → tag → upload)",
], default=3)

if workflow in (1, 3):
    from video_preprocess import snap_frames, preprocess_videos, get_video_info

    videos = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in VIDEO_EXTS:
                videos.append(os.path.join(root, f))

    if videos:
        print_section("VIDEO PREPROCESSING")
        print_info(f"Found {len(videos):,} videos")

        do_cut = ask_yes_no("Cut videos to max frame count?", default=True)
        max_frames = None
        if do_cut:
            raw_frames = ask_int("Max frames", default=49, minimum=1)
            max_frames = snap_frames(raw_frames)
            if max_frames != raw_frames:
                print_info(f"Snapped {raw_frames} → {max_frames} (F % 8 == 1 rule)")

        do_resize = ask_yes_no("Resize to multiples of 32 (W/H)?", default=True)

        workers = min(os.cpu_count() or 4, 16)

        if do_cut or do_resize:
            if not ask_yes_no(
                f"Process {len(videos):,} videos (cut={max_frames}, resize={do_resize}) — MODIFIES ORIGINALS?",
                default=False
            ):
                print_warning("Preprocessing skipped")
            else:
                print_section("PROCESSING")
                result = preprocess_videos(videos, max_frames=max_frames, resize=do_resize, max_workers=workers)
                print_success(f"Done: {result['success']:,} ok, {result['failed']:,} failed, {result['skipped']:,} skipped")
    else:
        print_info("No videos found — skipping preprocessing")

    if workflow == 1:
        print_success("Preprocessing complete")
        # Ask about HF upload at end
```

**Step 5: Update tagging defaults**

- `XAI_BATCH_DEFAULT_MODEL = "grok-4.20-beta-0309-reasoning"`
- Default grok_concurrency = "32" (OpenRouter)
- xAI Batch as default provider for video mode
- Batch size default = "auto"

**Step 6: Add HuggingFace upload at end of pipeline**

After tagging completes:
```python
# HuggingFace upload
if ask_yes_no("Upload dataset to HuggingFace?", default=False):
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or ""
    if not hf_token:
        hf_token = ask_input("HuggingFace token")

    repo_name = ask_input("Repository name (user/dataset)", "")
    private = ask_yes_no("Private repository?", default=True)

    print_section("UPLOADING TO HUGGINGFACE")
    # Use huggingface_hub upload_folder via venv subprocess
    upload_script = (
        "from huggingface_hub import HfApi\n"
        "import sys, os\n"
        "api = HfApi(token=os.environ['HF_TOKEN'])\n"
        "api.create_repo(sys.argv[1], repo_type='dataset', private=sys.argv[3]=='true', exist_ok=True)\n"
        "api.upload_folder(folder_path=sys.argv[2], repo_id=sys.argv[1], repo_type='dataset')\n"
        "print('__done__')\n"
    )
    env = os.environ.copy()
    env["HF_TOKEN"] = hf_token
    result = subprocess.run(
        [python, "-c", upload_script, repo_name, input_dir, "true" if private else "false"],
        env=env,
    )
    if result.returncode == 0:
        print_success(f"Uploaded to https://huggingface.co/datasets/{repo_name}")
    else:
        print_error("Upload failed")
```

**Step 7: Delete run_tagger.sh**

```bash
rm run_tagger.sh
```

**Step 8: Validate and commit**

```bash
python -m py_compile cli.py
git add cli.py
git rm run_tagger.sh
git commit -m "feat: rewrite CLI with Rich UI, MEGA download, preprocessing, HF upload"
```

---

### Task 6: Update tag_images_by_wd14_tagger.py defaults

**Files:**
- Modify: `tag_images_by_wd14_tagger.py`

**Step 1: Update default model constants**

```python
# Line ~807-808
DEFAULT_GROK_MODEL = "x-ai/grok-4.20-beta-0309-reasoning"
DEFAULT_XAI_BATCH_MODEL = "grok-4.20-beta-0309-reasoning"
```

**Step 2: Update default concurrency**

```python
# In setup_parser(), --grok_concurrency
parser.add_argument("--grok_concurrency", type=int, default=32)
```

**Step 3: Validate and commit**

```bash
python -m py_compile tag_images_by_wd14_tagger.py
git add tag_images_by_wd14_tagger.py
git commit -m "chore: update default model to grok-4.20-beta-0309-reasoning, concurrency 32"
```

---

### Task 7: Update README.md and cleanup

**Files:**
- Modify: `README.md`

**Step 1: Update README**

Update model references, add MEGA download section, add preprocessing section, remove run_tagger.sh references, add HuggingFace upload section.

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README with MEGA, preprocessing, upload features"
```

---

## Execution Order

1. Task 1 — ui.py (foundation, everything depends on it)
2. Task 2 — mega_download.py (independent module)
3. Task 3 — video_preprocess.py (independent module)
4. Task 4 — dataset_validate.py (independent module)
5. Task 5 — cli.py rewrite (depends on 1-4)
6. Task 6 — tagger defaults (independent)
7. Task 7 — README + cleanup
