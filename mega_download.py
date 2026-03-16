"""MEGA download with automatic flatten and pair preservation.

Downloads from a MEGA shared link using megatools (megadl), then flattens
all files from subfolders into a single target directory. Handles name
conflicts with numeric suffixes and preserves video+txt / image+txt pairs.

Uses megatools instead of MEGAcmd because it's lighter (no daemon),
installs with a single apt command, and supports --parallel-transfers
for maximum download speed.
"""
import os
import shutil
import subprocess
from pathlib import Path

from ui import console, make_progress, print_error, print_info, print_success, print_warning

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl"}
MEDIA_EXTS = VIDEO_EXTS | IMAGE_EXTS
PAIR_EXT = ".txt"


def check_megatools_installed() -> bool:
    """Check if megatools (megadl) is available."""
    return shutil.which("megadl") is not None


def install_megatools() -> bool:
    """Install megatools via apt."""
    print_info("Installing megatools...")
    result = subprocess.run(
        ["sudo", "apt", "install", "-y", "megatools"],
        capture_output=False,
    )
    if result.returncode != 0 or not check_megatools_installed():
        print_error("megatools installation failed. Install manually: sudo apt install megatools")
        return False
    print_success("megatools installed")
    return True


def mega_download(link: str, local_dir: str) -> bool:
    """Download from MEGA shared link to local directory.

    Uses megadl with max parallel transfers (16) and no speed limit.
    """
    os.makedirs(local_dir, exist_ok=True)
    print_info(f"Downloading to {local_dir} ...")
    print_info("megadl: 16 parallel transfers, no speed limit")

    # megadl downloads to current directory or --path
    result = subprocess.run(
        [
            "megadl",
            "--path", local_dir,
            "--limit-speed", "0",
            "--parallel-transfers", "16",
            link,
        ],
        timeout=None,  # No timeout for large downloads
    )
    if result.returncode != 0:
        print_error("MEGA download failed")
        return False

    # Verify that at least one file was actually downloaded
    has_files = False
    for _, _, files in os.walk(local_dir):
        if files:
            has_files = True
            break
    if not has_files:
        print_error("MEGA download completed but no files were found in target directory")
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

    Returns stats: {moved: int, conflicts: int, pairs: int, failed: int}
    """
    os.makedirs(target_dir, exist_ok=True)
    stats = {"moved": 0, "conflicts": 0, "pairs": 0, "failed": 0}

    # Collect all media files first
    media_files = []
    for root, _, files in os.walk(source_dir):
        # Skip target_dir if it's inside source_dir
        if os.path.abspath(root).startswith(os.path.abspath(target_dir) + os.sep):
            continue
        for f in files:
            if Path(f).suffix.lower() in MEDIA_EXTS:
                media_files.append(os.path.join(root, f))

    if not media_files:
        print_warning(f"No media files found in {source_dir}")
        return stats

    with make_progress() as progress:
        task = progress.add_task("Flattening files", total=len(media_files))

        for src_path in media_files:
            try:
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
            except Exception as e:
                stats["failed"] += 1
                print_warning(f"Failed to move {os.path.basename(src_path)}: {e}")

            progress.advance(task)

    return stats
