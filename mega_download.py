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
