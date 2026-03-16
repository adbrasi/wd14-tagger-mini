"""MEGA download with automatic flatten and pair preservation.

Downloads from a MEGA shared link using MEGAcmd (mega-get), which downloads
multiple files in parallel natively. Then flattens all files from subfolders
into a single target directory with name conflict resolution.
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


def check_mega_installed() -> bool:
    """Check if MEGAcmd (mega-get) is available."""
    return shutil.which("mega-get") is not None


def install_mega() -> bool:
    """Install MEGAcmd from official MEGA repo."""
    print_info("Installing MEGAcmd...")
    cmds = [
        "sudo mkdir -p /etc/apt/keyrings",
        "curl -fsSL https://mega.nz/linux/repo/xUbuntu_24.04/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/mega.nz.gpg",
        'echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/mega.nz.gpg] https://mega.nz/linux/repo/xUbuntu_24.04/ ./" | sudo tee /etc/apt/sources.list.d/mega.nz.list > /dev/null',
        "sudo apt update -qq",
        "sudo apt install -y megacmd",
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            # Try Debian 12 repo as fallback
            if "xUbuntu_24.04" in cmd:
                fallback = cmd.replace("xUbuntu_24.04", "Debian_12")
                result = subprocess.run(fallback, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    print_error(f"Failed: {cmd}")
                    print_error(result.stderr[:500])
                    subprocess.run(
                        "sudo rm -f /etc/apt/sources.list.d/mega.nz.list /etc/apt/keyrings/mega.nz.gpg",
                        shell=True, capture_output=True,
                    )
                    return False
            else:
                print_error(f"Failed: {cmd}")
                subprocess.run(
                    "sudo rm -f /etc/apt/sources.list.d/mega.nz.list /etc/apt/keyrings/mega.nz.gpg",
                    shell=True, capture_output=True,
                )
                return False

    # Max speed, no limits
    subprocess.run(["mega-speedlimit", "-d", "0"], capture_output=True)
    subprocess.run(["mega-speedlimit", "-u", "0"], capture_output=True)
    print_success("MEGAcmd installed")
    return True


def mega_download(link: str, local_dir: str) -> bool:
    """Download from MEGA shared link using mega-get (parallel natively)."""
    os.makedirs(local_dir, exist_ok=True)
    print_info(f"Downloading to {local_dir} ...")
    print_info("mega-get downloads multiple files in parallel")

    # mega-get handles folders recursively and in parallel
    result = subprocess.run(
        ["mega-get", link, local_dir],
        timeout=None,
    )
    if result.returncode != 0:
        print_error("MEGA download failed")
        return False

    # Verify files were downloaded
    has_files = False
    for _, _, files in os.walk(local_dir):
        if files:
            has_files = True
            break
    if not has_files:
        print_error("MEGA download completed but no files were found")
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
        # Skip target_dir if inside source_dir
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
