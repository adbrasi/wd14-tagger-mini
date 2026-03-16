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
