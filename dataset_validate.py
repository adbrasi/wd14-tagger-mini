"""Dataset validation: check media/txt pairs and report orphans.

Scans a directory for video+txt and image+txt pairs. Reports:
- Media files without .txt (uncaptioned)
- .txt files without media (orphans)

Handles multiple media extensions sharing the same stem (e.g., video.mp4 + video.jpg).
"""
import os
from pathlib import Path
from typing import Dict, List, Set

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl"}
MEDIA_EXTS = VIDEO_EXTS | IMAGE_EXTS


def scan_pairs(directory: str, recursive: bool = True) -> dict:
    """Scan directory for media/txt pairs.

    Handles multiple media files sharing the same stem by tracking all of them.

    Returns:
        {
            "media_with_txt": [(media_path, txt_path), ...],
            "media_without_txt": [media_path, ...],
            "txt_without_media": [txt_path, ...],
            "total_media": int,
            "total_txt": int,
        }
    """
    if not os.path.isdir(directory):
        return {
            "media_with_txt": [],
            "media_without_txt": [],
            "txt_without_media": [],
            "total_media": 0,
            "total_txt": 0,
        }

    # Track ALL media files per stem (handles video.mp4 + video.jpg)
    media_by_stem: Dict[str, List[str]] = {}  # stem -> [full_path, ...]
    txt_map: Dict[str, str] = {}  # stem -> txt path
    total_media = 0

    walker = os.walk(directory) if recursive else [(directory, [], os.listdir(directory))]

    for root, _, files in walker:
        for f in files:
            full = os.path.join(root, f)
            ext = Path(f).suffix.lower()
            stem = os.path.splitext(full)[0]

            if ext in MEDIA_EXTS:
                media_by_stem.setdefault(stem, []).append(full)
                total_media += 1
            elif ext == ".txt":
                txt_map[stem] = full

    media_stems = set(media_by_stem.keys())
    txt_stems = set(txt_map.keys())

    paired_stems = media_stems & txt_stems
    media_only_stems = media_stems - txt_stems
    txt_only_stems = txt_stems - media_stems

    # Build result lists — include ALL media files per stem
    media_with_txt = []
    for s in sorted(paired_stems):
        for mp in media_by_stem[s]:
            media_with_txt.append((mp, txt_map[s]))

    media_without_txt = []
    for s in sorted(media_only_stems):
        media_without_txt.extend(media_by_stem[s])

    return {
        "media_with_txt": media_with_txt,
        "media_without_txt": sorted(media_without_txt),
        "txt_without_media": [txt_map[s] for s in sorted(txt_only_stems)],
        "total_media": total_media,
        "total_txt": len(txt_map),
    }


def delete_files(file_list: List[str]) -> int:
    """Delete files and return count of successfully deleted.

    Logs warnings for files that could not be deleted.
    """
    deleted = 0
    failed = 0
    for f in file_list:
        try:
            os.remove(f)
            deleted += 1
        except OSError:
            failed += 1
    if failed > 0:
        import logging
        logging.warning(f"Failed to delete {failed} files (permissions or missing)")
    return deleted
